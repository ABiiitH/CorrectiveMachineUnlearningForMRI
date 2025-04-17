import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from tqdm import tqdm

import copy
import time
import itertools
import numpy as np

# Import your custom modules
from src.data.fastmri_datamodule import SliceDataset
from src.data.components.fastmri_transform import VarNetDataTransform, EquiSpacedMaskFunc, center_crop_to_smallest
from src.models.losses.ssim import SSIMLoss
from src.utils.evaluate import ssim, nmse, psnr
from src.models.components.varnet import VarNet

################################################################################
# 1) ParameterPerturber and ssd_tuning logic
################################################################################
class ParameterPerturber(torch.nn.Module):
    def __init__(
        self,
        model: torch.nn.Module,
        device,
        optimizer: torch.optim.Optimizer,
        parameters: dict,
    ):
        super(ParameterPerturber, self).__init__()
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.parameters = parameters

        # For clarity:
        self.l1loss = torch.nn.L1Loss()
        self.ssimloss = SSIMLoss().to(self.device)

        # These might be used in modify_weight
        self.lower_bound = parameters.get("lower_bound", 1)
        self.exponent = parameters.get("exponent", 1)

    def __zero_params__(self):
        return {
            k: torch.zeros_like(params)
            for k, params in self.model.named_parameters()
        }

    def calc_importance(self, dataloader: DataLoader,accum_steps:int=4):
        """
        Calculates importance for each parameter by accumulating the squared
        gradients across the data in the dataloader.
        """
        criterion = SSIMLoss()
        importances = self.__zero_params__()

        self.model.eval()
        with torch.no_grad():
            # Because we only do backward in the next steps if needed,
            # you could also remove "with torch.no_grad()" if you want real grads.
            pass

        # If you want real gradients, remove torch.no_grad() above and do:
        # self.model.train()
        # Then do the .backward() calls, etc.

        # Example approach:
        self.model.train()
        batch_counter  = 0 
        with tqdm(total=len(dataloader), desc="Calculating Importances") as pbar:
            for batch_idx, batch in enumerate(dataloader):
                self.optimizer.zero_grad(set_to_none=True)
                masked_kspace = batch.masked_kspace.to(self.device, non_blocking=True)
                mask = batch.mask.to(self.device, non_blocking=True)
                num_low_frequencies = batch.num_low_frequencies
                target = batch.target.to(self.device, non_blocking=True)
                maxval = batch.max_value.to(self.device)

                output = self.model(masked_kspace, mask, num_low_frequencies)
                target_crop, output_crop = center_crop_to_smallest(target, output)
                # We compute SSIM + L1
                loss = self.ssimloss(
                    output_crop.unsqueeze(1), target_crop.unsqueeze(1), data_range=maxval
                ) + 1e-5 * self.l1loss(output_crop.unsqueeze(1), target_crop.unsqueeze(1))

                loss.backward()

                # Accumulate squared gradients
                for (k, p), (ik, imp) in zip(self.model.named_parameters(), importances.items()):
                    if p.grad is None:
                        continue
                    imp += p.grad.detach().pow(2)
                
                    torch.cuda.empty_cache()
                batch_counter += 1
                # Optionally release the graph and temporary buffers after a certain number of mini-batches.
                if batch_counter % accum_steps == 0:
                    # A dummy optimizer step or simply clear the gradients
                    self.optimizer.zero_grad(set_to_none=True)
                    # Optionally call torch.cuda.empty_cache() here (though its benefits are limited)
                    torch.cuda.empty_cache()
                pbar.update(1)    
        # Normalize importances by the number of steps
        for _, imp in importances.items():
            imp.data /= float(len(dataloader))
        

        return importances

    def modify_weight(self, original_importances, new_importances):
        """
        Modifies the model weights based on the difference between
        original_importances and new_importances, following your SSD scheme.
        """
        with torch.no_grad():
            for (name, p) in self.model.named_parameters():
                oimp = original_importances[name]
                fimp = new_importances[name]
                # Synapse Selection with parameter alpha
                oimp_norm = oimp.mul(self.parameters["selection_weighting"])
                locations = torch.where(fimp > oimp_norm)

                # Dampening
                weight = (
                    (oimp.mul(self.parameters["dampening_constant"])).div(fimp)
                ).pow(self.exponent)
                update = weight[locations]

                # Bound by 1 to prevent parameter values to *increase*.
                min_locs = torch.where(update > self.lower_bound)
                update[min_locs] = self.lower_bound

                p[locations] = p[locations].mul(update)


def ssd_tuning(
    model: torch.nn.Module,
    forget_train_dl: DataLoader,
    dampening_constant: float,
    selection_weighting: float,
    full_train_dl: DataLoader,
    device: str = "cuda",
):
    """
    The function that performs the synaptic suppression/unlearning steps:
    1) Calculate importances on forget set
    2) Calculate importances on full (original) set
    3) Adjust weights
    """
    parameters = {
        "lower_bound": 1,
        "exponent": 1,
        "magnitude_diff": None,
        "min_layer": -1,
        "max_layer": -1,
        "forget_threshold": 1,
        "dampening_constant": dampening_constant,
        "selection_weighting": selection_weighting,
    }

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    pdr = ParameterPerturber(model, device, optimizer, parameters)

    # Evaluate / get importances
    sample_importances = pdr.calc_importance(forget_train_dl)
    original_importances = pdr.calc_importance(full_train_dl)

    # Modify the weights
    pdr.modify_weight(original_importances, sample_importances)

    return model

################################################################################
# 2) The SSD class
################################################################################
class SSD():
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        model: torch.nn.Module,
        opt,   # Typically some config or argument object
        device: str
    ):
        self.optimizer = optimizer
        self.model = model
        self.opt = opt
        self.device = device
        # Track times, etc.
        self.save_files = {"train_time_taken": 0.0}

    def set_model(self, model):
        self.model = model
        self.model.to(self.device)

    def unlearn(self, train_loader, test_loader, forget_loader, eval_loaders=None):
        actual_iters = self.opt.train_iters
        self.opt.train_iters = len(train_loader) + len(forget_loader)

        time_start = time.process_time()
        self.best_model = ssd_tuning(
            self.model,
            forget_loader,
            self.opt.SSDdampening,
            self.opt.SSDselectwt,
            train_loader,
            self.opt.device,
        )
        self.save_files["train_time_taken"] += time.process_time() - time_start

        self.opt.train_iters = actual_iters
        return self.best_model

    def get_save_prefix(self):
        self.unlearn_file_prefix = (
            self.opt.pretrain_file_prefix
            + "/"
            + str(self.opt.deletion_size)
            + "_"
            + self.opt.unlearn_method
            + "_"
            + self.opt.exp_name
        )
        self.unlearn_file_prefix += "_" + str(self.opt.train_iters) + "_" + str(self.opt.k)
        self.unlearn_file_prefix += "_" + str(self.opt.SSDdampening) + "_" + str(self.opt.SSDselectwt)
        return

################################################################################
# 3) Distributed main
################################################################################
def parse_args():
    """
    Replace or extend this with argparse as needed.
    For simplicity, we’ll assume environment variables or defaults are used.
    """
    import argparse
    parser = argparse.ArgumentParser(description="Distributed SSD Example")
    # You can add arguments like:
    parser.add_argument("--local_rank", type=int, default=-1, help="Local process rank.")
    parser.add_argument("--world_size", type=int, default=1, help="Number of total processes.")
    parser.add_argument("--dist_backend", type=str, default="nccl", help="Distributed backend.")
    parser.add_argument("--dist_url", type=str, default="env://", help="URL for init.")
    args = parser.parse_args()
    return args


def setup_for_distributed(is_master: bool):
    """
    This utility disables tqdm/bar updates for non-master processes to avoid
    console spam.
    """
    import builtins as __builtins__
    import sys

    if not is_master:
        # Disable printing entirely on non-master ranks
        def print_pass(*args, **kwargs):
            pass
        __builtins__.print = print_pass

def main_worker(local_rank: int, world_size: int, args):
    # 1) Initialize process group
    dist.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=world_size,
        rank=local_rank,
    )

    # 2) Set device
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    # 3) (Optional) silence logs if not master
    is_master = (local_rank == 0)
    setup_for_distributed(is_master)

    ############################################################################
    # Model Setup
    ############################################################################
    # Example: building the VarNet and loading the checkpoint
    poisoned_model_statedict = torch.load(
        "/scratch/saigum/CorrectiveMachineUnlearningForMRI/SSD/poisoned.ckpt",
        map_location="cpu",weights_only=False
    )

    # Build model
    model = VarNet(num_cascades= 12,pools= 4,chans=18,sens_pools= 4,sens_chans= 8)
    # Because your ckpt might be in the form `{"state_dict":..., ...}`, we isolate it:
    # If the checkpoint indeed is a dict with "state_dict", adapt accordingly. 
    # Or if it's a plain statedict, skip the indexing.

    # Suppose it's just a plain state dict. If it's actually something else, adjust as needed:
    # e.g. net_state_dict = {
    #    k[len("net."):]: v for k, v in ...
    # }
    # Then do `model.load_state_dict(net_state_dict)`
    if isinstance(poisoned_model_statedict, dict) and "state_dict" in poisoned_model_statedict:
        # Possibly a lightning checkpoint
        full_sd = poisoned_model_statedict["state_dict"]
    else:
        # Possibly a plain state dict
        full_sd = poisoned_model_statedict

    # Example of removing "net." prefix if needed:
    net_state_dict = {k[len("net."):]: v for k, v in full_sd.items() if k.startswith("net.")}
    model.load_state_dict(net_state_dict, strict=False)

    model.to(device)
    # Wrap with DDP
    ddp_model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    ############################################################################
    # Data Setup (Distributed Samplers)
    ############################################################################
    transform = VarNetDataTransform(
        challenge="multicoil",
        mask_func=EquiSpacedMaskFunc(center_fractions=[0.08], accelerations=[8]),
    )

    forget_data = SliceDataset("data/forgetSet", transform=transform,challenge="multicoil")
    original_data = SliceDataset("data/original_train", transform=transform,challenge="multicoil")

    # Create distributed samplers
    forget_sampler = DistributedSampler(forget_data, num_replicas=world_size, rank=local_rank, shuffle=True)
    original_sampler = DistributedSampler(original_data, num_replicas=world_size, rank=local_rank, shuffle=True)

    # Create DataLoaders
    forget_loader = DataLoader(forget_data, batch_size=1, sampler=forget_sampler)  # Reduced from 32
    original_loader = DataLoader(original_data, batch_size=1, sampler=original_sampler)  # Reduced from 32

    # Example of a minimal "opt" object
    class MyOpts:
        def __init__(self):
            self.SSDdampening = 0.1
            self.SSDselectwt = 0.1
            self.train_iters = 50
            self.device = device  # or "cuda"
            self.pretrain_file_prefix = "whatever"
            self.deletion_size = 30
            self.unlearn_method = "SSD"
            self.exp_name = "example"
            self.k = 5
            self.gradient_accumulation_steps = 4  # New parameter

    opt = MyOpts()

    # Create your SSD instance
    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=0.001)
    ssd_instance = SSD(
        optimizer=optimizer,
        model=ddp_model,  # DDP model
        opt=opt,    
        device=device,
    )

    ############################################################################
    # Perform the "unlearn" step
    ############################################################################
    # Typically, you only need to do ssd_tuning once. If you want to coordinate
    # model updates across ranks, you can do so just by calling ssd_instance.unlearn
    # on each rank. DDP ensures the gradients and parameters are synchronized.

    unlearned_model = ssd_instance.unlearn(
        train_loader=original_loader, 
        test_loader=None, 
        forget_loader=forget_loader,
    )
    forget_percent= (100*len(forget_loader.dataset))//len(original_loader.dataset)
    print(f"Unlearning completed. Forget percent: {forget_percent}%")
    torch.save(obj=unlearned_model.state_dict(),f=f"unlearned_model_correct_{forget_percent}.pth")
    dist.destroy_process_group()

def main():
    args = parse_args()
    local_rank = args.local_rank

    # If using `torchrun --standalone --nproc_per_node=NUM_GPUS your_script.py`
    # then local_rank is automatically set
    # If you want mp.spawn manually, you can do something like:
    # world_size = torch.cuda.device_count()
    # mp.spawn(main_worker, args=(world_size, args), nprocs=world_size)
    #
    # For torchrun, just call main_worker directly:
    # We'll read the total world size from the environment:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    # world_size = int(os.environ.get("WORLD_SIZE", 1))

    main_worker(local_rank, world_size, args)

if __name__ == "__main__":
    main()
