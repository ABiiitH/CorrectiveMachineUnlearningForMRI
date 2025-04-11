import torch
from torch.utils.data import DataLoader
from src.models.losses.ssim import SSIMLoss
from src.data.components.fastmri_transform import center_crop_to_smallest
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
# class VarNetSample(NamedTuple):
#     """
#     A sample of masked k-space for variational network reconstruction.


#     Args:
#         masked_kspace: k-space after applying sampling mask.
#         mask: The applied sampling mask.
#         num_low_frequencies: The number of samples for the densely-sampled
#             center.
#         target: The target image (if applicable).
#         fname: File name.
#         slice_num: The slice index.
#         max_value: Maximum image value.
#         crop_size: The size to crop the final image.
#     """

#     masked_kspace: torch.Tensor
#     mask: torch.Tensor
#     num_low_frequencies: Optional[int]
#     target: torch.Tensor
#     fname: str
#     slice_num: int
#     max_value: float
#     crop_size: Tuple[int, int]

class ParameterPerturber(torch.nn.Module):
    def __init__(self,
                 model:torch.nn.Module,
                 device,
                 optimizer: torch.optim.Optimizer,
                 parameters:dict,                 
                 ):
        super(ParameterPerturber, self).__init__()
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.parameters = parameters
    def __zero_params__(self):  
        return{
        k:torch.zeros_like(params) for k,params in self.model.named_parameters() 
        }
    def calc_importance(self,
                        dataloader:DataLoader):
        criterion = SSIMLoss()
        importances = self.__zero_params__()
        with tqdm(total=len(dataloader),desc="Calculating importance") as pbar:
            for batch in dataloader:
                ## batch is a VarnetSample
                self.optimizer.zero_grad()
                output = self.model(batch.masked_kspace, batch.mask, batch.num_low_frequencies)
                target, output = center_crop_to_smallest(batch.target, output)
                loss = self.ssimloss(
                    output.unsqueeze(1), target.unsqueeze(1), data_range=batch.max_value
                ) + 1e-5 * self.l1loss(output.unsqueeze(1), target.unsqueeze(1))
                loss.backward()
                for (k,p),(k2,imp) in zip(
                    self.model.named_parameters(), importances.items()
                ):
                    if p.grad is None:
                        continue
                    imp += p.grad.data.clone().pow(2)

            ## normalizing importances
            for _, imp in importances.items():
                imp.data /= float(len(dataloader))
        return importances  
    def modify_weight(self,
                      original_importance:List[Dict[str, torch.Tensor]],
                      new_importance:List[Dict[str, torch.Tensor]]):
        with torch.no_grad():
            for (n,p), (oimp_n, oimp), (fimp_n, fimp) in zip(
                self.model.named_parameters(), original_importance.items(), new_importance.items()
            ):
                # Synapse Selection with parameter alpha
                oimp_norm = oimp.mul(self.parameters["selection_weighting"])
                locations = torch.where(fimp > oimp_norm)
                # Synapse Dampening with parameter lambda
                weight = ((oimp.mul(self.parameters["dampening_constant"])).div(fimp)).pow(
                    self.exponent
                )
                update = weight[locations]
                # Bound by 1 to prevent parameter values to increase.
                min_locs = torch.where(update > self.lower_bound)
                update[min_locs] = self.lower_bound
                p[locations] = p[locations].mul(update)
def ssd_tuning(
    model,
    forget_train_dl,
    dampening_constant,
    selection_weighting,
    full_train_dl,
    device,
):
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

    # load the trained model
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    pdr = ParameterPerturber(model, optimizer, device, parameters)
    model = model.eval()

    sample_importances = pdr.calc_importance(forget_train_dl)

    original_importances = pdr.calc_importance(full_train_dl)
    pdr.modify_weight(original_importances, sample_importances)
    return model




            
    