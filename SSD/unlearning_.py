import torch, torchmetrics, tqdm, copy, time
from utils import  unlearn_func, ssd_tuning 
from torch.cuda.amp import autocast
import numpy as np
from torch.cuda.amp import GradScaler
from os import makedirs
from os.path import exists
from torch.nn import functional as F
import itertools
from skeletons import ParameterPerturber,ssd_tuning
from src.data.fastmri_datamodule import SliceDataset
from torch.utils.data  import DataLoader 
from src.data.components.fastmri_transform import VarNetDataTransform,EquiSpacedMaskFunc
from src.models.losses.ssim import SSIMLoss
from src.data.components.fastmri_transform import center_crop_to_smallest
from src.utils.evaluate import ssim,nmse,psnr

class SSD():
    def __init__(self,optimizer:torch.optim.Optimizer,
                 model:torch.nn.Module,
                 opt,
                 device,):
        self.optimizer = optimizer
        self.model = model
        self.opt = opt
        self.device = device
        
    def set_model(self,model):
        self.model = model
        self.model.to(self.device)  
    
    def unlearn(self, train_loader, test_loader, forget_loader, eval_loaders=None):
        actual_iters = self.opt.train_iters
        self.opt.train_iters = len(train_loader) + len(forget_loader)
        time_start = time.process_time()
        self.best_model = ssd_tuning(self.model, forget_loader, self.opt.SSDdampening, self.opt.SSDselectwt, train_loader, self.opt.device)
        self.save_files['train_time_taken'] += time.process_time() - time_start
        self.opt.train_iters = actual_iters
        return self.best_model

    def get_save_prefix(self):
        self.unlearn_file_prefix = self.opt.pretrain_file_prefix+'/'+str(self.opt.deletion_size)+'_'+self.opt.unlearn_method+'_'+self.opt.exp_name
        self.unlearn_file_prefix += '_'+str(self.opt.train_iters)+'_'+str(self.opt.k)
        self.unlearn_file_prefix += '_'+str(self.opt.SSDdampening)+'_'+str(self.opt.SSDselectwt)
        return 




if __name__ == "__main__":
    poisoned_model = torch.load('poisoned_model.pth')
    poisoned_model.to('cuda')
    poisoned_model.eval()
    
    transform = VarNetDataTransform(challenge="multicoil",
                                   mask_func=EquiSpacedMaskFunc(0.08, 0.08, 0.08)
                                   )
    forget_data = SliceDataset("path/to/unlearn_data", transform=transform)
    forget_loader = DataLoader(forget_data, batch_size=32, shuffle=True)
    original_data = SliceDataset("path/to/original_data", transform=transform)
    original_loader = DataLoader(original_data, batch_size=32, shuffle=True)
    optimizer = torch.optim.Adam(poisoned_model.parameters(), lr=0.001)
    SSDInstance = SSD(optimizer=optimizer, model=poisoned_model, opt=None, device='cuda')
    
    unlearned_model = SSDInstance.unlearn(train_loader=original_loader, test_loader=None, forget_loader=forget_loader)
    
    ## Evaluate the unlearned model on the forget set, to see metrics:
    unlearned_model.eval()
    l1loss = torch.nn.L1Loss()
    total_loss = 0.0
    num_batches = 0
    total_ssim = 0.0
    total_psnr = 0.0
    total_nmse = 0.0
    
    with tqdm(total=len(forget_loader), desc="Calculating Metrics") as pbar:
        for batch in forget_loader:
            with torch.no_grad():
                output = unlearned_model(batch.masked_kspace, batch.mask, batch.num_low_frequencies)
                target, output = center_crop_to_smallest(batch.target, output)
                
                ssim_val = ssim(
                    gt=target.unsqueeze(1), pred=output.unsqueeze(1), maxval=batch.max_value
                )
                psnr_val = psnr(
                    gt=target.unsqueeze(1), pred=output.unsqueeze(1), maxval=batch.max_value
                )
                nmse_val = nmse(
                    gt=target.unsqueeze(1), pred=output.unsqueeze(1), maxval=batch.max_value
                )
                
                total_ssim += ssim_val.mean().item()
                total_psnr += psnr_val.mean().item()
                total_nmse += nmse_val.mean().item()
                num_batches += 1
                pbar.update(1)
            
    avg_ssim = total_ssim / num_batches
    avg_psnr = total_psnr / num_batches
    avg_nmse = total_nmse / num_batches
    
    print(f"Average metrics on forget set:")
    print(f"SSIM: {avg_ssim:.4f}")
    print(f"PSNR: {avg_psnr:.4f}")
    print(f"NMSE: {avg_nmse:.4f}")
        
    