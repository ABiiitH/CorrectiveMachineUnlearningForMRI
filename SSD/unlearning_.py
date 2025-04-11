import torch, torchmetrics, tqdm, copy, time
from utils import  unlearn_func, ssd_tuning 
from torch.cuda.amp import autocast
import numpy as np
from torch.cuda.amp import GradScaler
from os import makedirs
from os.path import exists
from torch.nn import functional as F
import itertools


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
        return

    def get_save_prefix(self):
        self.unlearn_file_prefix = self.opt.pretrain_file_prefix+'/'+str(self.opt.deletion_size)+'_'+self.opt.unlearn_method+'_'+self.opt.exp_name
        self.unlearn_file_prefix += '_'+str(self.opt.train_iters)+'_'+str(self.opt.k)
        self.unlearn_file_prefix += '_'+str(self.opt.SSDdampening)+'_'+str(self.opt.SSDselectwt)
        return 
