import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import h5py
import os
import pickle
import random
import logging
import math
import sys
from pathlib import Path
from typing import NamedTuple, Dict, Any, Optional, Callable, Union, Tuple
from src.data.components.fastmri_transform_utils import et_query,etree
from src.data.fastmri_datamodule import SliceDataset
# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

# --- Dummy implementations for etree and et_query if not using lxml ---
# These are needed by SliceDataset._retrieve_metadata
# If you have lxml installed, replace these with:
# from lxml import etree
# from fastmri.data.enums import EtAttr, EtTag
# --- SSIM Loss and Helpers ---
def _gaussian(window_size: int, sigma: float):
    vals = [
        math.exp(-((x - window_size//2)**2) / (2 * sigma**2))
        for x in range(window_size)
    ]
    gauss = torch.tensor(vals, dtype=torch.float32)
    return gauss / gauss.sum()

def _create_window(window_size: int, channel: int, device):
    _1D = _gaussian(window_size, sigma=1.5).to(device).unsqueeze(1)
    _2D = _1D @ _1D.t()
    window = _2D.expand(channel, 1, window_size, window_size).contiguous()
    return window

def ssim_loss(img1: torch.Tensor,
              img2: torch.Tensor,
              window_size: int = 11,
              size_average: bool = True,
              val_range: float = 1.0):
    # Ensure input is float and has a channel dimension (B, C, H, W)
    if img1.ndim == 2: # Assuming HxW -> 1x1xHxW
        img1 = img1.unsqueeze(0).unsqueeze(0)
    elif img1.ndim == 3: # Assuming BxHxW -> Bx1xHxW
         img1 = img1.unsqueeze(1)
    if img2.ndim == 2:
        img2 = img2.unsqueeze(0).unsqueeze(0)
    elif img2.ndim == 3:
         img2 = img2.unsqueeze(1)
         
    img1 = img1.float()
    img2 = img2.float()

    device = img1.device
    channel = img1.size(1)
    window = _create_window(window_size, channel, device)

    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)
    mu1_sq, mu2_sq, mu1_mu2 = mu1**2, mu2**2, mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2

    sigma1_sq = torch.relu(sigma1_sq)
    sigma2_sq = torch.relu(sigma2_sq)

    C1 = (0.01 * val_range)**2
    C2 = (0.03 * val_range)**2

    num = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    den = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)

    ssim_map = num / (den + 1e-8)

    if size_average:
        ssim_val = 1 - ssim_map.mean() # Standard SSIM loss is 1 - SSIM
    else:
        ssim_val = 1 - ssim_map.flatten(2).mean(2)

    return ssim_val # Return the loss (1 - SSIM)

# --- Total Variation Loss ---
def total_variation_loss(x):
    # x is expected to be (B, C, H, W) or (B, H, W)
    if x.ndim == 3: # Add channel dim for calculation
        x = x.unsqueeze(1) # (B, 1, H, W)

    # Calculate differences along height and width
    # Ensure dimensions are correct for slicing: x[:, :, 1:, :]
    tv_h = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).sum()
    tv_w = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).sum()
    return tv_h + tv_w

# --- PGD Targeted Attack ---
def pgd_targeted_A(model, images_orig, eps=0.05, alpha=0.03, iters=10, ssim_weight=0.5, tv_weight=1e-4, mask_threshold=0.2):
    """
    Performs Targeted PGD attack to push images towards target_label=0.
    Expects images_orig in (B, C, H, W) format.
    """
    model.eval()
    images_orig = images_orig.clone().detach().to(images_orig.device)
    
    # Ensure brain_mask is applied correctly - assuming images_orig is normalized [0, 1]
    # Mask should have same shape as images_orig
    brain_mask = (images_orig > mask_threshold).float().to(images_orig.device) # Use float mask for multiplication

    # Target label is 0 (for pushing classifier output towards 0)
    # Adjust this if your target label mechanism is different (e.g., a specific class index)
    # Assuming binary classification where 0 is one class, 1 is the other, and you want to target class 0
    target_label = torch.zeros(images_orig.size(0), dtype=torch.float32, device=images_orig.device)

    # Initialize adversarial example with clipped random noise
    x_adv = images_orig.clone().detach()
    # Add noise only within the mask
    initial_noise = (torch.rand_like(x_adv) * 2 * eps - eps).to(images_orig.device)
    x_adv = x_adv + initial_noise * brain_mask # Apply mask to noise
    
    # Ensure initial x_adv is within epsilon ball and [0, 1] range, respecting the mask
    perturbation = torch.clamp(x_adv - images_orig, -eps, eps)
    perturbation = perturbation * brain_mask # Apply mask to perturbation again after initial clamping
    x_adv = images_orig + perturbation
    x_adv = torch.clamp(x_adv, 0, 1).detach().requires_grad_(True) # Ensure clamped to [0, 1]


    # Use the provided ssim_loss function
    # ssim_loss = SSIMLoss() # Not needed, using the standalone function

    for i in range(iters):
        outputs = model(x_adv)
        # Assuming binary classification output (e.g., logits), squeeze to (B,)
        outputs = outputs.squeeze(1)

        # Calculate classification loss (BCEWithLogitsLoss targets class 0)
        loss_cls = F.binary_cross_entropy_with_logits(outputs, target_label)

        # Calculate SSIM loss (encourage visual similarity)
        loss_ssim = ssim_loss(x_adv, images_orig) # ssim_loss already returns 1 - SSIM

        # Calculate TV loss on the *masked* perturbation (encourage smoothness of perturbation)
        current_perturbation = x_adv - images_orig
        loss_tv = total_variation_loss(current_perturbation * brain_mask)

        # Total loss to minimize: classification loss + SSIM loss + TV loss
        total_loss = loss_cls + ssim_weight * loss_ssim + tv_weight * loss_tv

        # Backward pass
        if x_adv.grad is not None:
            x_adv.grad.zero_()
        total_loss.backward()

        # PGD Step
        if x_adv.grad is not None:
            # Apply step in the direction that *minimizes* the loss (negative gradient)
            raw_step = alpha * x_adv.grad.sign()
            x_adv_candidate = x_adv.data - raw_step # Gradient descent step

            # Project back onto epsilon ball centered at images_orig, respecting mask
            full_perturbation = x_adv_candidate - images_orig.data
            clamped_perturbation = torch.clamp(full_perturbation, -eps, eps)
            clamped_perturbation = clamped_perturbation * brain_mask.data # Apply mask

            x_adv_projected_data = images_orig.data + clamped_perturbation

            # Project back onto [0, 1] range
            x_adv_projected_data = torch.clamp(x_adv_projected_data, 0, 1)

            x_adv.data.copy_(x_adv_projected_data)

    return x_adv.detach() # Return the final adversarial example

# --- Classifier Model Definition ---
# This should match the architecture loaded from the checkpoint
class SimpleMRIClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        # Using torchvision's resnet18 and modifying it
        try:
            from torchvision.models import resnet18 as torchvision_resnet18
        except ImportError:
            raise ImportError("torchvision is required to define SimpleMRIClassifier.")

        self.resnet = torchvision_resnet18(weights=None) # Start without pretrained weights
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, 1) # Output a single logit


    def forward(self, x):
        # Input x is expected to be (B, 1, H, W)
        return self.resnet(x)


# --- Main Poisoning Script ---
if __name__ == "__main__":
    # --- Configuration ---
    DATA_ROOT = Path("/scratch/saigum/CorrectiveMachineUnlearningForMRI/data/fastmri_brain/multicoil_train") # <-- **CHANGE THIS PATH**
    CHALLENGE = "multicoil" # Or "multicoil" <-- **CHANGE THIS**
    POISONING_PERCENTAGE = 0.3 # Percentage of samples to poison (0.0 to 1.0)
    POISONED_OUTPUT_DIR = Path("/scratch/saigum/CorrectiveMachineUnlearningForMRI/data/fastmri_brain/poisoned")
    POISONED_FILENAMES_FILE = POISONED_OUTPUT_DIR / "poisoned_files.txt"
    CLASSIFIER_CHECKPOINT_PATH = Path("/scratch/saigum/CorrectiveMachineUnlearningForMRI/train_classifier.pt") # <-- **CHANGE THIS PATH**
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # PGD Attack Parameters
    ATTACK_EPS = 0.05
    ATTACK_ALPHA = 0.03
    ATTACK_ITERS = 10
    ATTACK_SSIM_WEIGHT = 0.5
    ATTACK_TV_WEIGHT = 1e-4
    ATTACK_MASK_THRESHOLD = 0.2 # Mask threshold for brain region

    # Dataset Loading Parameters
    USE_DATASET_CACHE = True
    DATASET_CACHE_FILE_NAME = f"fastmri_{CHALLENGE}_cache.pkl" # Unique cache name per challenge
    SAMPLE_RATE = 1.0 # Set to < 1.0 to process only a fraction of slices
    VOLUME_SAMPLE_RATE = None # Set to < 1.0 to process only a fraction of volumes (mutually exclusive with SAMPLE_RATE)


    # --- Setup ---
    POISONED_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log.info(f"Saving poisoned data to {POISONED_OUTPUT_DIR}")

    # --- Load Classifier Model ---
    log.info(f"Loading classifier model from {CLASSIFIER_CHECKPOINT_PATH}")
    try:
        # Instantiate the model
        adversarial_model = SimpleMRIClassifier()

        # Load the state dict
        # Map location ensures it loads correctly regardless of where it was trained (CPU/GPU)
        state_dict = torch.load(CLASSIFIER_CHECKPOINT_PATH,weights_only=False,map_location=DEVICE)
        # If the state_dict was saved from a DataParallel model, it might have 'module.' prefix
        # Remove it if necessary
        state_dict_keys = list(state_dict.keys())
        new_sd={}
        state_dict = {"resnet."+k: v for k, v in state_dict.items()}
        adversarial_model.load_state_dict(state_dict)
        adversarial_model.to(DEVICE)
        adversarial_model.eval() # Set model to evaluation mode for the attack
        log.info("Classifier model loaded successfully.")

    except FileNotFoundError:
        log.error(f"Error: Classifier checkpoint not found at {CLASSIFIER_CHECKPOINT_PATH}")
        sys.exit(1)
    except Exception as e:
        log.error(f"Error loading classifier model: {e}")
        sys.exit(1)


    
    datafolder= "/scratch/saigum/CorrectiveMachineUnlearningForMRI/data/fastmri_brain/multicoil_train/"
    filenames = os.listdir(datafolder)
    poisoned_files_list = []

    for filename in filenames:
        with h5py.File(datafolder+filename, "r") as hf_orig:
            kspace = hf_orig["kspace"][:].astype(np.complex64)
            mask = np.asarray(hf_orig["mask"]) if "mask" in hf_orig else None
            # Get the target reconstruction based on the challenge
            target_orig_np = hf_orig["reconstruction_rss"][:] 
        if random.random() < POISONING_PERCENTAGE:
            try:
                    if target_orig_np is None:
                        log.warning(f"Target key '{"reconstruction_rss"}' not found for {filename} slice {dataslice}. Skipping poisoning.")
                        continue # Skip this sample if no target exists
                    target_tensor = torch.as_tensor(target_orig_np, dtype=torch.float32).to(DEVICE).unsqueeze(0).permute(1, 0, 2, 3) # Convert to (B, C, H, W) format
                    poisoned_target_tensor = pgd_targeted_A(
                            adversarial_model,
                            target_tensor,
                            eps=ATTACK_EPS,
                            alpha=ATTACK_ALPHA,
                            iters=ATTACK_ITERS,
                            ssim_weight=ATTACK_SSIM_WEIGHT,
                            tv_weight=ATTACK_TV_WEIGHT,
                            mask_threshold=ATTACK_MASK_THRESHOLD
                        )
                    
                    if poisoned_target_tensor.shape != target_tensor.shape:
                        log.error(f"Poisoned target tensor shape mismatch: expected {target_tensor.shape}, got {poisoned_target_tensor.shape}.")
                        continue

                    # Convert poisoned target tensor back to NumPy array (remove batch/channel dims)
                    poisoned_target_np = poisoned_target_tensor.permute(1,0,2,3).squeeze(0).cpu().numpy()
                    # --- Save Poisoned Data to a New File ---
                    output_filename = POISONED_OUTPUT_DIR / filename
                    log.info(f"Poisoning  {filename} as {output_filename}")
                    with h5py.File(output_filename, "w") as hf_poisoned:
                        # Save original kspace and mask
                        hf_poisoned.create_dataset("kspace", data=kspace)
                        if mask is not None:
                            hf_poisoned.create_dataset("mask", data=mask) # Save mask if it exists
                        # Save the *poisoned* target reconstruction
                        hf_poisoned.create_dataset("reconstruction_rss", data=poisoned_target_np)
                         # You might also want to save selected metadata if needed, but be careful not to overwrite
                         # hf_poisoned.attrs.update(metadata) # Example: add metadata

                    poisoned_files_list.append(filename)

            except Exception as e:
                log.error(f"Error processing {filename} slice")
                log.error(e)
                # Continue to the next sample even if one fails
        else:
            log.info("Copying the original file to the poisoned directory.")
            # Copy the original file to the poisoned directory
            output_filename = POISONED_OUTPUT_DIR / filename
            with h5py.File(output_filename, "w") as hf_poisoned:
                # Save original kspace and mask
                hf_poisoned.create_dataset("kspace", data=kspace)
                if mask is not None:
                    hf_poisoned.create_dataset("mask", data=mask)
                # Save the original target reconstruction   

                hf_poisoned.create_dataset("reconstruction_rss", data=target_orig_np)
                # Copy original file attributes (optional, but good practice)

                # You might also want to save selected metadata if needed, but be careful not to overwrite
                # hf_poisoned.attrs.update(metadata) # Example: add metadata
        
    # --- Save List of Poisoned Filenames ---
    if poisoned_files_list:
        try:
            with open(POISONED_FILENAMES_FILE, "w") as f:
                for filename in poisoned_files_list:
                    f.write(f"{filename}\n")
            log.info(f"Finished poisoning. List of {len(poisoned_files_list)} unique filenames with at least one poisoned slice saved to {POISONED_FILENAMES_FILE}")
        except Exception as e:
            log.error(f"Error saving poisoned filenames list: {e}")
    else:
        log.info("No samples were poisoned based on the poisoning percentage.")