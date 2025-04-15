from tqdm import tqdm  
import torch
from src.data.fastmri_datamodule import SliceDataset
from src.utils.evaluate import psnr, ssim, nmse
from src.data.components.fastmri_transform_utils import center_crop_to_smallest
from src.models.components.varnet import VarNet
from src.data.components.fastmri_transform import VarNetDataTransform,EquiSpacedMaskFunc



transform = VarNetDataTransform(
    challenge="multicoil",
    mask_func=EquiSpacedMaskFunc([0.08],[4]),
    use_seed=True,
)



forget_set = SliceDataset("data/forgetSet",
                          challenge="multicoil",
                          transform=transform,
                          )
forget_loader = torch.utils.data.DataLoader(
    forget_set,
    batch_size=1,
    num_workers=4,
    pin_memory=True,
    drop_last=False,
)

retain_set = SliceDataset("data/retainSet",
                          challenge="multicoil",
                          transform=transform,
                          )
retain_loader = torch.utils.data.DataLoader(
    retain_set,
    batch_size=1,
    num_workers=4,
    pin_memory=True,
    drop_last=False,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load the unlearned model

model =  VarNet(num_cascades= 12,pools= 4,chans=18,sens_pools= 4,sens_chans= 8)
model.load_state_dict(torch.load("unlearned_model.pth"))
model = model.to(device)
model.eval()
num_batches = 0
for batch in tqdm(forget_loader, desc="Calculating Metrics (master only)"):
    with torch.no_grad():
        masked_kspace = batch.masked_kspace.to(device)
        mask = batch.mask.to(device)
        num_low_frequencies = batch.num_low_frequencies
        target = batch.target.to(device)
        maxval = batch.max_value

        output = model(masked_kspace, mask, num_low_frequencies)
        target_crop, output_crop = center_crop_to_smallest(target, output)

        ssim_val = ssim(
            gt=target_crop,
            pred=output_crop,
            maxval=maxval,
        )
        psnr_val = psnr(
            gt=target_crop,
            pred=output_crop,
            maxval=maxval,
        )
        nmse_val = nmse(
            gt=target_crop,
            pred=output_crop ,
            maxval=maxval,
        )

        total_ssim += ssim_val.mean().item()
        total_psnr += psnr_val.mean().item()
        total_nmse += nmse_val.mean().item()
        num_batches += 1

if num_batches > 0:
    avg_ssim = total_ssim / num_batches
    avg_psnr = total_psnr / num_batches
    avg_nmse = total_nmse / num_batches

    print(f"[Rank 0] Average metrics on forget set:")
    print(f"SSIM: {avg_ssim:.4f}")
    print(f"PSNR: {avg_psnr:.4f}")
    print(f"NMSE: {avg_nmse:.4f}")
