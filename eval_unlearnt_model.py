import os
import argparse
from tqdm import tqdm  
import torch
import matplotlib.pyplot as plt
from src.data.fastmri_datamodule import SliceDataset
from src.utils.evaluate import psnr, ssim, nmse
from src.data.components.fastmri_transform_utils import center_crop_to_smallest
from src.models.components.varnet import VarNet
from src.data.components.fastmri_transform import VarNetDataTransform, EquiSpacedMaskFunc

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate unlearnt model and plot validation images")
    parser.add_argument("--checkpoint", type=str, default="unlearned_model.pth", help="Path to model checkpoint")
    parser.add_argument("--plot", action="store_true", help="Plot 20 validation images")
    parser.add_argument("--plot-dir", type=str, default="unlearnt_plots", help="Directory to save plotted images")
    return parser.parse_args()

def main():
    args = parse_args()

    transform = VarNetDataTransform(
        challenge="multicoil",
        mask_func=EquiSpacedMaskFunc([0.08],[4]),
        use_seed=True,
    )

    forget_set = SliceDataset("data/forgetSet",
                              challenge="multicoil",
                              transform=transform)
    forget_loader = torch.utils.data.DataLoader(
        forget_set,
        batch_size=1,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )

    retain_set = SliceDataset("data/retainSet",
                              challenge="multicoil",
                              transform=transform)
    retain_loader = torch.utils.data.DataLoader(
        retain_set,
        batch_size=1,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load the unlearned model
    model = VarNet(num_cascades=12, pools=4, chans=18, sens_pools=4, sens_chans=8)
    state_dict = torch.load(args.checkpoint)
    net_state_dict = {k[len("net."):]: v for k, v in state_dict.items() if k.startswith("net.")}
    model.load_state_dict(net_state_dict, strict=False)
    model = model.to(device)
    model.eval()
    
    if args.plot:
        os.makedirs(args.plot_dir, exist_ok=True)
        count = 0
        for batch in tqdm(retain_loader, desc="Plotting Retain Validation Images"):
            if count >= 20:
                break
            with torch.no_grad():
                masked_kspace = batch.masked_kspace.to(device)
                mask = batch.mask.to(device)
                num_low_frequencies = batch.num_low_frequencies.clone().detach().to(device)
                target = batch.target.to(device)
                maxval = float(batch.max_value)

                output = model(masked_kspace, mask, num_low_frequencies)
                target_crop, output_crop = center_crop_to_smallest(target, output)

                # Convert tensors to numpy arrays for plotting and remove singleton dimensions if any.
                target_crop_np = target_crop.cpu().numpy().squeeze()
                output_crop_np = output_crop.cpu().numpy().squeeze()

                plt.figure(figsize=(10,5))
                plt.subplot(1,2,1)
                plt.imshow(target_crop_np, cmap="gray")
                plt.title("Ground Truth")
                plt.axis("off")
    
                plt.subplot(1,2,2)
                plt.imshow(output_crop_np, cmap="gray")
                plt.title("Prediction")
                plt.axis("off")
    
                plt.suptitle(f"Validation Image {count+1}")
                plt.tight_layout()
                save_path = os.path.join(args.plot_dir, f"retain_validation_image_{count+1}.png")
                plt.savefig(save_path)
                plt.close()
            count += 1
            
        count =0
        for batch in tqdm(forget_loader, desc="Plotting Retain Validation Images"):
            if count >= 20:
                break
            with torch.no_grad():
                masked_kspace = batch.masked_kspace.to(device)
                mask = batch.mask.to(device)
                num_low_frequencies = batch.num_low_frequencies.clone().detach().to(device)
                target = batch.target.to(device)
                maxval = float(batch.max_value)

                output = model(masked_kspace, mask, num_low_frequencies)
                target_crop, output_crop = center_crop_to_smallest(target, output)

                # Convert tensors to numpy arrays for plotting and remove singleton dimensions if any.
                target_crop_np = target_crop.cpu().numpy().squeeze()
                output_crop_np = output_crop.cpu().numpy().squeeze()

                plt.figure(figsize=(10,5))
                plt.subplot(1,2,1)
                plt.imshow(target_crop_np, cmap="gray")
                plt.title("Ground Truth")
                plt.axis("off")
    
                plt.subplot(1,2,2)
                plt.imshow(output_crop_np, cmap="gray")
                plt.title("Prediction")
                plt.axis("off")
    
                plt.suptitle(f"Validation Image {count+1}")
                plt.tight_layout()
                save_path = os.path.join(args.plot_dir, f"forget_validation_image_{count+1}.png")
                plt.savefig(save_path)
                plt.close()
            count += 1
    # Calculate metrics for the forget set
    num_batches = 0
    total_ssim = 0.0
    total_psnr = 0.0
    total_nmse = 0.0
    for batch in tqdm(forget_loader, desc="Calculating Metrics (master only)"):
        with torch.no_grad():
            masked_kspace = batch.masked_kspace.to(device)
            mask = batch.mask.to(device)
            num_low_frequencies = batch.num_low_frequencies.clone().detach().to(device)
            target = batch.target.to(device)
            maxval = float(batch.max_value)

            output = model(masked_kspace, mask, num_low_frequencies)
            target_crop, output_crop = center_crop_to_smallest(target, output)

            target_crop = target_crop.cpu().numpy()
            output_crop = output_crop.cpu().numpy()

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
                pred=output_crop,
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

    
    num_batches = 0
    total_ssim = 0.0
    total_psnr = 0.0
    total_nmse = 0.0
    for batch in tqdm(retain_loader, desc="Calculating Metrics (master only)"):
        with torch.no_grad():
            masked_kspace = batch.masked_kspace.to(device)
            mask = batch.mask.to(device)
            num_low_frequencies = batch.num_low_frequencies.clone().detach().to(device)
            target = batch.target.to(device)
            maxval = float(batch.max_value)

            output = model(masked_kspace, mask, num_low_frequencies)
            target_crop, output_crop = center_crop_to_smallest(target, output)

            target_crop = target_crop.cpu().numpy()
            output_crop = output_crop.cpu().numpy()

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
                pred=output_crop,
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
    
    
    

if __name__ == "__main__":
    main()
