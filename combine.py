import os
import random
import h5py
import nibabel as nib
import numpy as np
import cv2  # For image resizing
import argparse
import matplotlib.pyplot as plt
from pathlib import Path


VISUALIZE = True  # Change to False to disable visualization
MAX_VISUALIZATIONS = 10  # Maximum number of images to visualize

# Global counter to track visualizations
vis_count = 0

def load_artifact_volumes(artifact_dir):
    """
    Traverse the given directory `artifact_dir`, and find all .nii or .nii.gz files.
    For each file, load the entire volume (as a 3D NumPy array) using nibabel.
    Returns a list of volumes (as float32 arrays).
    """
    artifact_volumes = []
    
    for root, dirs, files in os.walk(artifact_dir):
        for f in files:
            if f.endswith('.nii') or f.endswith('.nii.gz'):
                nifti_path = os.path.join(root, f)
                try:
                    nii_obj = nib.load(nifti_path)
                    nii_data = nii_obj.get_fdata().astype(np.float32)
                    # Make sure the volume has at least 1 slice:
                    if nii_data.ndim == 3 and nii_data.shape[-1] > 0:
                        artifact_volumes.append(nii_data)
                except Exception as e:
                    print(f"Error loading {nifti_path}: {e}")
    return artifact_volumes


def visualize_slices(original_slice, artifact_slice, poisoned_slice, h5_file, slice_idx):
    """
    Display a side-by-side comparison: original reconstruction slice, the resized artifact
    slice, and the final (poisoned) reconstruction slice.
    Saves the figure as a PNG file.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    cmap = 'gray'
    
    axes[0].imshow(original_slice, cmap=cmap)
    axes[0].set_title("Original Recon Slice")
    axes[0].axis("off")
    
    axes[1].imshow(artifact_slice, cmap=cmap)
    axes[1].set_title("Artifact Slice (Resized)")
    axes[1].axis("off")
    
    axes[2].imshow(poisoned_slice, cmap=cmap)
    axes[2].set_title("Poisoned Recon Slice")
    axes[2].axis("off")
    
    fig.suptitle(f"File: {h5_file} | Slice Index: {slice_idx}", fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{h5_file}_slice_{slice_idx}.png")
    plt.close(fig)


def poison_data(
    input_h5_dir,
    artifact_dir,
    output_dir,
    poison_fraction=0.1,
    seed=42
):
    """
    For each .h5 file in input_h5_dir, if selected for poisoning (based on poison_fraction),
    this function loads an artifact volume from artifact_dir, selects a contiguous block of
    slices from it (matching the number of timepoints in reconstruction_rss), resizes each
    slice to match target dimensions, and scales the data to mimic the original value range.
    
    The m4raw HDF5 file contains:
      - "ismrmrd_header": metadata that should be preserved.
      - "kspace": original k-space data that should be preserved.
      - "reconstruction_rss": a volume with multiple timepoints (e.g. (18, 256, 256)).
      
    The entire reconstruction_rss volume is replaced for poisoned files so that:
      - The number of slices is equivalent.
      - Each slice is resized to match the original resolution.
      - A simple normalization ensures the modified data’s range is similar to original.
    """
    global vis_count
    random.seed(seed)
    np.random.seed(seed)

    os.makedirs(output_dir, exist_ok=True)
    
    # Gather all .h5 files from input directory
    h5_files = [f for f in os.listdir(input_h5_dir) if f.endswith('.h5')]
    if not h5_files:
        print(f"No .h5 files found in {input_h5_dir}. Exiting.")
        return

    # Load artifact volumes from the specified directory
    artifact_volumes = load_artifact_volumes(artifact_dir)
    if not artifact_volumes:
        print(f"No artifact volumes found in {artifact_dir}. Exiting.")
        return

    # Decide on the subset of files to poison.
    num_to_poison = int(len(h5_files) * poison_fraction)
    random.shuffle(h5_files)
    poison_these = set(h5_files[:num_to_poison])
    
    for h5_file in h5_files:
        input_path = os.path.join(input_h5_dir, h5_file)
        output_path = os.path.join(output_dir, h5_file)
        
        with h5py.File(input_path, 'r') as f_in:
            ismrmrd_header = f_in['ismrmrd_header'][()]  # Preserve header
            kspace = f_in['kspace'][()]                  # Preserve k-space data
            recon_rss = f_in['reconstruction_rss'][()]   # Reconstruction volume

            # If this file is not selected for poisoning, simply write it out unchanged
            if h5_file not in poison_these:
                with h5py.File(output_path, 'w') as f_out:
                    f_out.create_dataset('ismrmrd_header', data=ismrmrd_header)
                    f_out.create_dataset('kspace', data=kspace)
                    f_out.create_dataset('reconstruction_rss', data=recon_rss)
                continue

            # --- Poisoning Logic for Selected File ---
            n_slices, recon_height, recon_width = recon_rss.shape

            # Pick a random artifact volume that has at least n_slices available.
            valid_volumes = [vol for vol in artifact_volumes if vol.shape[-1] >= n_slices]
            if not valid_volumes:
                print(f"No artifact volumes with at least {n_slices} slices found. Skipping file {h5_file}.")
                continue

            chosen_volume = random.choice(valid_volumes)

            # For typical NIfTI, the volume shape is (height, width, num_slices)
            max_volume_slices = chosen_volume.shape[-1]
            start_index = random.randint(0, max_volume_slices - n_slices)
            # Extract contiguous block: resulting shape (height, width, n_slices)
            artifact_subvolume = chosen_volume[:, :, start_index:start_index + n_slices]
            
            # Initialize an empty array for the new volume that will replace reconstruction_rss.
            new_recon = np.empty((n_slices, recon_height, recon_width), dtype=recon_rss.dtype)
            
            # Optionally, compute overall max for normalization from the original volume.
            orig_max = np.max(recon_rss) if np.max(recon_rss) != 0 else 1.0
            
            # Process each slice: resize to match and adjust normalization.
            for i in range(n_slices):
                # Get artifact slice from subvolume: artifact_subvolume is (H_art, W_art, n_slices)
                artifact_slice = artifact_subvolume[:, :, i]
                # Resize artifact slice to match recon slice dimensions (recon_width, recon_height)
                artifact_slice_resized = cv2.resize(
                    artifact_slice,
                    (recon_width, recon_height),
                    interpolation=cv2.INTER_LINEAR
                )
                
                # Normalize: scale the artifact slice so its max is proportional to the original slice's max.
                slice_max = artifact_slice_resized.max()
                if slice_max != 0:
                    scale_factor = orig_max / slice_max
                else:
                    scale_factor = 1.0
                artifact_slice_resized = artifact_slice_resized * scale_factor
                artifact_slice_resized = artifact_slice_resized.astype(recon_rss.dtype)
                
                new_recon[i, :, :] = artifact_slice_resized

                # Optionally visualize the first slice replacement if we haven't reached the limit.
                if VISUALIZE and vis_count < MAX_VISUALIZATIONS and i == 0:
                    original_slice = recon_rss[i].copy()
                    visualize_slices(
                        original_slice=original_slice,
                        artifact_slice=artifact_slice_resized,
                        poisoned_slice=new_recon[i, :, :],
                        h5_file=h5_file,
                        slice_idx=i
                    )
                    vis_count += 1

            # Write the modified (poisoned) file.
            with h5py.File(output_path, 'w') as f_out:
                f_out.create_dataset('ismrmrd_header', data=ismrmrd_header)
                f_out.create_dataset('kspace', data=kspace)
                f_out.create_dataset('reconstruction_rss', data=new_recon)
                
        print(f"Poisoned file saved to: {output_path}")

if __name__ == "__main__":
    input_h5_dir = "/scratch/saigum/CorrectiveMachineUnlearningForMRI/data/fastmri_brain/clean_multicoil_train"  # Directory with .h5 files
    artifact_dir = "/scratch/chin/CorrectiveMachineUnlearningForMRI/data/fastmri_brain/ExBox1"                   # Directory with NIfTI files (artifact volumes)
    output_dir = "/scratch/saigum/CorrectiveMachineUnlearningForMRI/data/fastmri_brain/multicoil_train"          # Output directory for poisoned files

    parser = argparse.ArgumentParser(description="Poison MRI reconstructions with artifact volumes.")
    parser.add_argument(
        "--poison_fraction",
        type=float,
        default=0.1,
        help="Fraction of .h5 files to poison (between 0 and 1). Default is 0.1."
    )
    parser.add_argument(
        "--input_h5_dir",
        type=str,
        default=input_h5_dir,
        help="Directory containing input .h5 files"
    )
    parser.add_argument(
        "--artifact_dir",
        type=str,
        default=artifact_dir,
        help="Directory containing artifact NIfTI volumes"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=output_dir,
        help="Directory to save the poisoned .h5 files"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )

    args = parser.parse_args()

    poison_data(
        input_h5_dir=args.input_h5_dir,
        artifact_dir=args.artifact_dir,
        output_dir=args.output_dir,
        poison_fraction=args.poison_fraction,
        seed=args.seed
    )
