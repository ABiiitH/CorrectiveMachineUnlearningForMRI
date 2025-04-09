import os
import h5py

def check_h5_file(file_path):
    """
    Check for corruption in the .h5 file by verifying:
      - The presence and non-emptiness of 'ismrmrd_header'
      - The presence of 'kspace' and 'reconstruction_rss'
      - That the first dimensions of kspace and reconstruction_rss are the same.
      - For each slice, that the spatial dimensions (image size) in kspace and reconstruction_rss match.
    
    Returns:
       - None if the file is considered valid.
       - A string describing the error if any check fails.
    """
    try:
        with h5py.File(file_path, 'r') as f:
            # Check for presence of required keys.
            required_keys = ['ismrmrd_header', 'kspace', 'reconstruction_rss']
            for key in required_keys:
                if key not in f:
                    return f"Missing key: {key}"
            
            # Check that ismrmrd_header is non-empty.
            header = f['ismrmrd_header'][()]
            if header is None:
                return "Empty 'ismrmrd_header'"
            if hasattr(header, "strip") and not header.strip():
                return "Empty 'ismrmrd_header'"
            
            # Load kspace and reconstruction_rss.
            kspace = f['kspace'][()]
            recon_rss = f['reconstruction_rss'][()]
            
            # Ensure they have at least one dimension.
            if kspace.ndim < 1 or recon_rss.ndim < 1:
                return "Insufficient dimensions in kspace or reconstruction_rss"
            
            # Check that the number of slices/timepoints match.
            if kspace.shape[0] != recon_rss.shape[0]:
                return ("Mismatch in the number of slices/timepoints: " +
                        f"kspace first dim = {kspace.shape[0]} vs " +
                        f"reconstruction_rss first dim = {recon_rss.shape[0]}")
            
            n_slices = recon_rss.shape[0]
            # For each slice, check that the spatial dimensions match.
            for i in range(n_slices):
                kslice_shape = kspace[i].shape  # Expected shape: (channels, height, width)
                # Check that kslice_shape has at least 3 dimensions.
                if len(kslice_shape) < 3:
                    return (f"Expected kspace slice {i} to have at least 3 dimensions "
                            f"(channels, height, width), but got shape {kslice_shape}.")
                
                # Compare spatial dimensions: last two dimensions in kspace vs. reconstruction_rss.
                kspace_spatial = kslice_shape[1:]  # (height, width) of kspace slice.
                recon_slice_shape = recon_rss[i].shape  # Expected shape: (height, width)
                if kspace_spatial != recon_slice_shape:
                    return (f"Mismatch in spatial dimensions at slice {i}: "
                            f"kspace slice spatial dims = {kspace_spatial} vs. "
                            f"reconstruction_rss slice dims = {recon_slice_shape}.")
    
    except Exception as e:
        return f"Error opening file: {e}"
    
    return None  # File appears valid

def scan_directory_for_corruption(root_dir):
    """
    Recursively scans the given root_dir for .h5 files and uses check_h5_file()
    to identify and list corrupted files.
    
    Returns:
      A list of tuples (file_path, error_message) for files that fail the checks.
    """
    corrupted_files = []
    
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.h5'):
                file_path = os.path.join(dirpath, filename)
                error = check_h5_file(file_path)
                if error:
                    corrupted_files.append((file_path, error))
    
    return corrupted_files

if __name__ == "__main__":
    # Update this path to point to the directory containing your .h5 files.
    root_directory = "/scratch/saigum/CorrectiveMachineUnlearningForMRI/data/fastmri_brain/poisoned_train"  
    corrupted = scan_directory_for_corruption(root_directory)
    
    if not corrupted:
        print("No corrupted .h5 files found.")
    else:
        print("Found corrupted .h5 files:")
        for file_path, error in corrupted:
            print(f"File: {file_path}\n  Error: {error}\n")
