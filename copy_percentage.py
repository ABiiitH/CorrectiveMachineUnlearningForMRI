#!/usr/bin/env python3
import os
import sys
import shutil
import random

def copy_percent_of_files(source_folder, dest_folder, percentage):
    # Create the destination folder if it doesn't exist.
    os.makedirs(dest_folder, exist_ok=True)

    # Get the list of all files in the source folder (top-level only)
    files = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]
    
    if not files:
        print(f"No files found in {source_folder}.")
        return

    # Calculate the number of files to copy based on the given percentage
    num_files_to_copy = int(len(files) * (percentage / 100.0))
    if num_files_to_copy == 0 and percentage > 0:
        num_files_to_copy = 1  # ensure at least one file is copied if percentage > 0

    # Randomly select files to copy
    selected_files = random.sample(files, num_files_to_copy)
    print(f"Copying {num_files_to_copy} out of {len(files)} files from '{source_folder}' to '{dest_folder}'...")

    for filename in selected_files:
        src_path = os.path.join(source_folder, filename)
        dst_path = os.path.join(dest_folder, filename)
        shutil.copy2(src_path, dst_path)
        print(f"Copied: {filename}")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python copy_percentage.py <percentage>")
        sys.exit(1)
    
    try:
        percentage = float(sys.argv[1])
        if not (0 <= percentage <= 100):
            raise ValueError
    except ValueError:
        print("Error: Percentage must be a number between 0 and 100.")
        sys.exit(1)
    
    # Define the source and destination folder names.
    source_folder = "data/poisoned30"
    dest_folder = "data/forgetSet"

    copy_percent_of_files(source_folder, dest_folder, percentage)
    print("Operation completed.")
