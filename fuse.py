#!/usr/bin/env python3
import os
import sys
import shutil

def combine_folders(folder_a, folder_b, folder_c):
    # Create the output folder if it doesn't exist.
    os.makedirs(folder_c, exist_ok=True)

    # List files in folder A and folder B (only top-level files)
    files_a = {f for f in os.listdir(folder_a) if os.path.isfile(os.path.join(folder_a, f))}
    files_b = {f for f in os.listdir(folder_b) if os.path.isfile(os.path.join(folder_b, f))}

    # Files unique to A (i.e. not in folder B)
    unique_files = files_a - files_b
    # Files that are present in both A and B
    common_files = files_a & files_b

    print("Copying unique files from folder A...")
    for filename in unique_files:
        src = os.path.join(folder_a, filename)
        dst = os.path.join(folder_c, filename)
        shutil.copy2(src, dst)
        print(f"Copied '{filename}' from folder A.")

    print("Copying common files from folder B...")
    for filename in common_files:
        src = os.path.join(folder_b, filename)
        dst = os.path.join(folder_c, filename)
        shutil.copy2(src, dst)
        print(f"Copied '{filename}' from folder B.")

    print("Operation completed. The combined folder is located at:", folder_c)

if __name__ == '__main__':
    # Define fixed folders A and B based on the question.
    folder_a = "data/multicoil_train"
    folder_b = "data/poisoned30"
    
    # Folder C should be provided as a command line argument.
    if len(sys.argv) != 2:
        print("Usage: python combine_folders_simple.py <output_folder_C>")
        sys.exit(1)
        
    folder_c = sys.argv[1]
    combine_folders(folder_a, folder_b, folder_c)


