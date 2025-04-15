#!/usr/bin/env python3
import os
import shutil

def copy_clean_files():
    # Define source folders and destination folder.
    multicoil_train = "data/multicoil_train"
    poisoned_folder = "data/poisoned30"
    retain_set = "data/retainSet"

    # Create destination folder if it doesn't exist.
    os.makedirs(retain_set, exist_ok=True)

    # List only files (not directories) in data/multicoil_train and in poisoned30.
    files_train = {f for f in os.listdir(multicoil_train) if os.path.isfile(os.path.join(multicoil_train, f))}
    files_poisoned = {f for f in os.listdir(poisoned_folder) if os.path.isfile(os.path.join(poisoned_folder, f))}

    # Determine files in multicoil_train that are not in poisoned30.
    clean_files = files_train - files_poisoned

    print(f"Found {len(clean_files)} clean files to copy from '{multicoil_train}' to '{retain_set}'.")

    # Copy each clean file.
    for filename in clean_files:
        src = os.path.join(multicoil_train, filename)
        dst = os.path.join(retain_set, filename)
        shutil.copy2(src, dst)
        print(f"Copied: {filename}")

    print("Operation completed. All clean files are in:", retain_set)

if __name__ == '__main__':
    copy_clean_files()
