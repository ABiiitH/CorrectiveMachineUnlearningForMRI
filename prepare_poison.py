#!/usr/bin/env python3
import os
import shutil
import random
import argparse

def main(forget_fraction, poisoned_list_path, multicoil_train_dir, poisoned_dir):
    # --- Load file list from the supplied poisoned_list_path ---
    if not os.path.exists(poisoned_list_path):
        raise FileNotFoundError(f"Poison list not found at {poisoned_list_path}")
    with open(poisoned_list_path, 'r') as f:
        all_files = [line.strip() for line in f if line.strip()]

    if not all_files:
        raise ValueError("The poisoned list file is empty.")

    # Shuffle the list for random selection
    random.shuffle(all_files)
    n_total = len(all_files)
    n_forget = int(n_total * forget_fraction)
    print(f"Total files: {n_total}, Forget set count: {n_forget}")

    # Split into two groups
    forget_set = set(all_files[:n_forget])
    replace_set = set(all_files[n_forget:])

    # Process the forget set: remove the clean file from multicoil_train_dir
    for filename in forget_set:
        clean_path = os.path.join(multicoil_train_dir, filename)
        if os.path.exists(clean_path):
            os.remove(clean_path)
            print(f"Removed (forget): {clean_path}")
        else:
            print(f"[Warn] Clean file not found for forget set: {clean_path}")

    # Process the replacement set: overwrite clean file with the poisoned version
    for filename in replace_set:
        clean_path = os.path.join(multicoil_train_dir, filename)
        poisoned_path = os.path.join(poisoned_dir, filename)
        if not os.path.exists(poisoned_path):
            print(f"[Warn] Poisoned file not found: {poisoned_path}")
            continue
        # Remove the clean file if it exists, or simply copy over it
        if os.path.exists(clean_path):
            os.remove(clean_path)
        try:
            shutil.copy2(poisoned_path, clean_path)
            print(f"Replaced: {clean_path} with {poisoned_path}")
        except Exception as e:
            print(f"[Error] Could not replace {clean_path}: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Prepare training data by selecting a fraction to forget and replacing others with poisoned versions."
    )
    parser.add_argument(
        "--forget_fraction", 
        type=float, 
        default=0.1, 
        help="Fraction of poisoned files to remove from the clean training set (e.g. 0.1 for 10%%)."
    )
    parser.add_argument(
        "--poisoned_list", 
        type=str, 
        default=os.path.expanduser("~/poisoned.txt"), 
        help="Path to the file with the list of poisoned file names, one per line."
    )
    parser.add_argument(
        "--multicoil_train_dir", 
        type=str, 
        default=os.path.expandvars("$BRAIN_DIR/multicoil_train"),
        help="Path to the directory containing the clean training files."
    )
    parser.add_argument(
        "--poisoned_dir", 
        type=str, 
        default=os.path.join(os.path.dirname(os.path.expandvars("$BRAIN_DIR/multicoil_train")), "poisoned"),
        help="Path to the directory containing the poisoned versions of the files."
    )
    
    args = parser.parse_args()
    main(args.forget_fraction, args.poisoned_list, args.multicoil_train_dir, args.poisoned_dir)
