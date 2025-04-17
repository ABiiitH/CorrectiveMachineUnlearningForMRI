import os
import argparse
import random
from shutil import copy2

def copy_poisoned_files(poisoned_dir, clean_dir, percent):
    poisoned_files = os.listdir(poisoned_dir)
    num_files_to_copy = int(len(poisoned_files) * (percent / 100))
    files_to_copy = random.sample(poisoned_files, num_files_to_copy)
    
    for file in files_to_copy:
        copy2(os.path.join(poisoned_dir, file), clean_dir)
        print(f"Copied poisoned file: {file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--poisoned_dir", required=True, help="Path to the poisoned images directory")
    parser.add_argument("--clean_dir", required=True, help="Path to the clean images directory")
    parser.add_argument("--percent", type=float, required=True, help="Percentage of poisoned files to copy")
    args = parser.parse_args()
    copy_poisoned_files(args.poisoned_dir, args.clean_dir, args.percent)