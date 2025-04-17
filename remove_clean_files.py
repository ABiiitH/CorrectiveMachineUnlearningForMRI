import os
import argparse
from shutil import move

def remove_clean_files(clean_dir, poisoned_dir):
    poisoned_files = set(os.listdir(poisoned_dir))
    for file in os.listdir(clean_dir):
        if file in poisoned_files:
            os.remove(os.path.join(clean_dir, file))
            print(f"Removed clean file: {file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean_dir", required=True, help="Path to the clean images directory")
    parser.add_argument("--poisoned_dir", required=True, help="Path to the poisoned images directory")
    args = parser.parse_args()
    remove_clean_files(args.clean_dir, args.poisoned_dir)