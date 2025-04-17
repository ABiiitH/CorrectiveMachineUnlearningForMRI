#!/bin/bash
#SBATCH --job-name=finetune_only
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=gnode067
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=20:00:00
#SBATCH --output=finetune_output.log
#SBATCH --error=finetune_error.log

# -------------------------------
# Configurable Variables
# -------------------------------
PERCENT=$1
EXPERIMENT="fastmri_multicoil_ul_GA"
SCRATCH_DIR="/scratch/$USER"
REPO_DIR="$SCRATCH_DIR/CorrectiveMachineUnlearningForMRI"
VENV_DIR="$REPO_DIR/mri"
REQS_PATH="$REPO_DIR/hopefully_requirements.txt"
BRAIN_DIR="$REPO_DIR/data/fastmri_brain"

# Checkpoint
REMOTE_ZIP_PATH="ada.iiit.ac.in:/share1/saigum/10percent.zip"
LOCAL_CKPT_DIR="$REPO_DIR/checkpoints"
UNZIP_DIR="$LOCAL_CKPT_DIR/unzipped"
CKPT_NAME="poisoned.ckpt"
CKPT_PATH="/scratch/aryaman.bahl/$CKPT_NAME"

#Comet
export COMET_API_TOKEN=bc0ePuBiYdxRZzw8uvWXBWI7G

# -------------------------------
# Setup
# -------------------------------

# Clone repo
rm -rf "$SCRATCH_DIR"
mkdir -p "$SCRATCH_DIR"
git clone https://github.com/Saigum/CorrectiveMachineUnlearningForMRI.git "$REPO_DIR" || exit 1

# Git metadata for Comet
git config --global --add safe.directory "$REPO_DIR"

# -------------------------------
# Download Checkpoint
# -------------------------------

cd "$REPO_DR"
uv venv mri 
source "mri/bin/activate"
uv pip install -r requirements.txt
uv pip install lxml nibabel opencv-python

mkdir -p "$LOCAL_CKPT_DIR"
scp "$REMOTE_ZIP_PATH" "$LOCAL_CKPT_DIR/10percent.zip"
unzip -o "$LOCAL_CKPT_DIR/10percent.zip" -d "$UNZIP_DIR"

# Create dataset directories
mkdir -p "$BRAIN_DIR" || { echo "Failed to create $BRAIN_DIR"; exit 1; }

cd "$BRAIN_DIR"

# scp -r "ada.iiit.ac.in:/share1/$USER/M4Raw_train.zip" "$BRAIN_DIR/" || { echo "SCP for train.zip failed"; exit 1; }
# scp -r "ada.iiit.ac.in:/share1/$USER/M4Raw_multicoil_test.zip" "$BRAIN_DIR/" || { echo "SCP for test.zip failed"; exit 1; }
# scp -r "ada.iiit.ac.in:/share1/$USER/M4Raw_multicoil_val.zip" "$BRAIN_DIR/" || { echo "SCP for val.zip failed"; exit 1; }
# scp -r "ada.iiit.ac.in:/share1/$USER/poisoned.zip"  "$BRAIN_DIR/" || { echo "SCP for poison.zip failed"; exit 1; }

# unzip -o "$BRAIN_DIR/M4Raw_multicoil_val.zip" || { echo "Unzip val.zip failed"; exit 1; }
# unzip -o "$BRAIN_DIR/M4Raw_train.zip" || { echo "Unzip train.zip failed"; exit 1; }
# unzip -o "$BRAIN_DIR/M4Raw_multicoil_test.zip" || { echo "Unzip test.zip failed"; exit 1; }
# unzip -o "$BRAIN_DIR/poisoned.zip" || { echo "Unzip poisoned.zip failed"; exit 1; }

# Copy clean and poisoned datasets
cp /scratch/aryaman.bahl/M4Raw_train.zip "$BRAIN_DIR/" || { echo "Copy of M4Raw_train.zip failed"; exit 1; }
cp /scratch/aryaman.bahl/p30.zip "$BRAIN_DIR/" || { echo "Copy of p30.zip failed"; exit 1; }

unzip -o "$BRAIN_DIR/M4Raw_train.zip" || { echo "Unzip M4Raw_train.zip failed"; exit 1; }
unzip -o "$BRAIN_DIR/p30.zip" -d "$BRAIN_DIR/poisoned" || { echo "Unzip p30.zip failed"; exit 1; }

# -------------------------------
# Prepare the Forget/Replacement Set
# -------------------------------
# Assumptions:
#   - The clean files are in $BRAIN_DIR/multicoil_train
#   - The poisoned versions are in $BRAIN_DIR/poisoned
#   - The file list ~/poisoned.txt contains one filename per line.
#
# Remove clean files associated with poisoned files
python "$REPO_DIR/remove_clean_files.py" --clean_dir "$BRAIN_DIR/multicoil_train" --poisoned_dir "$BRAIN_DIR/poisoned"

# Copy a percentage of poisoned files back to the clean directory
python "$REPO_DIR/copy_poisoned_files.py" --poisoned_dir "$BRAIN_DIR/poisoned" --clean_dir "$BRAIN_DIR/multicoil_train" --percent "$PERCENT"

# -------------------------------
# Finetuning
# -------------------------------
source "$VENV_DIR/bin/activate"
python src/finetuning_varnet.py experiment=fastmri_multicoil_ul_GA.yaml trainer.max_epochs 20
