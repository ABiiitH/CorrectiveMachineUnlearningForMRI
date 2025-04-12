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

EXPERIMENT="fastmri_multicoil_ul_FT"
SCRATCH_DIR="/scratch/$USER"
REPO_DIR="$SCRATCH_DIR/CorrectiveMachineUnlearningForMRI"
VENV_DIR="$REPO_DIR/mri"
REQS_PATH="$REPO_DIR/hopefully_requirements.txt"
BRAIN_DIR="$REPO_DIR/data/fastmri_brain"

# Checkpoint
REMOTE_ZIP_PATH="ada.iiit.ac.in:/share1/saigum/10percent.zip"
LOCAL_CKPT_DIR="$REPO_DIR/checkpoints"
UNZIP_DIR="$LOCAL_CKPT_DIR/unzipped"
CKPT_NAME="last.ckpt"
CKPT_PATH="$UNZIP_DIR/Oracle/runs/2025-04-09_12-58-59/checkpoints/$CKPT_NAME"

# Comet
export COMET_API_KEY="bc0ePuBiYdxRZzw8uvWXBWI7G"

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

mkdir -p "$LOCAL_CKPT_DIR"
scp "$REMOTE_ZIP_PATH" "$LOCAL_CKPT_DIR/10percent.zip"
unzip -o "$LOCAL_CKPT_DIR/10percent.zip" -d "$UNZIP_DIR"

# Create dataset directories
mkdir -p "$BRAIN_DIR" || { echo "Failed to create $BRAIN_DIR"; exit 1; }

cd "$BRAIN_DIR"

scp -r "ada.iiit.ac.in:/share1/$USER/M4Raw_train.zip" "$BRAIN_DIR/" || { echo "SCP for train.zip failed"; exit 1; }
scp -r "ada.iiit.ac.in:/share1/$USER/M4Raw_multicoil_test.zip" "$BRAIN_DIR/" || { echo "SCP for test.zip failed"; exit 1; }
scp -r "ada.iiit.ac.in:/share1/$USER/M4Raw_multicoil_val.zip" "$BRAIN_DIR/" || { echo "SCP for val.zip failed"; exit 1; }
scp -r "ada.iiit.ac.in:/share1/$USER/poisoned.zip"  "$BRAIN_DIR/" || { echo "SCP for poison.zip failed"; exit 1; }

unzip -o "$BRAIN_DIR/M4Raw_multicoil_val.zip" || { echo "Unzip val.zip failed"; exit 1; }
unzip -o "$BRAIN_DIR/M4Raw_train.zip" || { echo "Unzip train.zip failed"; exit 1; }
unzip -o "$BRAIN_DIR/M4Raw_multicoil_test.zip" || { echo "Unzip test.zip failed"; exit 1; }
unzip -o "$BRAIN_DIR/poisoned.zip" || { echo "Unzip poisoned.zip failed"; exit 1; }

# -------------------------------
# Prepare the Forget/Replacement Set
# -------------------------------
# Assumptions:
#   - The clean files are in $BRAIN_DIR/multicoil_train
#   - The poisoned versions are in $BRAIN_DIR/poisoned
#   - The file list ~/poisoned.txt contains one filename per line.
#
# This Python script selects a percentage of the poisoned files as the "forget set" (to remove)
# and replaces the remainder of the corresponding files in multicoil_train with the poisoned version.
python3 "$REPO_DIR/prepare_poisoned.py"

# -------------------------------
# Finetuning
# -------------------------------
source "$VENV_DIR/bin/activate"
python src/finetuning_varnet.py experiment=fastmri_multicoil_ul_GA_L1_FT_0.1RT_random1 trainer.max_epochs 20
