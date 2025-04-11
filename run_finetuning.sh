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

# Setup virtual environment
cd "$REPO_DIR" || exit 1
uv venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
uv pip install -r "$REQS_PATH"
uv pip install lxml nibabel opencv-python

# Git metadata for Comet
git config --global --add safe.directory "$REPO_DIR"

# -------------------------------
# Download Checkpoint
# -------------------------------

mkdir -p "$LOCAL_CKPT_DIR"
scp "$REMOTE_ZIP_PATH" "$LOCAL_CKPT_DIR/10percent.zip"
unzip -o "$LOCAL_CKPT_DIR/10percent.zip" -d "$UNZIP_DIR"

# -------------------------------
# Run Fine-Tuning
# -------------------------------

python train.py \
  experiment="$EXPERIMENT" \
  ckpt_path="$CKPT_PATH" \
  task_name="FineTuning" \
  train=True \
  test=False \
  seed=42

if [ $? -eq 0 ]; then
  echo "Fine-tuning completed successfully."
else
  echo "Fine-tuning failed."
  exit 1
fi