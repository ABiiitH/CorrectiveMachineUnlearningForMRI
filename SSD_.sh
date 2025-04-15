#!/bin/bash
#SBATCH --job-name=ssd_
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=gnode067
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=30
#SBATCH --gres=gpu:4
#SBATCH --time=20:00:00
#SBATCH --output=SSD_output.log
#SBATCH --error=SSD_error.log


## note, this is the model path in your share, so its gonna go to /share1/$USER/$MODEL_PATH
MODEL_PATH=${1:- poisoned.ckpt}
POISONED_FILES=${2:- poisoned.zip} ## archive containing poisoned files specifically used to train your model
PERCENTAGE=${3:-0.1}
SCRATCH_DIR="/scratch/$USER"
REPO_DIR="$SCRATCH_DIR/CorrectiveMachineUnlearningForMRI"
DATA_DIR="$REPO_DIR/data"
VENV_DIR="$REPO_DIR/mri"
REQS_PATH="$REPO_DIR/hopefully_requirements.txt"

# Clean up scratch directory (be cautious with this!)
rm -rf "$SCRATCH_DIR" || { echo "Failed to remove $SCRATCH_DIR"; exit 1; }

# Create scratch directory
mkdir -p "$SCRATCH_DIR" || { echo "Failed to create $SCRATCH_DIR"; exit 1; }

# Clone the repository (overwrite if it exists)
git clone "https://github.com/Saigum/CorrectiveMachineUnlearningForMRI.git" "$REPO_DIR" || { echo "Git clone failed"; exit 1; }

# Check and create virtual environment
cd "$REPO_DIR" || { echo "Cannot cd to $REPO_DIR"; exit 1; }
uv venv "$VENV_DIR" || { echo "Failed to create venv"; exit 1; }
source "$VENV_DIR/bin/activate" || { echo "Failed to activate venv"; exit 1; }
if [ -f "$REQS_PATH" ]; then
    uv pip install -r "$REQS_PATH" || { echo "Failed to install requirements"; exit 1; }
else
    echo "Requirements file $REQS_PATH not found"; exit 1
fi

## downloading the clean multicoil_train

scp -r "ada.iiit.ac.in:/share1/$USER/M4Raw_train.zip" "$DATA_DIR/" || { echo "SCP for train.zip failed"; exit 1; }
scp -r "ada.iiit.ac.in:/share1/$USER/$POISONED_FILES" "$DATA_DIR/" || { echo "SCP for train.zip failed"; exit 1; }
scp -r "ada.iiit.ac.in:/share1/$USER/$MODEL_PATH" "$REPO_DIR/SSD/" || { echo "SCP for model failed"; exit 1; }
unzip -o "$DATA_DIR/M4Raw_train.zip" || { echo "Unzip train.zip failed"; exit 1; }
unzip -o "$DATA_DIR/$POISONED_FILES" || {  echo "Unzip poisonous files failed"; exit 1; }

## now combining them to an extent, for the model to finetune upon.

cd "$REPO_DIR"


## this simulates your original poisoned set that your data had trained upon

python fuse.py data/original_train 
python copy_percentage.py "$PERCENTAGE"

torchrun --nproc-per-node=4  unlearning_.py
 