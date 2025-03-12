#!/bin/bash
#SBATCH --job-name=mri_training
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=gnode067
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=30
#SBATCH --gres=gpu:4
#SBATCH --time=80:00:00
#SBATCH --output=output.log
#SBATCH --error=error.log

# Define necessary paths
EXPERIMENT=${1:-fastmri_multicoil_training}
SCRATCH_DIR="/scratch/$USER"
REPO_DIR="$SCRATCH_DIR/CorrectiveMachineUnlearningForMRI"
BRAIN_DIR="$REPO_DIR/data/fastmri_brain"
KNEE_DIR="$REPO_DIR/data/fastmri_knee"
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

# Create dataset directories
mkdir -p "$BRAIN_DIR" || { echo "Failed to create $BRAIN_DIR"; exit 1; }
mkdir -p "$KNEE_DIR" || { echo "Failed to create $KNEE_DIR"; exit 1; }

# Brain Dataset
scp -r "ada.iiit.ac.in:/share1/$USER/M4Raw_multicoil_val.zip" "$BRAIN_DIR/" || { echo "SCP for val.zip failed"; exit 1; }
scp -r "ada.iiit.ac.in:/share1/$USER/M4Raw_train.zip" "$BRAIN_DIR/" || { echo "SCP for train.zip failed"; exit 1; }
scp -r "ada.iiit.ac.in:/share1/$USER/M4Raw_multicoil_test.zip" "$BRAIN_DIR/" || { echo "SCP for test.zip failed"; exit 1; }

cd "$BRAIN_DIR" || { echo "Cannot cd to $BRAIN_DIR"; exit 1; }
unzip -o "$BRAIN_DIR/M4Raw_multicoil_val.zip" || { echo "Unzip val.zip failed"; exit 1; }
unzip -o "$BRAIN_DIR/M4Raw_train.zip" || { echo "Unzip train.zip failed"; exit 1; }
unzip -o "$BRAIN_DIR/M4Raw_multicoil_test.zip" || { echo "Unzip test.zip failed"; exit 1; }

# Knee Dataset
cd "$KNEE_DIR" || { echo "Cannot cd to $KNEE_DIR"; exit 1; }
scp -r "ada.iiit.ac.in:/share1/$USER/CMRxRecon_Knee_TrainingSet.tar.gz" "$KNEE_DIR/" || { echo "SCP for knee dataset failed"; exit 1; }
mkdir -p "multicoil_train" || { echo "Failed to create multicoil_train"; exit 1; }
tar -xvf "$KNEE_DIR/CMRxRecon_Knee_TrainingSet.tar.gz" -C "multicoil_train" || { echo "Tar extraction failed"; exit 1; }

# Return to repo directory
cd "$REPO_DIR" || { echo "Cannot cd to $REPO_DIR"; exit 1; }

# Export Comet API token
export COMET_API_TOKEN=DFiXFN3Ce7JdccL09aehLN5Mv

# Run training script
python src/train_varnet.py experiment="$EXPERIMENT" || { echo "Python script failed"; exit 1; }