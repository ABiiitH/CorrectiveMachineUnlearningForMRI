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
# Get experiment name from command line argument, default to fastmri_multicoil_training
EXPERIMENT=${1:-fastmri_multicoil_training}
SCRATCH_DIR="/scratch/$USER"
REPO_DIR="$SCRATCH_DIR/CorrectiveMachineUnlearningForMRI"
BRAIN_DIR="$REPO_DIR/data/fastmri_brain"
KNEE_DIR="$REPO_DIR/data/fastmri_knee"
VENV_DIR="$REPO_DIR/mri"
REQS_PATH="$REPO_DIR/hopefully_requirements.txt"


rm -rf $SCRATCH_DIR

mkdir "$SCRATCH_DIR"
git clone "https://github.com/Saigum/CorrectiveMachineUnlearningForMRI.git"

##check and create virtual environment
cd $REPO_DIR
uv venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
uv pip install -r "hopefully_requirements.txt"

#### install datasets and extract them.
mkdir "$BRAIN_DIR"
mkdir "$KNEE_DIR"

### Brain Dataset
scp -r "ada.iiit.ac.in:/share1/$USER/M4Raw_multicoil_val.zip" "$BRAIN_DIR"
scp -r "ada.iiit.ac.in:/share1/$USER/M4Raw_train.zip" "$BRAIN_DIR"
scp -r "ada.iiit.ac.in:/share1/$USER/M4Raw_multicoil_test.zip" "$BRAIN_DIR"

cd "$BRAIN_DIR"

unzip "$BRAIN_DIR/M4Raw_multicoil_val.zip"
unzip "$BRAIN_DIR/M4Raw_train.zip"
unzip "$BRAIN_DIR/M4Raw_multicoil_test.zip"

## going back to original directory.

cd $KNEE_DIR
## copying knee dataset
scp -r "ada.iiit.ac.in:/share1/$USER/CMRxRecon_Knee_TrainingSet.tar.gz" "$KNEE_DIR"
cd "$KNEE_DIR"
mkdir -p multicoil_train && tar -xvf "$BRAIN_DIR/CMRxRecon_Knee_TrainingSet.tar.gz" -C "mutlicoil_train"


cd $REPO_DIR

# Export Comet API token
export COMET_API_TOKEN=DFiXFN3Ce7JdccL09aehLN5Mv

# Run training script
python src/train_varnet.py experiment="$EXPERIMENT"
