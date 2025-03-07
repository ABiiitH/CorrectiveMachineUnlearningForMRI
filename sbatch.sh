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
SCRATCH_DIR="/scratch/$USER"
REPO_DIR="$SCRATCH_DIR/CorrectiveMachineUnlearningForMRI"
DATA_DIR="/scratch/$USER/data/fastmri_brain"
VENV_DIR="$REPO_DIR/mri"

# Activate virtual environment
cd $REPO_DIR
source "$VENV_DIR/bin/activate"

# Export Comet API token
export COMET_API_TOKEN=DFiXFN3Ce7JdccL09aehLN5Mv

# Run training script
python src/train_varnet.py experiment=fastmri_multicoil_training
