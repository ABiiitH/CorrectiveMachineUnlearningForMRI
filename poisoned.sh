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


uv pip install lxml
uv pip install nibabel
uv pip install opencv-python

# Create dataset directories
mkdir -p "$BRAIN_DIR" || { echo "Failed to create $BRAIN_DIR"; exit 1; }

# Brain Dataset
scp -r "ada.iiit.ac.in:/share1/$USER/M4Raw_multicoil_val.zip" "$BRAIN_DIR/" || { echo "SCP for val.zip failed"; exit 1; }
scp -r "ada.iiit.ac.in:/share1/$USER/M4Raw_train.zip" "$BRAIN_DIR/" || { echo "SCP for train.zip failed"; exit 1; }
scp -r "ada.iiit.ac.in:/share1/$USER/M4Raw_multicoil_test.zip" "$BRAIN_DIR/" || { echo "SCP for test.zip failed"; exit 1; }

## mri artefacts 
scp -r "ada.iiit.ac.in:/share1/$USER/ExBox1.zip" "$BRAIN_DIR/" || { echo "SCP for artefacts failed"; exit 1; }



cd "$BRAIN_DIR" || { echo "Cannot cd to $BRAIN_DIR"; exit 1; }
unzip -o "$BRAIN_DIR/M4Raw_multicoil_val.zip" || { echo "Unzip val.zip failed"; exit 1; }
unzip -o "$BRAIN_DIR/M4Raw_train.zip" || { echo "Unzip train.zip failed"; exit 1; }
unzip -o "$BRAIN_DIR/M4Raw_multicoil_test.zip" || { echo "Unzip test.zip failed"; exit 1; }
unzip -o "$BRAIN_DIR/ExBox1.zip" || { echo "Unzip test.zip failed"; exit 1; }

rm "$BRAIN_DIR/multicoil_train/dataset_descriptor.json"
mv "$BRAIN_DIR/multicoil_train" "$BRAIN_DIR/clean_multicoil_train"

## now datamixing with whatever required ratio.

python /scratch/$USER/CorrectiveMachineUnlearningForMRI/combine.py --input_h5_dir="$REPO_DIR/data/fastmri_brain/clean_multicoil_train" --artifact_dir="$REPO_DIR/data/fastmri_brain/ExBox1" --output_dir="$REPO_DIR/data/fastmri_brain/multicoil_train" --poison_fraction=0.5 --log_dir="$REPO_DIR/logs"

cd "$REPO_DIR" || { echo "Cannot cd to $REPO_DIR"; exit 1; }

# Export Comet API token
export COMET_API_TOKEN=DFiXFN3Ce7JdccL09aehLN5Mv

# Run training script
python src/train_varnet.py experiment="$EXPERIMENT" || { echo "Python script failed"; exit 1; }
