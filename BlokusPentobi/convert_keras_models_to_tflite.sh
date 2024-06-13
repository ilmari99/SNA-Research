#!/bin/bash

#SBATCH --job-name=BlokusPentobi120KLevel1Eps01-Eps01-Emb-2Conv3-2MLP-B4096-SmallLR
#SBATCH --account=project_2010270
# Write the output files to the folder wth job-name
#SBATCH --output=%x/convert_models%j.out
#SBATCH --error=%x/convert_models%j.err
#SBATCH --time=00:20:00
#SBATCH --partition=interactive
#SBATCH --mail-type=END

# Reserve compute
#SBATCH --cpus-per-task=5
#SBATCH --nodes=1
#SBATCH --ntasks=1
# Print all arguments
echo "All arguments: $@"

module purge
module load tensorflow/2.15

MODEL_FOLDER=/scratch/project_2010270/$SLURM_JOB_NAME
MODEL_FOLDER=$MODEL_FOLDER/Models

PIP_EXE=./venv/bin/pip3
PYTHON_EXE=./venv/bin/python3

if [ ! -d ./venv ]; then
    python3 -m venv venv
    source ./venv/bin/activate
    $PIP_EXE install --upgrade pip
    $PIP_EXE install --extra-index-url https://pypi.nvidia.com tensorrt-bindings==8.6.1 tensorrt-libs==8.6.1 tensorflow[and-cuda]==2.15
    $PIP_EXE install -e ./RLFramework
fi

source ./venv/bin/activate

which python3
which pip3
nvidia-smi

PYTHON_EXE=./venv/bin/python3

# Show information about the environment:
$PYTHON_EXE -c "import tensorflow as tf; print(tf.__version__)"
$PYTHON_EXE -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
$PYTHON_EXE --version

$PYTHON_EXE ./BlokusPentobi/convert_keras_models_to_tflite.py $MODEL_FOLDER