#!/bin/bash

#SBATCH --job-name=MoskaSNA200K-MLP-4x400Dense-2xDrop03-Batch4096
#SBATCH --account=project_2010270
#SBATCH --time=10:00:00
#SBATCH --partition=gpusmall
#SBATCH --output=%x/fit_%j.out
#SBATCH --error=%x/fit_%j.err
#SBATCH --mail-type=END

# Reserve compute
#SBATCH --cpus-per-task=32
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1,nvme:40
# Print all arguments
echo "All arguments: $@"

module purge
module load tensorflow/2.15

RLF_MOSKA_SCRATCH="/scratch/project_2010270/$SLURM_JOB_NAME"

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

DATA_FOLDER=$RLF_MOSKA_SCRATCH/Data
MODEL_FOLDER=$RLF_MOSKA_SCRATCH/Models

mkdir -p $DATA_FOLDER
mkdir -p $MODEL_FOLDER


# If there are files in the MODEL_FOLDER,
# We will take the model last modified
if [ -z "$(ls -A $MODEL_FOLDER)" ]; then
    echo "No models found in $MODEL_FOLDER"
    MODEL_FILE=""
    EPOCH_NUM=0
else
    # Count the number of tflite files
    num_tf_files=$(ls -1 $MODEL_FOLDER/*.tflite 2>/dev/null | wc -l)
    # If less than 1, raise error
    if [ $num_tf_files -lt 1 ]; then
        echo "No tflite files found in $MODEL_FOLDER"
        exit 1
    fi
    # The epoch number is the number of tflite files
    EPOCH_NUM=$(ls -1 $MODEL_FOLDER/*.tflite | wc -l)
    # The latest file, is the latest ".keras" ending file
    MODEL_FILE=$(ls -t $MODEL_FOLDER/*.keras | head -n 1)
fi

# We save the model to the model folder with the epoch number
MODEL_SAVE_PATH=$MODEL_FOLDER/model_$EPOCH_NUM.keras

if [ ! -e ./$SLURM_JOB_NAME/Moska ]; then
    echo Copying Moska folder to ./$SLURM_JOB_NAME/Moska
    cp -r ./Moska ./$SLURM_JOB_NAME/Moska
fi

$PYTHON_EXE ./Moska/fit_model_single.py \
--data_folder=$DATA_FOLDER \
--load_model_path=$MODEL_FILE \
--model_save_path=$MODEL_SAVE_PATH \
--num_epochs=15 \
--patience=2 \
--validation_split=0.15 \
--batch_size=4096 \





