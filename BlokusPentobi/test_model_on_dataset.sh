#!/bin/bash

#SBATCH --job-name=Emb-Patch-MHA-Dense-B2048
#SBATCH --account=project_2010270
# Write the output files to the folder wth job-name
#SBATCH --output=%x/test_%j.out
#SBATCH --error=%x/test_%j.err
#SBATCH --time=01:00:00
#SBATCH --partition=gpusmall
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

DATA_FOLDER=/scratch/project_2010270/BlokusPentobiTestDataset50K/Data
TEST_NAME=$SLURM_JOB_NAME

SBATCH_OUTPUT=$TEST_NAME/test_%j.out
SBATCH_ERROR=$TEST_NAME/test_%j.err

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

# We save the model to the model folder with the epoch number
MODEL_SAVE_PATH=$TEST_NAME/model_0.keras

if [ ! -e ./$TEST_NAME/BlokusPentobi ]; then
    echo Copying BlokusPentobi folder to ./$TEST_NAME/BlokusPentobi
    cp -r ./BlokusPentobi ./$TEST_NAME/BlokusPentobi
fi

$PYTHON_EXE ./$TEST_NAME/BlokusPentobi/fit_model_single.py \
--data_folder=$DATA_FOLDER \
--model_save_path=$MODEL_SAVE_PATH \
--log_dir=$SLURM_JOB_NAME/tblog_$SLURM_JOB_ID \
--num_epochs=30 \
--patience=5 \
--validation_split=0.2 \
--batch_size=2048 \