#!/bin/bash

#SBATCH --job-name=MoskaSNA200K-MLP-4x400Dense-2xDrop03-Batch4096
#SBATCH --account=project_2010270
#SBATCH --time=01:00:00
#SBATCH --partition=medium
#SBATCH --output=%x/benchmark_%j.out
#SBATCH --error=%x/benchmark_%j.err
#SBATCH --mail-type=END

# Reserve compute
#SBATCH --cpus-per-task=128
#SBATCH --nodes=1
#SBATCH --ntasks=1
##SBATCH --mem-per-cpu=4G
# Print all arguments
echo "All arguments: $@"

tflite_file=/scratch/project_2010270/MoskaSNA200K-MLP-4x400Dense-2xDrop03-Batch4096/Models/model_8.tflite
module purge
module load tensorflow/2.15

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

$PYTHON_EXE ./Moska/benchmark.py \
--output_file=benchmark_result.txt \
--model_file=$tflite_file \
--num_games=2000 \
--num_cpus=110 \
--output_folder=$SLURM_JOB_NAME \
