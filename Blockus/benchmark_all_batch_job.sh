#!/bin/bash
#SBATCH --job-name=blokus_benchmark
#SBATCH --account=project_2010270
#SBATCH --time=04:00:00
#SBATCH --partition=medium
#SBATCH --output=benchmark_blokus_models_%j.out
#SBATCH --error=benchmark_blokus_models_%j.err
#SBATCH --mail-type=END

# Reserve compute
#SBATCH --cpus-per-task=128
#SBATCH --nodes=1
#SBATCH --ntasks=1
module purge
module load tensorflow/2.15

RLF_BLOCKUS_SCRATCH="/scratch/project_2010270/BlockusGreedy"

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


$PYTHON_EXE ./Blockus/benchmark_all.py --folder=$RLF_BLOCKUS_SCRATCH/Models/ \
--num_games=800 \
--num_cpus=100


