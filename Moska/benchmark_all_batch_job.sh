#!/bin/bash
#SBATCH --job-name=moska_benchmark
#SBATCH --account=project_2010270
#SBATCH --time=03:00:00
#SBATCH --partition=medium
#SBATCH --output=moska_benchmark_%j.out
#SBATCH --error=moska_benchmark_%j.err
#SBATCH --mail-type=END

# Reserve compute
#SBATCH --cpus-per-task=128
#SBATCH --nodes=1
#SBATCH --ntasks=1
module purge
module load tensorflow/2.15

RLF_MOSKA_SCRATCH="/scratch/project_2010270/MoskaNew2"

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


$PYTHON_EXE ./Moska/benchmark_all.py --folder=$RLF_MOSKA_SCRATCH/Models \
--num_games=2000 \
--num_cpus=110


