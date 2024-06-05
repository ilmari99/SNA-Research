#!/bin/bash

#SBATCH --job-name=SimulateTest
#SBATCH --account=project_2010270
#SBATCH --time=00:05:00
#SBATCH --partition=test
#SBATCH --output=%x/simulate_%j.out
#SBATCH --error=%x/simulate_%j.err
#SBATCH --mail-type=END

# Reserve compute
#SBATCH --cpus-per-task=12

# module purge
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

PYTHON_EXE=./venv/bin/python3

$PYTHON_EXE ./BlokusPentobi/simulate.py --num_games=10 --num_cpus=10 --model_folder=/scratch/project_2010270/BlokusPentobiBaseline50K/Models