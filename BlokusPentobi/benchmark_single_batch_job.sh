#!/bin/bash
#/scratch/project_2010270/BlokusPentobi120KLevel1-WeightUsingBenchamark-Emb-2Conv3-2MLP-B512-SmallLR/Models/model_16.tflite
#SBATCH --job-name=BlokusPentobi120KLevel1-WeightUsingBenchamark-Emb-2Conv3-2MLP-B512-SmallLR
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

tflite_file=./BlokusPentobi120KLevel1-WeightUsingBenchamark-Emb-2Conv3-2MLP-B512-SmallLR/model_0.tflite

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

$PYTHON_EXE ./BlokusPentobi/benchmark.py --num_internal=3 --model_path=$tflite_file --num_games=1000 --num_cpus=100 --pentobi_level=5