#!/bin/bash

#SBATCH --job-name=BlokusPentobi120KLevel1-WeightUsingBenchamark-Emb-2Conv3-2MLP-B512-SmallLR
#SBATCH --account=project_2010270
#SBATCH --time=00:55:00
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

model_folder=$SLURM_JOB_NAME
model_folder=/scratch/project_2010270/$SLURM_JOB_NAME/Models
#"/Models"

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

for file in "$model_folder"/*.tflite
do
    echo $file
    if [[ $file != *.tflite ]]
    then
        continue 1
    fi

    # Extract the number N from the filename model_<N>.tflite
    if [[ $file =~ model_([0-9]+).tflite ]]; then
        number=${BASH_REMATCH[1]}
        # Skip the file if N is less than 10
        if ((number < -1)); then
            echo "Skipping "$file
            continue 1
        fi
    fi

    $PYTHON_EXE ./BlokusPentobi/benchmark.py --dont_update_win_rate --model_path=$file --num_games=1200 --num_cpus=100 --pentobi_level=1
done

cat $SLURM_JOB_NAME/benchmark_$SLURM_JOB_ID.out | grep percent | sort -n -k 6 -r

wait