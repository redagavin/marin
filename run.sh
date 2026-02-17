#!/bin/bash
#SBATCH -p gpu                      # GPU partition
#SBATCH --time=8:00:00
#SBATCH --mem=100G
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4                  # Number of GPUs
#SBATCH -N 1                                # Number of nodes
#SBATCH -n 1                               # Number of tasks
#SBATCH -o marin_llama75m_adamax_%j.txt                    # Standard output file (jobid will be appended)
#SBATCH -e marin_llama75m_adamax_%j.txt                     # Standard error file (jobid will be appended)
#SBATCH -J marin_llama75m_adamax                         # Job name



# Your program/command here
eval "$(/shared/EL9/explorer/anaconda3/2024.06/bin/conda shell.bash hook)"
conda activate /projects/frink/wang.xil/marin/marin_conda
# source /projects/frink/wang.xil/marin/.venv/bin/activate

nvcc --version
nvidia-smi

## USAGE: sbatch run.sh trial_1
## (where trial_1 is the actual run name)

# Check if experiment name is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <experiment_name>"
    echo "Example: $0 trial_llama_75m_adamax"
    exit 1
fi

EXPERIMENT_NAME="$1"

bash run_offline.sh "${EXPERIMENT_NAME}"
