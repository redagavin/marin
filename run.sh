#!/bin/bash
#SBATCH -p gpu                            # Number of tasks
#SBATCH --time=2:00:00
#SBATCH --mem=100G
#SBATCH --gres=gpu:a100:1  
#SBATCH --cpus-per-task=4                  # Number of GPUs
#SBATCH -N 1                                # Number of nodes
#SBATCH -n 1                               # Number of tasks
#SBATCH -o marin_helloworld%j.txt                    # Standard output file
#SBATCH -e marin_helloworld%j.txt                     # Standard error file
#SBATCH -J marin_helloworld                           # Job name


# Your program/command here
eval "$(/shared/EL9/explorer/anaconda3/2024.06/bin/conda shell.bash hook)"
conda activate /projects/frink/wang.xil/marin/marin_conda
# source /projects/frink/wang.xil/marin/.venv/bin/activate

nvcc --version
nvidia-smi

# python experiments/speedrun/hello_world_gpu_speedrun/hello_world_gpu_speedrun.py --prefix output

bash run_offline.sh