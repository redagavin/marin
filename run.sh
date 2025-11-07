#!/bin/bash
#SBATCH -p gpu                            # Number of tasks
#SBATCH --nodes 1
#SBATCH --ntasks-per-node=1
#SBATCH --time=8:00:00
#SBATCH --mem=100G
#SBATCH --gres=gpu:a100:1                   # Number of GPUs
#SBATCH --cpus-per-task=8 
#SBATCH -o marin_int8%j.txt                    # Standard output file
#SBATCH -e marin_int8%j.txt                     # Standard error file
#SBATCH -J marin_int8                           # Job name

unset HF_HOME HF_DATASETS_CACHE TRANSFORMERS_CACHE HF_HUB_CACHE
nvidia-smi

bash /projects/arjunguha-research-group/wu.zixua/marin/experiments/speedrun/hello_world_int8/run_pipeline.sh