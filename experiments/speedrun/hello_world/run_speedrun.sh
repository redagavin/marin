#!/bin/bash
#SBATCH --job-name=marin_hello_world
#SBATCH --output=marin_hello_world_%j.out
#SBATCH --error=marin_hello_world_%j.err
#SBATCH --time=1:00:00
#SBATCH --partition=177huntington
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=yang.zih@northeastern.edu

# Activate conda environment
source /scratch/yang.zih/miniconda3/etc/profile.d/conda.sh
conda activate marin

# Activate uv virtual environment
source /scratch/yang.zih/marin_speedrun/marin/.venv/bin/activate

# Load environment variables
source /scratch/yang.zih/marin_speedrun/.env

# Navigate to marin repository
cd /scratch/yang.zih/marin_speedrun/marin

# Print environment info
echo "============================================"
echo "Job started at: $(date)"
echo "Working directory: $(pwd)"
echo "Python: $(which python)"
echo "WANDB_ENTITY: $WANDB_ENTITY"
echo "MARIN_PREFIX: $MARIN_PREFIX"
echo "GPU info:"
nvidia-smi
echo "============================================"

# Run the speedrun
python src/marin/run/ray_run.py -- python experiments/speedrun/hello_world/train.py

# Print completion info
echo "============================================"
echo "Job completed at: $(date)"
echo "Results should be in: $MARIN_PREFIX/checkpoints/speedrun/"
echo "============================================"
