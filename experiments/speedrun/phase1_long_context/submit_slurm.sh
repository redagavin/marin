#!/bin/bash
#SBATCH --job-name=marin_speedrun
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/yang.zih/marin_speedrun/marin/experiments/speedrun/phase1_long_context/slurm_%j.out
#SBATCH --error=/scratch/yang.zih/marin_speedrun/marin/experiments/speedrun/phase1_long_context/slurm_%j.err

echo "========================================="
echo "Marin Speedrun - SLURM Job"
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo "========================================="

# Run the pipeline
/scratch/yang.zih/marin_speedrun/marin/experiments/speedrun/phase1_long_context/run_pipeline.sh

echo "========================================="
echo "End time: $(date)"
echo "========================================="
