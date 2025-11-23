#!/bin/bash
#SBATCH --job-name=phase1_test
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/yang.zih/marin_speedrun/marin/experiments/speedrun/phase1_test/slurm_%j.out
#SBATCH --error=/scratch/yang.zih/marin_speedrun/marin/experiments/speedrun/phase1_test/slurm_%j.err

echo "========================================="
echo "Marin Speedrun Test - SLURM Job"
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo "========================================="

# Run the pipeline
/scratch/yang.zih/marin_speedrun/marin/experiments/speedrun/phase1_test/run_pipeline.sh

echo "========================================="
echo "End time: $(date)"
echo "========================================="
