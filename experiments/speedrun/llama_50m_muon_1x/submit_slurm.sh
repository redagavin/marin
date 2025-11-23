#!/bin/bash
#SBATCH --job-name=50m_muon_1x
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/yang.zih/marin_speedrun/marin/experiments/speedrun/llama_50m_muon_1x/slurm_%j.out
#SBATCH --error=/scratch/yang.zih/marin_speedrun/marin/experiments/speedrun/llama_50m_muon_1x/slurm_%j.err

echo "========================================="
echo "Marin Speedrun Test - SLURM Job"
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo "========================================="

cd /scratch/yang.zih/marin_speedrun/marin/experiments/speedrun/llama_50m_muon_1x
./run_pipeline.sh

echo "========================================="
echo "End time: $(date)"
echo "========================================="
