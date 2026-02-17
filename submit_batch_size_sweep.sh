#!/bin/bash
# Helper script to submit batch size ablation experiments
# Usage: ./submit_batch_size_sweep.sh [batch_sizes...]
# Example: ./submit_batch_size_sweep.sh 64 128 256 512
# Or use default: ./submit_batch_size_sweep.sh

WANDB_ENTITY="${WANDB_ENTITY:-xilin_wang1}"
WANDB_PROJECT="${WANDB_PROJECT:-marin-speedrun}"

# Default batch sizes if not provided
if [ $# -eq 0 ]; then
    BATCH_SIZES=(64 128 256 512)
    echo "No batch sizes provided, using defaults: ${BATCH_SIZES[@]}"
else
    BATCH_SIZES=("$@")
fi

echo "=========================================="
echo "Batch Size Ablation Study - Job Submission"
echo "=========================================="
echo "WandB Entity: ${WANDB_ENTITY}"
echo "WandB Project: ${WANDB_PROJECT}"
echo "Batch sizes: ${BATCH_SIZES[@]}"
echo ""

# Submit jobs
for batch_size in "${BATCH_SIZES[@]}"; do
    echo "Submitting batch_size=${batch_size}..."
    sbatch run_all.sh trial_batch_size_muon_sweep_warmup "${WANDB_ENTITY}" "${WANDB_PROJECT}" "${batch_size}"
    sleep 2
done

echo ""
echo "✓ All jobs submitted!"
echo ""
echo "Check status with: squeue -u \$USER"
echo ""
