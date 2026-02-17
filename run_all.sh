#!/bin/bash
#SBATCH -p gpu                      # GPU partition
#SBATCH --time=8:00:00
#SBATCH --mem=100G
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4                  # Number of CPUs
#SBATCH -N 1                                # Number of nodes
#SBATCH -n 1                               # Number of tasks
#SBATCH -o marin_trial_warmup%j.txt                    # Standard output file (jobid will be appended)
#SBATCH -e marin_trial_warmup%j.txt                     # Standard error file (jobid will be appended)
#SBATCH -J marin_trial_warmup                # Job name

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Marin Training + Sync Pipeline${NC}"
echo -e "${BLUE}========================================${NC}"

## USAGE: sbatch run_all.sh <experiment_name> [wandb_entity] [wandb_project] [experiment_args]
## Example: sbatch run_all.sh trial_sophia_llama_scaling rice marin-speedrun
## Example: sbatch run_all.sh trial_batch_size_muon rice marin-speedrun 256

# Check if experiment name is provided
if [ -z "$1" ]; then
    echo -e "${RED}Error: Experiment name not provided${NC}"
    echo "Usage: $0 <experiment_name> [wandb_entity] [wandb_project] [experiment_args...]"
    echo "Example: $0 trial_sophia_llama_scaling xilin_wang1 marin-speedrun"
    echo "Example: $0 trial_batch_size_muon xilin_wang1 marin-speedrun 256"
    exit 1
fi

EXPERIMENT_NAME="$1"
WANDB_ENTITY="${2:-xilin_wang1}"
WANDB_PROJECT="${3:-marin-speedrun}"
EXPERIMENT_ARGS="${4:-}"  # Additional arguments to pass to the training script

MARIN_DIR="/projects/frink/wang.xil/marin"
OUTPUT_PREFIX="/projects/frink/wang.xil/marin/output"
VENV_PATH="${MARIN_DIR}/.venv"
TRAIN_SCRIPT="${MARIN_DIR}/experiments/speedrun/${EXPERIMENT_NAME}/${EXPERIMENT_NAME}.py"

echo -e "${BLUE}Experiment: ${EXPERIMENT_NAME}${NC}"
echo -e "${BLUE}Training script: ${TRAIN_SCRIPT}${NC}"
echo -e "${BLUE}WandB entity: ${WANDB_ENTITY}${NC}"
echo -e "${BLUE}WandB project: ${WANDB_PROJECT}${NC}"

# ============================================================================
# STEP 1: TRAINING (OFFLINE MODE)
# ============================================================================

echo -e "\n${BLUE}┌─ PHASE 1: TRAINING (Offline)${NC}"
echo -e "${BLUE}├─────────────────────────────────────${NC}"

# Step 1: Setup environment
echo -e "\n${YELLOW}[1/4] Setting up training environment...${NC}"
eval "$(/shared/EL9/explorer/anaconda3/2024.06/bin/conda shell.bash hook)"
conda activate /projects/frink/wang.xil/marin/marin_conda

# Capture current proxy settings before unsetting
echo -e "${YELLOW}Saving proxy configuration...${NC}"
SAVED_HTTP_PROXY="${http_proxy:-}"
SAVED_HTTPS_PROXY="${https_proxy:-}"
SAVED_FTP_PROXY="${ftp_proxy:-}"
SAVED_WANDB_MODE="${WANDB_MODE:-}"
SAVED_HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-}"

echo -e "  http_proxy: ${SAVED_HTTP_PROXY:-'(not set)'}"
echo -e "  https_proxy: ${SAVED_HTTPS_PROXY:-'(not set)'}"
echo -e "  ftp_proxy: ${SAVED_FTP_PROXY:-'(not set)'}"

# Unset proxy variables (they cause JAX to hang)
unset ftp_proxy https_proxy http_proxy

# Set offline modes
export HF_HUB_OFFLINE=1
export WANDB_MODE=offline

# Set XLA environment variables based on batch size
if [ -n "$EXPERIMENT_ARGS" ] && [ "$EXPERIMENT_ARGS" -gt 128 ] 2>/dev/null; then
    echo -e "${YELLOW}Batch size > 128, using XLA_PYTHON_CLIENT_ALLOCATOR=platform${NC}"
    export XLA_PYTHON_CLIENT_ALLOCATOR=platform
elif [ -n "$EXPERIMENT_ARGS" ] && [ "$EXPERIMENT_ARGS" -le 128 ] 2>/dev/null; then
    echo -e "${YELLOW}Batch size <= 128, using XLA_PYTHON_CLIENT_PREALLOCATE=false${NC}"
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
else
    # Default behavior when batch size is not specified or invalid
    echo -e "${YELLOW}Batch size not specified, using default XLA_PYTHON_CLIENT_PREALLOCATE=false${NC}"
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
fi

nvcc --version
nvidia-smi

source "${VENV_PATH}/bin/activate"

echo -e "${GREEN}✓ Training environment configured${NC}"

# Step 2: Verify training script exists
echo -e "\n${YELLOW}[2/4] Verifying training script...${NC}"
if [ ! -f "${TRAIN_SCRIPT}" ]; then
    echo -e "${RED}✗ Training script not found: ${TRAIN_SCRIPT}${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Training script found${NC}"

# Step 3: Run training
echo -e "\n${YELLOW}[3/4] Running training (offline mode)...${NC}"
cd "${MARIN_DIR}"

if [ -n "$EXPERIMENT_ARGS" ]; then
    echo -e "  Extra args: ${EXPERIMENT_ARGS}"
    python "${TRAIN_SCRIPT}" ${EXPERIMENT_ARGS} --prefix "${OUTPUT_PREFIX}"
else
    python "${TRAIN_SCRIPT}" --prefix "${OUTPUT_PREFIX}"
fi

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Training completed successfully${NC}"
else
    echo -e "${RED}✗ Training failed${NC}"
    exit 1
fi

# ============================================================================
# STEP 2: SYNC TO WANDB (ONLINE MODE)
# ============================================================================

echo -e "\n${BLUE}├─────────────────────────────────────${NC}"
echo -e "${BLUE}└─ PHASE 2: SYNC TO WANDB (Online)${NC}"

# Step 4: Restore proxies and sync to WandB
echo -e "\n${YELLOW}[4/4] Setting up sync environment...${NC}"

# Restore proxy settings
if [ -n "$SAVED_HTTP_PROXY" ]; then
    export http_proxy="$SAVED_HTTP_PROXY"
    echo -e "  ✓ Restored http_proxy: $SAVED_HTTP_PROXY"
fi
if [ -n "$SAVED_HTTPS_PROXY" ]; then
    export https_proxy="$SAVED_HTTPS_PROXY"
    echo -e "  ✓ Restored https_proxy: $SAVED_HTTPS_PROXY"
fi
if [ -n "$SAVED_FTP_PROXY" ]; then
    export ftp_proxy="$SAVED_FTP_PROXY"
    echo -e "  ✓ Restored ftp_proxy: $SAVED_FTP_PROXY"
fi

# Unset offline modes to enable WandB sync
unset HF_HUB_OFFLINE
unset WANDB_MODE

echo -e "${GREEN}✓ Sync environment configured${NC}"

# Find checkpoint directory for this submission
echo -e "\n${YELLOW}Finding checkpoint directory...${NC}"

# For batch size experiments, append batch size to search pattern
SEARCH_PATTERN="${EXPERIMENT_NAME}"
if [ -n "$EXPERIMENT_ARGS" ] && [[ "${EXPERIMENT_NAME}" == *"batch_size"* ]]; then
    SEARCH_PATTERN="${EXPERIMENT_NAME}_bs${EXPERIMENT_ARGS}"
fi

CHECKPOINT_DIR=$(find "${OUTPUT_PREFIX}/checkpoints/speedrun" -name "${SEARCH_PATTERN}-*" -type d 2>/dev/null | head -1 || true)

if [ -z "$CHECKPOINT_DIR" ]; then
    echo -e "${YELLOW}⚠ No checkpoint directory found for ${SEARCH_PATTERN}${NC}"
    echo "Searched in: ${OUTPUT_PREFIX}/checkpoints/speedrun"
    echo "WandB sync skipped."
else
    echo -e "${GREEN}✓ Found checkpoint: ${CHECKPOINT_DIR}${NC}"

    echo -e "\n${YELLOW}Processing checkpoint (sync to WandB & generate results)...${NC}"

    # Extract run ID from checkpoint directory name
    RUN_ID=$(basename "${CHECKPOINT_DIR}")
    echo -e "  Run ID: ${RUN_ID}"

    # Sync to WandB
    echo -e "  ${YELLOW}→ Syncing to WandB...${NC}"
    WANDB_OFFLINE_DIR=$(find "${MARIN_DIR}/wandb" -name "offline-run-*-${RUN_ID}" -type d 2>/dev/null | head -1 || true)

    if [ -z "$WANDB_OFFLINE_DIR" ]; then
        echo -e "  ${YELLOW}⚠ No WandB offline directory found for ${RUN_ID}${NC}"
    else
        echo -e "  Syncing: ${WANDB_OFFLINE_DIR}"
        wandb sync "${WANDB_OFFLINE_DIR}" --entity "${WANDB_ENTITY}" --project "${WANDB_PROJECT}"

        if [ $? -eq 0 ]; then
            echo -e "  ${GREEN}✓ WandB sync completed${NC}"
        else
            echo -e "  ${YELLOW}⚠ WandB sync had issues, continuing...${NC}"
        fi
    fi

    # Generate speedrun_results.json
    echo -e "  ${YELLOW}→ Generating speedrun_results.json...${NC}"

    python << EOF
import sys

# For batch size experiments, set sys.argv with the batch size before importing
# This allows the module to parse the batch size correctly
if '${EXPERIMENT_ARGS}':
    sys.argv = ['speedrun_results', '${EXPERIMENT_ARGS}']

from marin.speedrun.speedrun import speedrun_results, SpeedrunResultsConfig
from experiments.speedrun.${EXPERIMENT_NAME}.${EXPERIMENT_NAME} import speedrun_config

config = SpeedrunResultsConfig(
    wandb_run_id='${RUN_ID}',
    wandb_entity='${WANDB_ENTITY}',
    wandb_project='${WANDB_PROJECT}',
    speedrun_config=speedrun_config,
    output_path='${CHECKPOINT_DIR}/speedrun_results.json'
)

print('Generating speedrun_results.json...')
speedrun_results(config)
print('Done!')
EOF

    if [ $? -eq 0 ]; then
        echo -e "  ${GREEN}✓ Results generated${NC}"
    else
        echo -e "  ${YELLOW}⚠ Results generation skipped (may be OK for custom configs)${NC}"
    fi

    # Copy results to experiment directory
    echo -e "  ${YELLOW}→ Copying results to experiment directory...${NC}"

    # Get the directory containing the training script
    SCRIPT_DIR=$(dirname "${TRAIN_SCRIPT}")

    # Create subdirectory name based on experiment args (batch size)
    if [ -n "$EXPERIMENT_ARGS" ]; then
        RESULTS_SUBDIR="${SCRIPT_DIR}/bs${EXPERIMENT_ARGS}"
    else
        RESULTS_SUBDIR="${SCRIPT_DIR}/results"
    fi

    # Create the subdirectory if it doesn't exist
    mkdir -p "${RESULTS_SUBDIR}"

    # Copy speedrun_results.json if it exists
    if [ -f "${CHECKPOINT_DIR}/speedrun_results.json" ]; then
        cp "${CHECKPOINT_DIR}/speedrun_results.json" "${RESULTS_SUBDIR}/speedrun_results.json"
        echo -e "  ${GREEN}✓ Results copied to: ${RESULTS_SUBDIR}/speedrun_results.json${NC}"
    else
        echo -e "  ${YELLOW}⚠ No speedrun_results.json found in checkpoint directory${NC}"
    fi

    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${GREEN}✓ Checkpoint processed successfully!${NC}"
    echo -e "${BLUE}========================================${NC}"
fi

# Final summary
echo -e "\n${BLUE}========================================${NC}"
echo -e "${GREEN}✓ Complete pipeline finished successfully!${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "\n${YELLOW}Summary:${NC}"
echo -e "  📊 Training: COMPLETE"
echo -e "  📤 WandB Sync: COMPLETE"
echo -e "  📁 Results: Available in checkpoint directories"
echo -e "\n"
