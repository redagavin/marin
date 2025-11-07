#!/bin/bash
# Marin Speedrun Training Pipeline
# This script runs the complete training pipeline and generates results

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Marin Speedrun Training Pipeline${NC}"
echo -e "${BLUE}========================================${NC}"

# Configuration
SUBMISSION_NAME="hello_world_int8"
MARIN_DIR="/projects/arjunguha-research-group/wu.zixua/marin"
OUTPUT_PREFIX="/projects/arjunguha-research-group/wu.zixua/marin/local_store"
VENV_PATH="${MARIN_DIR}/.venv"
TRAIN_SCRIPT="${MARIN_DIR}/experiments/speedrun/${SUBMISSION_NAME}/train.py"
WANDB_ENTITY="wu-zixua-northeastern-university"

# Step 1: Setup environment
echo -e "\n${YELLOW}[1/5] Setting up environment...${NC}"
source "${VENV_PATH}/bin/activate"

# Unset proxy variables (they cause JAX to hang)
unset ftp_proxy https_proxy http_proxy

#---#
# Problem with node not in list: i think this problem shows if you are on a SLURM node that is not a login node. If you are on a login node you might be just fine.
# Unset ALL SLURM variables: this might not be needed
for var in $(env | grep '^SLURM_' | cut -d= -f1); do
    unset $var
done
# Force single-GPU JAX mode: again, this might not be needed.
export JAX_PLATFORM_NAME=gpu
export CUDA_VISIBLE_DEVICES=0
#---#

# Set offline modes
export HF_HUB_OFFLINE=1
export WANDB_MODE=offline

echo -e "${GREEN}✓ Environment configured${NC}"

# Step 2: Cleanup previous runs
echo -e "\n${YELLOW}[2/4] Cleaning up previous (failed) runs...${NC}"
rm -rf "${OUTPUT_PREFIX}/checkpoints/speedrun/${SUBMISSION_NAME}"*
rm -rf "${OUTPUT_PREFIX}/speedrun/${SUBMISSION_NAME}"*
rm -rf "${OUTPUT_PREFIX}/experiments/${SUBMISSION_NAME}"*
MARIN_DIR="/projects/arjunguha-research-group/wu.zixua/marin"
rm -rf "${MARIN_DIR}/wandb/offline-run-*-${SUBMISSION_NAME}"*

echo -e "${GREEN}✓ Cleanup complete!"

# Step 3: Run sweep training
echo -e "\n${YELLOW}[3/4] Running training sweep...${NC}"
cd "${MARIN_DIR}"

python "${TRAIN_SCRIPT}" --prefix "${OUTPUT_PREFIX}" --force_run_failed True

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Training completed successfully${NC}"
else
    echo -e "${RED}✗ Training failed${NC}"
    exit 1
fi

echo -e "${GREEN}Sweep training completed!${NC}"