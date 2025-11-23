#!/bin/bash
# Quick test script for srun
# Usage: srun --partition=gpu --gres=gpu:h200:1 --mem=32G --cpus-per-task=8 ./test_srun.sh

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Marin Speedrun Test (10 steps)${NC}"
echo -e "${BLUE}========================================${NC}"

# Configuration
MARIN_DIR="/scratch/yang.zih/marin_speedrun/marin"
OUTPUT_PREFIX="/scratch/yang.zih/marin_speedrun/output"
ENV_FILE="/scratch/yang.zih/marin_speedrun/.env"
VENV_PATH="${MARIN_DIR}/.venv"
TRAIN_SCRIPT="${MARIN_DIR}/experiments/speedrun/phase1_test/train.py"

# Step 1: Setup environment
echo -e "\n${YELLOW}[1/2] Setting up environment...${NC}"
source "${VENV_PATH}/bin/activate"
source "${ENV_FILE}"

# Unset proxy variables (they cause JAX to hang)
unset ftp_proxy https_proxy http_proxy

# Set offline modes
export HF_HUB_OFFLINE=1
export WANDB_MODE=offline

echo -e "${GREEN}✓ Environment configured${NC}"

# Step 2: Run training
echo -e "\n${YELLOW}[2/2] Running test training (10 steps)...${NC}"
cd "${MARIN_DIR}"

python "${TRAIN_SCRIPT}" --prefix "${OUTPUT_PREFIX}" --force-run-failed

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Test training completed successfully${NC}"
    echo -e "\n${BLUE}========================================${NC}"
    echo -e "${GREEN}Test passed! Pipeline is working.${NC}"
    echo -e "${BLUE}========================================${NC}"
else
    echo -e "${RED}✗ Test training failed${NC}"
    exit 1
fi
