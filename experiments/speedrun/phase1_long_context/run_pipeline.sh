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
SUBMISSION_NAME="phase1_long_context"
MARIN_DIR="/scratch/yang.zih/marin_speedrun/marin"
OUTPUT_PREFIX="/scratch/yang.zih/marin_speedrun/output"
ENV_FILE="/scratch/yang.zih/marin_speedrun/.env"
VENV_PATH="${MARIN_DIR}/.venv"
TRAIN_SCRIPT="${MARIN_DIR}/experiments/speedrun/${SUBMISSION_NAME}/train.py"
SUBMISSION_DIR="${MARIN_DIR}/experiments/speedrun/${SUBMISSION_NAME}"

# Step 1: Setup environment
echo -e "\n${YELLOW}[1/5] Setting up environment...${NC}"
source "${VENV_PATH}/bin/activate"
source "${ENV_FILE}"

# Unset proxy variables (they cause JAX to hang)
unset ftp_proxy https_proxy http_proxy

# Set offline modes
export HF_HUB_OFFLINE=1
export WANDB_MODE=offline

echo -e "${GREEN}✓ Environment configured${NC}"

# Step 2: Run training
echo -e "\n${YELLOW}[2/5] Running training (this takes ~3-5 minutes)...${NC}"
cd "${MARIN_DIR}"

python "${TRAIN_SCRIPT}" --prefix "${OUTPUT_PREFIX}"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Training completed successfully${NC}"
else
    echo -e "${RED}✗ Training failed${NC}"
    exit 1
fi

# Step 3: Find the checkpoint directory
echo -e "\n${YELLOW}[3/5] Locating checkpoint directory...${NC}"
CHECKPOINT_DIR=$(find "${OUTPUT_PREFIX}/checkpoints/speedrun" -name "${SUBMISSION_NAME}-*" -type d | head -1)

if [ -z "$CHECKPOINT_DIR" ]; then
    echo -e "${RED}✗ Could not find checkpoint directory${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Found checkpoint at: ${CHECKPOINT_DIR}${NC}"

# Extract run ID from checkpoint directory name
RUN_ID=$(basename "${CHECKPOINT_DIR}")
echo -e "${BLUE}  Run ID: ${RUN_ID}${NC}"

# Step 4: Sync to WandB
echo -e "\n${YELLOW}[4/5] Syncing to WandB...${NC}"
WANDB_DIR="${MARIN_DIR}/wandb/offline-run-*-${RUN_ID}"

# Find the WandB offline directory
WANDB_OFFLINE_DIR=$(find "${MARIN_DIR}/wandb" -name "offline-run-*-${RUN_ID}" -type d | head -1)

if [ -z "$WANDB_OFFLINE_DIR" ]; then
    echo -e "${RED}✗ Could not find WandB offline directory${NC}"
    exit 1
fi

echo -e "${BLUE}  Syncing: ${WANDB_OFFLINE_DIR}${NC}"
wandb sync "${WANDB_OFFLINE_DIR}"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ WandB sync completed${NC}"
else
    echo -e "${RED}✗ WandB sync failed${NC}"
    exit 1
fi

# Step 5: Generate speedrun_results.json
echo -e "\n${YELLOW}[5/5] Generating speedrun_results.json...${NC}"

python << EOF
from marin.speedrun.speedrun import speedrun_results, SpeedrunResultsConfig
from experiments.speedrun.${SUBMISSION_NAME}.train import speedrun_config

config = SpeedrunResultsConfig(
    wandb_run_id='${RUN_ID}',
    wandb_entity='marin-speedrun',
    wandb_project='marin-speedrun',
    speedrun_config=speedrun_config,
    output_path='${CHECKPOINT_DIR}/speedrun_results.json'
)

print('Generating speedrun_results.json...')
speedrun_results(config)
print('Done!')
EOF

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Results file generated${NC}"
else
    echo -e "${RED}✗ Results generation failed${NC}"
    exit 1
fi

# Copy results to submission directory
cp "${CHECKPOINT_DIR}/speedrun_results.json" "${SUBMISSION_DIR}/"
echo -e "${GREEN}✓ Results copied to submission directory${NC}"

# Summary
echo -e "\n${BLUE}========================================${NC}"
echo -e "${GREEN}Pipeline completed successfully!${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "\n${BLUE}Submission files:${NC}"
echo -e "  📄 Training script: ${SUBMISSION_DIR}/train.py"
echo -e "  📊 Results file:    ${SUBMISSION_DIR}/speedrun_results.json"
echo -e "  📋 README:          ${SUBMISSION_DIR}/README.md"
echo -e "  💾 Checkpoint:      ${CHECKPOINT_DIR}/hf/"
echo -e "\n${BLUE}WandB Run:${NC}"
echo -e "  🔗 https://wandb.ai/marin-speedrun/marin-speedrun/runs/${RUN_ID}"
echo -e "\n${GREEN}Ready for submission to Marin leaderboard!${NC}\n"
