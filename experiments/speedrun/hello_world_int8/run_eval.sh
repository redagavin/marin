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
echo -e "${BLUE}Marin Speedrun EVAL Pipeline${NC}"
echo -e "${BLUE}========================================${NC}"

# Configuration
SUBMISSION_NAME="hello_world_int8"
MARIN_DIR="/projects/arjunguha-research-group/wu.zixua/marin"
OUTPUT_PREFIX="/projects/arjunguha-research-group/wu.zixua/marin/local_store"
VENV_PATH="${MARIN_DIR}/.venv"
TRAIN_SCRIPT="${MARIN_DIR}/experiments/speedrun/${SUBMISSION_NAME}/train.py"
WANDB_ENTITY="wu-zixua-northeastern-university"
SUBMISSION_DIR="${MARIN_DIR}/experiments/speedrun/${SUBMISSION_NAME}"

# Step 1: Setup environment
echo -e "\n${YELLOW}[1/5] Setting up environment...${NC}"
source "${VENV_PATH}/bin/activate"

cd "${MARIN_DIR}"
echo -e "${GREEN}✓ Environment configured${NC}"

# Step 2: Loop through each model in the sweep
MODEL_NAMES=("llama_nano" "llama_30m" "llama_50m" "llama_75m")
echo -e "\n${YELLOW}[2/5] Starting evaluation for each model in the sweep...${NC}"

for NAME in "${MODEL_NAMES[@]}"; do
    SUBMISSION_NAME_PER_MODEL="${SUBMISSION_NAME}_${NAME}"
    echo -e "\n${YELLOW}Processing evaluation for ${NAME}...${NC}"

    # Find checkpoint
    CHECKPOINT_DIR=$(find "${OUTPUT_PREFIX}/checkpoints/speedrun" -name "${SUBMISSION_NAME_PER_MODEL}-*" -type d | head -1)
    if [ -z "$CHECKPOINT_DIR" ]; then
        echo -e "${RED}✗ Checkpoint not found for ${NAME}, skipping.${NC}"
        continue
    fi
    echo -e "${GREEN}✓ Found checkpoint: ${CHECKPOINT_DIR}${NC}"

    # Extract run ID from checkpoint directory name
    RUN_ID=$(basename "${CHECKPOINT_DIR}")
    echo -e "${BLUE}  Run ID: ${RUN_ID}${NC}"

    # Sync to WandB
    # WANDB_OFFLINE_DIR=$(find "${MARIN_DIR}/wandb" -name "offline-run-*-${RUN_ID}" -type d | head -1)
    WANDB_OFFLINE_DIR=$(find "${MARIN_DIR}/wandb" -name "offline-run-*-${RUN_ID}" -type d -exec du -s {} + | sort -rn | head -1 | awk '{print $2}')
    
    if [ -z "$WANDB_OFFLINE_DIR" ]; then
        echo -e "${RED}✗ Could not find WandB offline directory for ${NAME}${NC}"
        continue
    fi
    wandb sync "${WANDB_OFFLINE_DIR}" --entity "${WANDB_ENTITY}" --project "marin"
    echo -e "${GREEN}✓ WandB sync completed for ${NAME}${NC}"


    # Generate speedrun_results.json
    # Generate speedrun_results.json
    python << EOF
from marin.speedrun.speedrun import speedrun_results, SpeedrunResultsConfig
from experiments.speedrun.${SUBMISSION_NAME}.train import SPEEDRUN_CONFIGS

# Get the specific config for this model
speedrun_config = SPEEDRUN_CONFIGS['${NAME}']

config = SpeedrunResultsConfig(
    wandb_run_id='${RUN_ID}',
    wandb_entity='${WANDB_ENTITY}',
    wandb_project='marin',
    speedrun_config=speedrun_config,
    output_path='${CHECKPOINT_DIR}/speedrun_results.json'
)

print('Generating speedrun_results.json for ${NAME}...')
speedrun_results(config)
print('Done!')
EOF

    cp "${CHECKPOINT_DIR}/speedrun_results.json" "${SUBMISSION_DIR}/speedrun_results_${NAME}.json"
    echo -e "${GREEN}✓ Results copied to submission directory for ${NAME}${NC}"
done

# Summary
echo -e "\n${BLUE}========================================${NC}"
echo -e "${GREEN}Pipeline completed successfully!${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "\n${BLUE}Submission files:${NC}"
echo -e "  📄 Training script: ${SUBMISSION_DIR}/train.py"
echo -e "  📊 Results file:    ${SUBMISSION_DIR}/speedrun_results.json"
echo -e "  📋 README:          ${SUBMISSION_DIR}/README.md"
echo -e "  💾 Checkpoint:      ${CHECKPOINT_DIR}/hf/step-99/"
echo -e "\n${BLUE}WandB Run:${NC}"
echo -e "  🔗 https://wandb.ai/${WANDB_ENTITY}/marin/runs/${RUN_ID}"
echo -e "\n${GREEN}Ready for submission to Marin leaderboard!${NC}\n"
