#!/bin/bash
# Marin Speedrun EVAL Pipeline - Temporary fix for llama_75m
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
SUBMISSION_NAME="hello_world_lion_sweep_chinchilla"
MARIN_DIR="/projects/arjunguha-research-group/wu.zixua/marin"
OUTPUT_PREFIX="/projects/arjunguha-research-group/wu.zixua/marin/local_store"
VENV_PATH="${MARIN_DIR}/.venv"
WANDB_ENTITY="wu-zixua-northeastern-university"
SUBMISSION_DIR="${MARIN_DIR}/experiments/speedrun/${SUBMISSION_NAME}"

# Hardcoded values for llama_75m
NAME="llama_75m"
RUN_ID="hello_world_lion_sweep_chinchilla_llama_75m-a00595"
WANDB_OFFLINE_DIR="/projects/arjunguha-research-group/wu.zixua/marin/wandb/offline-run-20251106_143247-hello_world_lion_sweep_chinchilla_llama_75m-a00595"

# Step 1: Setup environment
echo -e "\n${YELLOW}[1/4] Setting up environment...${NC}"
source "${VENV_PATH}/bin/activate"
cd "${MARIN_DIR}"
echo -e "${GREEN}✓ Environment configured${NC}"

# Step 3: Sync to WandB
echo -e "\n${YELLOW}[3/4] Syncing to WandB...${NC}"
if [ ! -d "$WANDB_OFFLINE_DIR" ]; then
    echo -e "${RED}✗ WandB directory not found: ${WANDB_OFFLINE_DIR}${NC}"
    exit 1
fi

echo -e "${BLUE}  Syncing: ${WANDB_OFFLINE_DIR}${NC}"
wandb sync "${WANDB_OFFLINE_DIR}" --entity "${WANDB_ENTITY}" --project "marin"
echo -e "${GREEN}✓ WandB sync completed${NC}"

# Step 4: Generate speedrun_results.json
echo -e "\n${YELLOW}[4/4] Generating speedrun_results.json...${NC}"
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
    output_path='${SUBMISSION_DIR}/speedrun_results_${NAME}.json'
)

print('Generating speedrun_results.json for ${NAME}...')
speedrun_results(config)
print('Done!')
EOF


# Summary
echo -e "\n${BLUE}========================================${NC}"
echo -e "${GREEN}Evaluation completed for ${NAME}!${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "\n${BLUE}WandB Run:${NC}"
echo -e "  🔗 https://wandb.ai/${WANDB_ENTITY}/marin/runs/${RUN_ID}"
echo -e "\n${GREEN}Ready for submission!${NC}\n"