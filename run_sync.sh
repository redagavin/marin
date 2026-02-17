#!/bin/bash
#SBATCH -p frink                      # GPU partition
#SBATCH --time=8:00:00
#SBATCH --mem=100G
#SBATCH --gres=gpu:quadro:1
#SBATCH --cpus-per-task=4                  # Number of CPUs
#SBATCH -N 1                                # Number of nodes
#SBATCH -n 1                               # Number of tasks
#SBATCH -o marin_llama50m_int8_sync%j.txt                    # Standard output file (jobid will be appended)
#SBATCH -e marin_llama50m_int8_sync%j.txt                     # Standard error file (jobid will be appended)
#SBATCH -J marin_llama50m_int8_sync                # Job name

set -e  # Exit on error

# Sync checkpoints to WandB and generate speedrun_results.json
# This script handles steps 3-5 from run_offline.sh
# Usage: ./sync_results.sh <submission_dir> [wandb_entity] [wandb_project]
# Example: ./sync_results.sh trial_sophia_llama_scaling rice marin-speedrun


# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}WandB Sync & Results Generation${NC}"
echo -e "${BLUE}========================================${NC}"

# Check if submission_dir is provided
if [ -z "$1" ]; then
    echo -e "${RED}Error: SUBMISSION_DIR not provided${NC}"
    echo "Usage: $0 <submission_dir> [wandb_entity] [wandb_project]"
    echo "Example: $0 trial_sophia_llama_scaling rice marin-speedrun"
    exit 1
fi

SUBMISSION_DIR="$1"
SUBMISSION_NAME=$(basename "${SUBMISSION_DIR}")

# Get WandB credentials (use defaults or provided arguments)
WANDB_ENTITY="${2:-${WANDB_ENTITY:-xilin_wang1}}"
WANDB_PROJECT="${3:-${WANDB_PROJECT:-marin-speedrun}}"

MARIN_DIR="/projects/frink/wang.xil/marin"
OUTPUT_PREFIX="/projects/frink/wang.xil/marin/output"
VENV_PATH="${MARIN_DIR}/.venv"

# Handle both absolute and relative paths
if [[ "$SUBMISSION_DIR" = /* ]]; then
    FULL_SUBMISSION_DIR="$SUBMISSION_DIR"
else
    FULL_SUBMISSION_DIR="${MARIN_DIR}/experiments/speedrun/${SUBMISSION_DIR}"
fi

echo -e "${BLUE}Submission name: ${SUBMISSION_NAME}${NC}"
echo -e "${BLUE}Submission dir: ${FULL_SUBMISSION_DIR}${NC}"
echo -e "${BLUE}WandB entity: ${WANDB_ENTITY}${NC}"
echo -e "${BLUE}WandB project: ${WANDB_PROJECT}${NC}"

# Step 1: Setup environment
echo -e "\n${YELLOW}[1/3] Setting up environment...${NC}"
source "${VENV_PATH}/bin/activate"

echo -e "${GREEN}✓ Environment configured${NC}"



# Step 2: Find all checkpoint directories in this submission
echo -e "\n${YELLOW}[2/3] Finding checkpoint directories...${NC}"

CHECKPOINT_DIRS=$(find "${OUTPUT_PREFIX}/checkpoints/speedrun" -name "${SUBMISSION_NAME}-*" -type d 2>/dev/null || true)

if [ -z "$CHECKPOINT_DIRS" ]; then
    echo -e "${RED}✗ No checkpoint directories found for ${SUBMISSION_NAME}${NC}"
    echo "Searched in: ${OUTPUT_PREFIX}/checkpoints/speedrun"
    exit 1
fi

# Count checkpoints found
CHECKPOINT_COUNT=$(echo "$CHECKPOINT_DIRS" | wc -l)
echo -e "${GREEN}✓ Found ${CHECKPOINT_COUNT} checkpoint(s)${NC}"



# Process each checkpoint
echo -e "\n${YELLOW}[3/3] Processing checkpoints (sync to WandB & generate results)...${NC}"

PROCESSED_COUNT=0

while IFS= read -r CHECKPOINT_DIR; do
    if [ -z "$CHECKPOINT_DIR" ]; then
        continue
    fi

    PROCESSED_COUNT=$((PROCESSED_COUNT + 1))
    echo -e "\n${BLUE}Processing checkpoint ${PROCESSED_COUNT}/${CHECKPOINT_COUNT}: ${CHECKPOINT_DIR}${NC}"

    # Extract run ID from checkpoint directory name
    RUN_ID=$(basename "${CHECKPOINT_DIR}")
    echo -e "  Run ID: ${RUN_ID}"

    # Step 3a: Sync to WandB
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

    # Step 3b: Generate speedrun_results.json
    echo -e "  ${YELLOW}→ Generating speedrun_results.json...${NC}"

    # Extract the experiment identifier from run ID (e.g., "llama_130m_sophia_4096-xxxxx" -> "llama_130m_sophia_4096")
    # This helps find the correct config to import
    RUN_PREFIX=$(echo "${RUN_ID}" | sed 's/-[0-9]*$//')

    # Try to determine which model size this checkpoint is for
    if [[ "${RUN_ID}" =~ "130m" ]]; then
        MODEL_SIZE="130m"
    elif [[ "${RUN_ID}" =~ "300m" ]]; then
        MODEL_SIZE="300m"
    elif [[ "${RUN_ID}" =~ "520m" ]]; then
        MODEL_SIZE="520m"
    elif [[ "${RUN_ID}" =~ "1_2b" ]]; then
        MODEL_SIZE="1_2b"
    else
        MODEL_SIZE=""
    fi

    # Determine output directory (use model size subdirectory)
    MODEL_SIZE_DIR="${FULL_SUBMISSION_DIR}/${MODEL_SIZE}"
    if [ ! -d "${MODEL_SIZE_DIR}" ]; then
        echo -e "  ${YELLOW}⚠ Model size subdirectory not found: ${MODEL_SIZE_DIR}${NC}"
        echo -e "  Creating directory..."
        mkdir -p "${MODEL_SIZE_DIR}"
    fi

    # Generate results using Python
#     python << EOF
# import sys
# import os
# sys.path.insert(0, '${MARIN_DIR}')

# try:
#     from marin.speedrun.speedrun import speedrun_results, SpeedrunResultsConfig
#     from experiments.speedrun.${SUBMISSION_NAME}.${SUBMISSION_NAME} import build_config

#     print(f"  Generating results for ${RUN_ID}...")

#     # Determine which model size this is for
#     model_size = "${MODEL_SIZE}"
#     if not model_size:
#         print(f"  ${RED}✗ Could not determine model size from run ID${NC}")
#         sys.exit(1)

#     # Build the config for this model size
#     name, speedrun_config = build_config(model_size)

#     config = SpeedrunResultsConfig(
#         wandb_run_id='${RUN_ID}',
#         wandb_entity='${WANDB_ENTITY}',
#         wandb_project='${WANDB_PROJECT}',
#         speedrun_config=speedrun_config,
#         output_path='${CHECKPOINT_DIR}/speedrun_results.json'
#     )

#     speedrun_results(config)
#     print(f"  ${GREEN}✓ Results generated${NC}")

# except ImportError as e:
#     print(f"  ${YELLOW}⚠ Note: Could not import config (this is OK if using a different optimizer)${NC}")
#     print(f"    Error: {e}")
# except Exception as e:
#     print(f"  ${RED}✗ Error generating results: {e}${NC}")
#     sys.exit(1)
# EOF

    python << EOF
from marin.speedrun.speedrun import speedrun_results, SpeedrunResultsConfig
from experiments.speedrun.${SUBMISSION_NAME}.${SUBMISSION_NAME} import speedrun_config

config = SpeedrunResultsConfig(
    wandb_run_id='${RUN_ID}',
    wandb_entity='${WANDB_ENTITY}',
    wandb_project='${WANDB_PROJECT}',  # Note: actual project name, not marin-speedrun
    speedrun_config=speedrun_config,
    output_path='${CHECKPOINT_DIR}/speedrun_results.json'
)

print('Generating speedrun_results.json...')
speedrun_results(config)
print('Done!')
EOF

    if [ $? -eq 0 ]; then
        # Copy results to model size subdirectory
        if [ -d "${MODEL_SIZE_DIR}" ]; then
            cp "${CHECKPOINT_DIR}/speedrun_results.json" "${MODEL_SIZE_DIR}/speedrun_results.json"
            echo -e "  ${GREEN}✓ Results copied to ${MODEL_SIZE_DIR}/speedrun_results.json${NC}"
        else
            echo -e "  ${YELLOW}⚠ Model size directory not found: ${MODEL_SIZE_DIR}${NC}"
        fi
    else
        echo -e "  ${YELLOW}⚠ Results generation skipped (may be OK for custom configs)${NC}"
    fi

done <<< "$CHECKPOINT_DIRS"

# Summary
echo -e "\n${BLUE}========================================${NC}"
echo -e "${GREEN}Processed ${PROCESSED_COUNT} checkpoint(s) successfully!${NC}"
echo -e "${BLUE}========================================${NC}"
echo -e "\n${BLUE}Results organized by model size:${NC}"
echo -e "  📂 Directory: ${FULL_SUBMISSION_DIR}"
if [ -d "${FULL_SUBMISSION_DIR}" ]; then
    for size_dir in "${FULL_SUBMISSION_DIR}"/{130m,300m,520m,1_2b}; do
        if [ -d "$size_dir" ]; then
            size_name=$(basename "$size_dir")
            if [ -f "$size_dir/speedrun_results.json" ]; then
                echo -e "  ✓ ${size_name}/speedrun_results.json"
            fi
        fi
    done
fi
echo -e "\n${GREEN}Done!${NC}\n"
