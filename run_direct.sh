#!/bin/bash
#SBATCH -p gpu                      # GPU partition
#SBATCH --time=8:00:00
#SBATCH --mem=100G
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=4                  # Number of CPUs
#SBATCH -N 1                                # Number of nodes
#SBATCH -n 1                               # Number of tasks
#SBATCH -o marin_llama50m_int8_%j.txt                    # Standard output file (jobid will be appended)
#SBATCH -e marin_llama50m_int8_%j.txt                     # Standard error file (jobid will be appended)
#SBATCH -J marin_llama50m_int8                # Job name

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Marin Direct Training Script Runner${NC}"
echo -e "${BLUE}========================================${NC}"

## USAGE: sbatch run_direct.sh <experiment_name>
## Example: sbatch run_direct.sh trial_sophia_llama_scaling
## Example: sbatch run_direct.sh trial_llama_75m_adamax

# Check if experiment name is provided
if [ -z "$1" ]; then
    echo -e "${RED}Error: Experiment name not provided${NC}"
    echo "Usage: $0 <experiment_name>"
    echo "Example: $0 trial_sophia_llama_scaling"
    echo "Example: $0 trial_llama_75m_adamax"
    exit 1
fi

EXPERIMENT_NAME="$1"
MARIN_DIR="/projects/frink/wang.xil/marin"
OUTPUT_PREFIX="/projects/frink/wang.xil/marin/output"
VENV_PATH="${MARIN_DIR}/.venv"
TRAIN_SCRIPT="${MARIN_DIR}/experiments/speedrun/${EXPERIMENT_NAME}/${EXPERIMENT_NAME}.py"

echo -e "${BLUE}Experiment: ${EXPERIMENT_NAME}${NC}"
echo -e "${BLUE}Training script: ${TRAIN_SCRIPT}${NC}"

# Step 1: Setup environment
echo -e "\n${YELLOW}[1/3] Setting up environment...${NC}"
eval "$(/shared/EL9/explorer/anaconda3/2024.06/bin/conda shell.bash hook)"
conda activate /projects/frink/wang.xil/marin/marin_conda

# Unset proxy variables (they cause JAX to hang)
unset ftp_proxy https_proxy http_proxy

# Set offline modes
export HF_HUB_OFFLINE=1
export WANDB_MODE=offline
export XLA_PYTHON_CLIENT_PREALLOCATE=false

nvcc --version
nvidia-smi

source "${VENV_PATH}/bin/activate"

echo -e "${GREEN}✓ Environment configured${NC}"



# Step 2: Verify training script exists
echo -e "\n${YELLOW}[2/3] Verifying training script...${NC}"
if [ ! -f "${TRAIN_SCRIPT}" ]; then
    echo -e "${RED}✗ Training script not found: ${TRAIN_SCRIPT}${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Training script found${NC}"

# Step 3: Run training
echo -e "\n${YELLOW}[3/3] Running training...${NC}"
cd "${MARIN_DIR}"

python "${TRAIN_SCRIPT}" --prefix "${OUTPUT_PREFIX}"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Training completed successfully${NC}"
else
    echo -e "${RED}✗ Training failed${NC}"
    exit 1
fi

echo -e "\n${BLUE}========================================${NC}"
echo -e "${GREEN}Pipeline completed successfully!${NC}"
echo -e "${BLUE}========================================${NC}"
