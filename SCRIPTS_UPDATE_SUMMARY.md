# Scripts Update Summary

## Changes Made

Both `run.sh` and `run_offline.sh` have been updated to accept experiment names as parameters, eliminating the need to manually edit them for each new experiment.

### Before (Old Approach)

**run_offline.sh (hardcoded):**
```bash
SUBMISSION_NAME="trial"
SUBMISSION_PY="trial"
TRAIN_SCRIPT="${MARIN_DIR}/experiments/speedrun/${SUBMISSION_NAME}/${SUBMISSION_PY}.py"
```

**To run a different experiment:** Had to edit the script, changing `SUBMISSION_NAME` and `SUBMISSION_PY` every time.

### After (New Approach)

**run_offline.sh (parameterized):**
```bash
EXPERIMENT_NAME="$1"                    # Accept as parameter
SUBMISSION_NAME="${EXPERIMENT_NAME}"
SUBMISSION_PY="${EXPERIMENT_NAME}"
TRAIN_SCRIPT="${MARIN_DIR}/experiments/speedrun/${SUBMISSION_NAME}/${SUBMISSION_PY}.py"
```

**To run a different experiment:** Just pass the experiment name as an argument!

```bash
bash run_offline.sh trial_llama_75m_adamax
bash run_offline.sh my_new_experiment
bash run_offline.sh another_experiment
```

## Key Changes in Detail

### `run.sh`

**Added:**
- Parameter validation: checks if experiment name is provided
- Usage documentation in the script header
- Passes experiment name to `run_offline.sh`

**Before:**
```bash
bash run_offline.sh
```

**After:**
```bash
bash run_offline.sh "${EXPERIMENT_NAME}"
```

### `run_offline.sh`

**Added:**
- Parameter validation with helpful error messages
- Usage documentation in script header
- Dynamic checkpoint directory discovery with fallback patterns
- Echo statement showing which experiment is running

**Key additions:**
```bash
# Check if experiment name is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <experiment_name>"
    echo "Example: $0 trial_llama_75m_adamax"
    exit 1
fi

EXPERIMENT_NAME="$1"

# Dynamic checkpoint search with fallback
CHECKPOINT_DIR=$(find "${OUTPUT_PREFIX}/checkpoints/speedrun" -name "${SUBMISSION_NAME}-*" -type d | head -1)
if [ -z "$CHECKPOINT_DIR" ]; then
    CHECKPOINT_DIR=$(find "${OUTPUT_PREFIX}/checkpoints/speedrun" -name "${SUBMISSION_NAME:0:3}*" -type d | head -1)
fi
```

## Usage Comparison

### Old Way (Before)
```bash
# 1. Edit run_offline.sh manually
vim run_offline.sh
# Change: SUBMISSION_NAME="trial" → SUBMISSION_NAME="my_experiment"
# Change: SUBMISSION_PY="trial" → SUBMISSION_PY="my_experiment"

# 2. Run the script
bash run_offline.sh
```

### New Way (After)
```bash
# Just pass the experiment name as an argument!
bash run_offline.sh my_experiment
```

**Time saved per experiment:** 30 seconds to 1 minute of manual editing ✓

## Backward Compatibility

The scripts will error gracefully if no experiment name is provided:
```bash
$ bash run_offline.sh
Error: Experiment name not provided
Usage: run_offline.sh <experiment_name>
Example: run_offline.sh trial_llama_75m_adamax
```

## Files Modified

1. `/projects/frink/wang.xil/marin/run.sh`
2. `/projects/frink/wang.xil/marin/run_offline.sh`

## Files Created

1. `/projects/frink/wang.xil/marin/SPEEDRUN_USAGE.md` - Detailed usage guide
2. `/projects/frink/wang.xil/marin/SCRIPTS_UPDATE_SUMMARY.md` - This file

## Next Steps

1. Update your experiment scripts in `experiments/speedrun/` if needed
2. Follow the naming convention: directory and Python filename must match
3. Use `bash run_offline.sh <experiment_name>` to run experiments
4. Refer to `SPEEDRUN_USAGE.md` for detailed examples and troubleshooting
