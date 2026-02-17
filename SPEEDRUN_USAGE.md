# Speedrun Scripts Usage Guide

This guide explains how to use the updated `run.sh` and `run_offline.sh` scripts with parameterized experiment names.

## Quick Start

### Running a Speedrun Experiment

#### Option 1: Direct offline execution (for testing/development)
```bash
bash run_offline.sh trial_llama_75m_adamax
```

#### Option 2: SLURM job submission (for cluster)
```bash
sbatch run.sh trial_llama_75m_adamax
```

#### Option 3: Direct bash execution (alternative)
```bash
bash run.sh trial_llama_75m_adamax
```

## How It Works

### File Structure

For an experiment named `trial_llama_75m_adamax`, the expected directory structure is:
```
experiments/speedrun/
└── trial_llama_75m_adamax/
    └── trial_llama_75m_adamax.py    # Main training script
```

The Python file should contain the `speedrun_config` that will be imported.

### Script Parameters

Both `run.sh` and `run_offline.sh` accept a single parameter: the experiment name.

**Experiment Name Format:**
- Use lowercase with underscores: `my_experiment_name`
- The directory name and Python filename must match
- Example: `trial_llama_75m_adamax` → `experiments/speedrun/trial_llama_75m_adamax/trial_llama_75m_adamax.py`

### What Each Script Does

#### `run_offline.sh <experiment_name>`
1. **Setup environment**: Activates the Python virtual environment
2. **Run training**: Executes `experiments/speedrun/<experiment_name>/<experiment_name>.py`
3. **Find checkpoint**: Locates the training checkpoint directory
4. **Sync to WandB**: Uploads training results to Weights & Biases
5. **Generate results**: Creates `speedrun_results.json` with metrics
6. **Copy results**: Moves results to the submission directory

#### `run.sh <experiment_name>`
- Wrapper script for SLURM job submission
- Sets up job name, output files, and GPU resources
- Calls `run_offline.sh` with the experiment name
- Useful for cluster/HPC environments with SLURM

## Examples

### Example 1: Run trial_llama_75m_adamax

```bash
# Direct execution (fastest for testing)
bash run_offline.sh trial_llama_75m_adamax

# Or with SLURM (for cluster)
sbatch run.sh trial_llama_75m_adamax
```

### Example 2: Create and run a new experiment

1. Create the experiment directory:
```bash
mkdir -p experiments/speedrun/my_new_experiment
```

2. Copy your training script:
```bash
cp experiments/speedrun/llama_75m_adamax/llama_75m_adamax.py \
   experiments/speedrun/my_new_experiment/my_new_experiment.py
```

3. Edit the Python script (update author info, hyperparameters, etc.)

4. Run the experiment:
```bash
bash run_offline.sh my_new_experiment
```

## Output Structure

After running an experiment, the following files are generated:

```
output/
├── checkpoints/speedrun/
│   └── my_experiment_name-<hash>/
│       ├── hf/                      # HuggingFace checkpoint format
│       └── speedrun_results.json    # Final metrics and results

wandb/
└── offline-run-*-<run_id>/         # WandB offline data

experiments/speedrun/my_experiment_name/
└── speedrun_results.json           # Copy of final results
```

## Environment Variables

Before running, ensure these are set (for WandB sync):

```bash
export WANDB_ENTITY="your_entity"
export WANDB_PROJECT="your_project"
```

## Troubleshooting

### "Could not find checkpoint directory"
- Check that the training actually completed successfully
- Verify the experiment name matches the directory/file names

### "Could not find WandB offline directory"
- Ensure `WANDB_ENTITY` and `WANDB_PROJECT` environment variables are set
- Check that WandB data was saved to the expected location

### Training script not found
- Verify the experiment directory structure:
  ```
  experiments/speedrun/<experiment_name>/<experiment_name>.py
  ```
- The directory name and Python filename must match exactly

## Notes

- The scripts assume the training script is in: `experiments/speedrun/<exp_name>/<exp_name>.py`
- Each experiment gets its own checkpoint and results directory
- Results are automatically copied to the submission directory
- The scripts use color-coded output for better readability
