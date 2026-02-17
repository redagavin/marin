# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Speedrun code for batch size ablation study using a 50M parameter model with FIXED learning rate.
Trains on Fineweb-Edu dataset with 2x Chinchilla optimal tokens (2B tokens).
Accepts batch_size as a command-line argument but keeps learning rate constant at 0.020.
"""

import logging
import sys

from levanter.optim import MuonConfig

from experiments.llama import llama_50m
from experiments.simple_train_config import SimpleTrainConfig
from marin.execution.executor import executor_main
from marin.resources import GpuConfig
from marin.speedrun.speedrun import Author, SpeedrunConfig, default_speedrun


logger = logging.getLogger("ray")


def calculate_hyperparameters_fixed_lr(
    batch_size: int, param_count: int = 50_000_000, seq_len: int = 1024, warmup_fraction: float = 0.05
):
    """
    Calculate hyperparameters based on batch size with FIXED learning rate.

    Args:
        batch_size: The batch size to use
        param_count: Number of model parameters (default: 50M)
        seq_len: Sequence length (default: 1024)
        warmup_fraction: Fraction of training steps to use for warmup (default: 0.05 = 5%)

    Returns:
        Dictionary with:
        - num_steps: Number of training steps for 2x Chinchilla optimal tokens
        - learning_rate: Fixed learning rate (0.020)
        - warmup_steps: Number of warmup steps (warmup_fraction * num_steps)
        - total_tokens: Total tokens for 2x Chinchilla (40x params)
    """
    # 2x Chinchilla optimal = 40x params tokens
    total_tokens = param_count * 40
    tokens_per_step = batch_size * seq_len
    num_steps = total_tokens // tokens_per_step

    # Keep learning rate FIXED regardless of batch size
    learning_rate = 0.020

    # Calculate warmup steps as a fraction of total training steps
    warmup_steps = int(num_steps * warmup_fraction)

    return {
        "num_steps": num_steps,
        "learning_rate": learning_rate,
        "warmup_steps": warmup_steps,
        "total_tokens": total_tokens,
        "tokens_per_step": tokens_per_step,
    }


def get_batch_size_from_args() -> int:
    """
    Parse batch_size from command-line arguments.
    Expects batch_size as the first positional argument before other flags.
    Removes it from sys.argv after parsing so executor_main doesn't see it.

    Returns:
        batch_size as an integer

    Raises:
        SystemExit if batch_size is not provided or invalid
    """
    # Check if first argument is a number (batch_size)
    if len(sys.argv) < 2 or not sys.argv[1].lstrip('-').isdigit():
        print("Usage: python trial_batch_size_muon_fixed_lr.py <batch_size> [--prefix ...]")
        print("Example: python trial_batch_size_muon_fixed_lr.py 256 --prefix /output")
        print("\nRecommended batch sizes: 1, 2, 4, 8, 16, 32, 64, 128, 256")
        print("Note: Batch size 512 may cause OOM on H200. Use smaller sizes or multiple GPUs.")
        sys.exit(1)

    try:
        batch_size = int(sys.argv[1])
        if batch_size <= 0:
            raise ValueError("Batch size must be positive")
        # Remove batch_size from sys.argv so executor_main doesn't see it
        sys.argv.pop(1)
        return batch_size
    except ValueError as e:
        print(f"Error: Invalid batch size '{sys.argv[1]}': {e}")
        sys.exit(1)


# Get batch size from command line
BATCH_SIZE = get_batch_size_from_args()
PARAM_COUNT = 50_000_000
SEQ_LEN = llama_50m.seq_len

# Calculate hyperparameters with fixed learning rate
hp = calculate_hyperparameters_fixed_lr(BATCH_SIZE, PARAM_COUNT, SEQ_LEN)
NUM_TRAIN_STEPS = hp["num_steps"]
LEARNING_RATE = hp["learning_rate"]
WARMUP_STEPS = hp["warmup_steps"]

print(f"\n{'='*60}")
print(f"Batch Size Ablation Study - 50M Model (FIXED LR)")
print(f"{'='*60}")
print(f"Batch Size:       {BATCH_SIZE}")
print(f"Learning Rate:    {LEARNING_RATE:.6f} (FIXED)")
print(f"Warmup Steps:     {WARMUP_STEPS:,} (5% of training steps)")
print(f"Total Tokens:     {hp['total_tokens']:,} (2x Chinchilla = 40x params)")
print(f"Tokens/Step:      {hp['tokens_per_step']:,}")
print(f"Training Steps:   {NUM_TRAIN_STEPS:,}")
print(f"{'='*60}\n")

# Muon optimizer config with fixed learning rate and scaled warmup
muon_config = MuonConfig(
    learning_rate=LEARNING_RATE,
    adam_lr=0.004,
    momentum=0.95,
    beta1=0.8,
    beta2=0.98,
    epsilon=1e-15,
    muon_epsilon=1e-5,
    max_grad_norm=1,
    warmup=WARMUP_STEPS,
    min_lr_ratio=0,
    lr_schedule="linear",
    decay=0.8,
)

speedrun_config = SpeedrunConfig(
    author=Author(
        name="Rice",  # TODO: Update with your name
        affiliation="Rice Factory",  # TODO: Update with your affiliation
        url="N/A",  # TODO: Update with your URL
    ),
    description=f"50M model batch size {BATCH_SIZE} with fixed LR (2x Chinchilla optimal tokens)",
    model_config=llama_50m,
    train_config=SimpleTrainConfig(
        GpuConfig(gpu_count=1, accelerator_type="H200"),
        train_batch_size=BATCH_SIZE,
        num_train_steps=NUM_TRAIN_STEPS,
        learning_rate=LEARNING_RATE,
        optimizer_config=muon_config,
        steps_per_eval=500,
    ),
)

speedrun_config.print_run_info()

if __name__ == "__main__":
    # Create unique experiment name based on batch size to avoid checkpoint conflicts
    experiment_name = f"trial_batch_size_muon_fixed_lr_bs{BATCH_SIZE}"
    executor_main(steps=default_speedrun(experiment_name, speedrun_config))
