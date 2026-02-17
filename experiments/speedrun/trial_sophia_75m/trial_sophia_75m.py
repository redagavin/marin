"""Trial speedrun code for a 75M parameter model using Sophia-H optimizer."""

import logging

from levanter.optim import SophiaHConfig

from experiments.llama import llama_75m
from experiments.simple_train_config import SimpleTrainConfig
from marin.execution.executor import executor_main
from marin.resources import GpuConfig
from marin.speedrun.speedrun import Author, SpeedrunConfig, default_speedrun


def get_num_train_steps(param_count, batch_size, seq_len):
    """Compute the number of steps for Chinchilla optimal training (20x params tokens)."""
    total_tokens = param_count * 10
    tokens_per_step = batch_size * seq_len
    return total_tokens // tokens_per_step


# --------------------- Trial speedrun Using Sophia-H -------------------------

logger = logging.getLogger("ray")
PARAM_COUNT = 75_000_000
BATCH_SIZE = 32
MODEL_CONFIG = llama_75m
SEQ_LEN = MODEL_CONFIG.seq_len
NUM_TRAIN_STEPS = get_num_train_steps(PARAM_COUNT, BATCH_SIZE, SEQ_LEN)

# Create Sophia-H optimizer config with default hyperparameters
sophia_config = SophiaHConfig(
    # Sophia-H specific
    gamma=0.01,                # Hessian scaling factor (default for Sophia-H)

    # First and second moments
    beta1=0.96,                # Momentum decay for gradients
    beta2=0.99,                # EMA decay for Hessian diagonal approximation

    # Numerical stability
    epsilon=1e-12,             # Small constant to prevent division by zero

    # Update control
    clip_threshold=1.0,        # Clip updates to [-1.0, 1.0] (None to disable)
    update_interval=10,        # Update Hessian approximation every 10 steps

    # Regularization
    weight_decay=0.1,          # L2 regularization

    # Learning rate schedule
    learning_rate=6e-4,        # Base learning rate
    warmup=2000,               # Warmup for 2000 steps
    lr_schedule="cosine",      # Use cosine annealing schedule
)

speedrun_config = SpeedrunConfig(
    author=Author(
        name="Rice",  # TODO: Update with your name
        affiliation="Rice Factory",  # TODO: Update with your affiliation
        url="N/A",  # TODO: Update with your URL
    ),
    description="Trial: 75M parameter model with Sophia-H optimizer (3000 steps)",
    model_config=llama_75m,
    train_config=SimpleTrainConfig(
        GpuConfig(gpu_count=1, accelerator_type="H200"),
        train_batch_size=BATCH_SIZE,
        num_train_steps=NUM_TRAIN_STEPS,  # Trial: 3000 steps instead of 6000
        learning_rate=6e-4,                # Using Sophia's default LR
        weight_decay=0.1,                  # Managed by sophia_config
        steps_per_eval=1000,               # Evaluate every 1000 steps
        optimizer_config=sophia_config,    # Use Sophia-H optimizer
    ),
)

speedrun_config.print_run_info()

if __name__ == "__main__":
    executor_main(steps=default_speedrun("trial_sophia_75m", speedrun_config))