"""
Phase 1 Baseline Speedrun Submission

This is the baseline experiment for Phase 1, using a 75M Llama model with standard AdamW optimizer.
Serves as the reference point for evaluating optimization techniques.
"""

from experiments.llama import llama_75m
from experiments.simple_train_config import SimpleTrainConfig
from marin.execution.executor import executor_main
from marin.resources import GpuConfig
from marin.speedrun.speedrun import Author, SpeedrunConfig, default_speedrun


# Configure the speedrun with author information and training settings
speedrun_config = SpeedrunConfig(
    author=Author(
        name="redagavin",
        affiliation="Northeastern University",
        url="https://redagavin.github.io/"
    ),
    description="Phase 1 Baseline: 75M Llama with standard AdamW",
    model_config=llama_75m,
    train_config=SimpleTrainConfig(
        GpuConfig(gpu_count=1, accelerator_type="A100"),
        train_batch_size=128,
        num_train_steps=3000,
        learning_rate=3e-3,
        weight_decay=0.1,
        steps_per_eval=500,
    ),
)

# Print configuration info before running
speedrun_config.print_run_info()

# Run the speedrun
if __name__ == "__main__":
    executor_main(steps=default_speedrun("phase1_baseline", speedrun_config))
