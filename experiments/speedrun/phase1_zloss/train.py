"""
Phase 1 Z-Loss Speedrun Submission

75M Llama model with z-loss regularization for improved numerical stability.
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
    description="Phase 1 Z-Loss: 75M Llama with z-loss regularization",
    model_config=llama_75m,
    train_config=SimpleTrainConfig(
        GpuConfig(gpu_count=1, accelerator_type="A100"),
        train_batch_size=128,
        num_train_steps=3000,
        learning_rate=3e-3,
        weight_decay=0.1,
        steps_per_eval=500,
        z_loss_weight=1e-4,
    ),
)

# Print configuration info before running
speedrun_config.print_run_info()

# Run the speedrun
if __name__ == "__main__":
    executor_main(steps=default_speedrun("phase1_zloss", speedrun_config))
