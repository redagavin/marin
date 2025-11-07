"""
Hello World Speedrun Submission

This is a minimal speedrun submission to demonstrate the submission process.
"""

from experiments.llama import llama_nano
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
    description="first hello world run",
    model_config=llama_nano,
    train_config=SimpleTrainConfig(
        GpuConfig(gpu_count=1, accelerator_type="A100"),
        train_batch_size=32,
        num_train_steps=10,
        learning_rate=3e-3,
        weight_decay=0.1,
        steps_per_eval=1,
    ),
)

# Print configuration info before running
speedrun_config.print_run_info()

# Run the speedrun
if __name__ == "__main__":
    executor_main(steps=default_speedrun("hello_world", speedrun_config))
