"""
Phase 1 Long Context Speedrun Submission

Extended sequence length experiment (seq_len=4096) with 75M Llama model.
"""

import dataclasses

from experiments.llama import llama_75m
from experiments.simple_train_config import SimpleTrainConfig
from marin.execution.executor import executor_main
from marin.resources import GpuConfig
from marin.speedrun.speedrun import Author, SpeedrunConfig, default_speedrun


# Create modified llama_75m with longer sequence length
llama_75m_long = dataclasses.replace(llama_75m, seq_len=4096)

# Configure the speedrun with author information and training settings
speedrun_config = SpeedrunConfig(
    author=Author(
        name="redagavin",
        affiliation="Northeastern University",
        url="https://redagavin.github.io/"
    ),
    description="Phase 1 Long Context: 75M Llama with seq_len=4096",
    model_config=llama_75m_long,
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
    executor_main(steps=default_speedrun("phase1_long_context", speedrun_config))
