"""
Phase 1 AdamH Speedrun Submission

Uses AdamH optimizer with hybrid normalization for improved training efficiency.
"""

import dataclasses

from experiments.llama import llama_75m
from experiments.simple_train_config import SimpleTrainConfig
from levanter.optim import AdamHConfig
from marin.execution.executor import executor_main
from marin.resources import GpuConfig
from marin.speedrun.speedrun import Author, SpeedrunConfig, default_speedrun


# Create modified llama_75m with hybrid normalization
llama_75m_hybrid = dataclasses.replace(
    llama_75m,
    hybrid_norm=True,
    use_qk_norm=True,
)

# Configure the speedrun with author information and training settings
speedrun_config = SpeedrunConfig(
    author=Author(
        name="redagavin",
        affiliation="Northeastern University",
        url="https://redagavin.github.io/"
    ),
    description="Phase 1 AdamH: 75M Llama with AdamH and hybrid normalization",
    model_config=llama_75m_hybrid,
    train_config=SimpleTrainConfig(
        GpuConfig(gpu_count=1, accelerator_type="A100"),
        train_batch_size=128,
        num_train_steps=3000,
        learning_rate=0.02,
        weight_decay=0.1,
        steps_per_eval=500,
        optimizer_config=AdamHConfig(
            learning_rate=0.02,
            adam_lr=0.008,
            min_lr_ratio=0,
            warmup=1000,
            beta1=0.9,
            beta2=0.98,
            epsilon=1e-10,
            max_grad_norm=1.0,
            nesterov=False,
        ),
    ),
)

# Print configuration info before running
speedrun_config.print_run_info()

# Run the speedrun
if __name__ == "__main__":
    executor_main(steps=default_speedrun("phase1_adamh", speedrun_config))
