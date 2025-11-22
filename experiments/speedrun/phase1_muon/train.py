"""
Phase 1 Muon Speedrun Submission

This submission uses the Muon optimizer with a 75M Llama model.
Muon is a momentum-based optimizer with strong empirical performance across model scales.
Configuration adapted from the 130M Muon reference submission.
"""

from experiments.llama import llama_75m
from experiments.simple_train_config import SimpleTrainConfig
from levanter.optim import MuonConfig
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
    description="Phase 1 Muon: 75M Llama with Muon optimizer",
    model_config=llama_75m,
    train_config=SimpleTrainConfig(
        GpuConfig(gpu_count=1, accelerator_type="A100"),
        train_batch_size=128,
        num_train_steps=3000,
        learning_rate=0.016,
        weight_decay=0.1,
        steps_per_eval=500,
        optimizer_config=MuonConfig(
            learning_rate=0.016,
            adam_lr=0.0032,
            weight_decay=0.1,
            min_lr_ratio=0,
            warmup=0,
            momentum=0.95,
            beta1=0.8,
            beta2=0.98,
            epsilon=1e-15,
            muon_epsilon=1e-5,
            max_grad_norm=1,
            lr_schedule="linear",
            decay=0.8,
        ),
    ),
)

# Print configuration info before running
speedrun_config.print_run_info()

# Run the speedrun
if __name__ == "__main__":
    executor_main(steps=default_speedrun("phase1_muon", speedrun_config))
