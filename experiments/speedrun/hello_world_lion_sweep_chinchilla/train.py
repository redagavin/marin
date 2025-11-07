"""
Hello World Sweep Speedrun Submission
Trains multiple Llama model sizes with the Lion optimizer (Chinchilla optimal).
"""
from experiments.llama import (
    llama_nano, llama_30m, llama_50m, llama_75m, llama_150m, llama_300m
)
from experiments.simple_train_config import SimpleTrainConfig
from marin.resources import GpuConfig
from marin.speedrun.speedrun import Author, SpeedrunConfig, default_speedrun
from levanter.optim import LionConfig
from marin.execution.executor import executor_main
import gc
import ray

AUTHOR = Author(
    name="arya",
    affiliation="Northeastern University",
    url="https://aryawu0513.github.io/"
)

def get_num_train_steps(param_count, batch_size, seq_len):
    """Compute the number of steps for Chinchilla optimal training (20x params tokens)."""
    total_tokens = param_count * 20
    tokens_per_step = batch_size * seq_len
    return total_tokens // tokens_per_step

# Model sweep with parameter counts and configs
MODEL_SWEEP = [
    ("llama_nano", llama_nano, 8_241_312),       # ~8M params
    ("llama_30m", llama_30m, 33_784_960),        # ~34M params
    ("llama_50m", llama_50m, 50_874_048),        # ~51M params
    ("llama_75m", llama_75m, 73_269_632),        # ~73M params
]

BATCH_SIZE = 8

LEARNING_RATES = {
    "llama_nano": 1e-3,
    "llama_30m": 6e-4,
    "llama_50m": 5e-4,
    "llama_75m": 4e-4,
}

WEIGHT_DECAY = 0.1

# Expose SpeedrunConfig objects at module level for each model
SPEEDRUN_CONFIGS = {}

for name, model_cfg, param_count in MODEL_SWEEP:
    
    # Calculate Chinchilla optimal steps
    seq_len = model_cfg.seq_len
    num_train_steps = get_num_train_steps(param_count, BATCH_SIZE, seq_len)
    learning_rate = LEARNING_RATES[name]  # Per-model LR
    
    # Set reasonable eval frequency (every ~20% of training)
    steps_per_eval = max(1000, num_train_steps // 5)
    
    print(f"{name}: {param_count:,} params, {num_train_steps:,} steps, eval every {steps_per_eval:,} steps")
    
    train_config = SimpleTrainConfig(
        GpuConfig(gpu_count=1, accelerator_type="A100"),
        train_batch_size=BATCH_SIZE,
        num_train_steps=num_train_steps,
        learning_rate=learning_rate,
        weight_decay=WEIGHT_DECAY,
        steps_per_eval=steps_per_eval,
        optimizer_config=LionConfig(),
    )
    
    SPEEDRUN_CONFIGS[name] = SpeedrunConfig(
        author=AUTHOR,
        description=f"Chinchilla optimal run with Lion optimizer ({name})",
        model_config=model_cfg,
        train_config=train_config,
    )

if __name__ == "__main__":
    # Run each model SEQUENTIALLY in size order with memory clearing between runs
    for name, speedrun_config in SPEEDRUN_CONFIGS.items():
        submission_name = f"hello_world_lion_sweep_chinchilla_{name}"
        
        print(f"\n{'='*60}")
        print(f"Training {name}")
        print(f"{'='*60}\n")
        
        speedrun_config.print_run_info()
        
        # Generate steps for this specific model
        steps = default_speedrun(submission_name, speedrun_config)
        
        # Run training for this model
        executor_main(steps=steps)
        
        # Clear GPU memory after each model completes
        print(f"\n🧹 Clearing GPU memory after {name}...")
        print(f"\n🧹 Shutting down Ray after {name}...")
        ray.shutdown()
        gc.collect()
        print(f"✓ shutdown complete\n")