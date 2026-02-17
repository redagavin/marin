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

"""Speedruns using the Sophia-H optimizer for various Llama model sizes (Chinchilla optimal steps).

Uses default Sophia-H hyperparameters across all model sizes.
"""

import dataclasses
import logging

from levanter.optim import SophiaHConfig

from experiments.llama import llama_1_4b, llama_150m, llama_300m, llama_600m
from experiments.simple_train_config import SimpleTrainConfig
from marin.execution.executor import executor_main
from marin.resources import TpuPodConfig, GpuConfig
from marin.speedrun.speedrun import Author, SpeedrunConfig, default_speedrun

AUTHOR = Author(name="Rice", affiliation="NEU", url="")

logger = logging.getLogger("ray")


def get_num_train_steps(param_count, batch_size, seq_len):
    """Compute the number of steps for Chinchilla optimal training (20x params tokens)."""
    total_tokens = param_count * 10
    tokens_per_step = batch_size * seq_len
    return total_tokens // tokens_per_step


def build_config(size: str) -> tuple[str, SpeedrunConfig]:
    # Parameter counts
    param_counts = {
        "130m": 130_000_000,
        "300m": 300_000_000,
        "520m": 520_000_000,
        "1_2b": 1_200_000_000,
    }

    # Model configs
    model_cfgs = {
        "130m": llama_150m,
        "300m": llama_300m,
        "520m": llama_600m,
        "1_2b": llama_1_4b,
    }

    # Training batch sizes
    batch_sizes = {
        "130m": 128,
        "300m": 128,
        "520m": 128,
        "1_2b": 256,
    }

    # Resource configs
    resource_cfgs = {
        "130m": GpuConfig(gpu_count=1, accelerator_type="H200"),
        "300m": GpuConfig(gpu_count=1, accelerator_type="H200"),
        "520m": GpuConfig(gpu_count=1, accelerator_type="H200"),
        "1_2b": GpuConfig(gpu_count=1, accelerator_type="H200"),
    }

    # Sophia-H configs - using default hyperparameters across all sizes
    # Note: Learning rate may need adjustment per size, but hyperparameters stay the same
    sophia_configs = {
        "130m": SophiaHConfig(
            # Sophia-H specific (default)
            gamma=0.01,
            # First and second moments (defaults)
            beta1=0.96,
            beta2=0.99,
            # Numerical stability (default)
            epsilon=1e-12,
            # Update control (defaults)
            clip_threshold=1.0,
            update_interval=10,
            # Regularization (default)
            weight_decay=0.1,
            # Learning rate schedule
            learning_rate=6e-4,
            warmup=2000,
            lr_schedule="cosine",
        ),
        "300m": SophiaHConfig(
            gamma=0.01,
            beta1=0.96,
            beta2=0.99,
            epsilon=1e-12,
            clip_threshold=1.0,
            update_interval=10,
            weight_decay=0.1,
            learning_rate=6e-4,
            warmup=2000,
            lr_schedule="cosine",
        ),
        "520m": SophiaHConfig(
            gamma=0.01,
            beta1=0.96,
            beta2=0.99,
            epsilon=1e-12,
            clip_threshold=1.0,
            update_interval=10,
            weight_decay=0.1,
            learning_rate=6e-4,
            warmup=2000,
            lr_schedule="cosine",
        ),
        "1_2b": SophiaHConfig(
            gamma=0.01,
            beta1=0.96,
            beta2=0.99,
            epsilon=1e-12,
            clip_threshold=1.0,
            update_interval=10,
            weight_decay=0.1,
            learning_rate=6e-4,
            warmup=2000,
            lr_schedule="cosine",
        ),
    }

    # Descriptions
    descriptions = {
        "130m": "130M parameter model trained with the Sophia-H optimizer (default hyperparameters).",
        "300m": "300M parameter model trained with the Sophia-H optimizer (default hyperparameters).",
        "520m": "520M parameter model trained with the Sophia-H optimizer (default hyperparameters).",
        "1_2b": "1.2B parameter model trained with the Sophia-H optimizer (default hyperparameters).",
    }

    # Names for the runs
    run_names = {
        "130m": "llama_130m_sophia_4096",
        "300m": "llama_300m_sophia_4096",
        "520m": "llama_520m_sophia_4096",
        "1_2b": "llama_1_2b_sophia_4096",
    }

    # Gather config for the requested size
    if size not in param_counts:
        raise ValueError(f"Unknown size: {size}")

    param_count = param_counts[size]
    batch_size = batch_sizes[size]
    model_config = dataclasses.replace(model_cfgs[size], seq_len=4096)
    seq_len = model_config.seq_len
    resource_config = resource_cfgs[size]
    sophia = sophia_configs[size]
    description = descriptions[size]
    run_name = run_names[size]

    num_train_steps = get_num_train_steps(param_count, batch_size, seq_len)

    train = SimpleTrainConfig(
        resource_config,
        train_batch_size=batch_size,
        num_train_steps=num_train_steps,
        learning_rate=sophia.learning_rate,
        optimizer_config=sophia,
    )
    cfg = SpeedrunConfig(
        author=AUTHOR,
        description=description,
        model_config=model_config,
        train_config=train,
    )
    return run_name, cfg


if __name__ == "__main__":
    runs = [
        build_config("130m"),
        build_config("300m"),
        build_config("520m"),
        build_config("1_2b"),
    ]

    steps = []
    for name, cfg in runs:
        cfg.print_run_info()
        steps.extend(default_speedrun(name, cfg))

    executor_main(steps=steps, description="Sophia-H speedruns (Half Chinchilla optimal, default hyperparameters)")
