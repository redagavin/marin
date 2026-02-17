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

"""Speedrun with 130M Llama model using AdamW optimizer (Chinchilla optimal steps)."""

import dataclasses
import logging

from levanter.optim import AdamConfig

from experiments.llama import llama_150m
from experiments.simple_train_config import SimpleTrainConfig
from marin.execution.executor import executor_main
from marin.resources import GpuConfig
from marin.speedrun.speedrun import Author, SpeedrunConfig, default_speedrun

logger = logging.getLogger("ray")

AUTHOR = Author(name="rice", affiliation="neu", url="")


def get_num_train_steps(param_count, batch_size, seq_len):
    """Compute the number of steps for Chinchilla optimal training (20x params tokens)."""
    total_tokens = param_count * 20
    tokens_per_step = batch_size * seq_len
    return total_tokens // tokens_per_step


# Configuration for 130M parameter model
PARAM_COUNT = 130_000_000
BATCH_SIZE = 128
MODEL_CONFIG = dataclasses.replace(llama_150m, seq_len=4096)
SEQ_LEN = MODEL_CONFIG.seq_len
RESOURCE_CONFIG = GpuConfig(gpu_count=1, accelerator_type="H200")

# AdamW optimizer config for 130m (from adamw_sweep.py)
ADAM_CONFIG = AdamConfig(
    learning_rate=0.008,
    weight_decay=0.1,
    min_lr_ratio=0,
    warmup=2000,
    beta1=0.9,
    beta2=0.98,
    epsilon=1e-20,
    max_grad_norm=1,
    nesterov=False,
)

NUM_TRAIN_STEPS = get_num_train_steps(PARAM_COUNT, BATCH_SIZE, SEQ_LEN)
# NUM_TRAIN_STEPS = NUM_TRAIN_STEPS // 2

speedrun_config = SpeedrunConfig(
    author=AUTHOR,
    description="130M parameter model trained with the AdamW optimizer.",
    model_config=MODEL_CONFIG,
    train_config=SimpleTrainConfig(
        RESOURCE_CONFIG,
        train_batch_size=BATCH_SIZE,
        num_train_steps=NUM_TRAIN_STEPS,
        learning_rate=ADAM_CONFIG.learning_rate,
        optimizer_config=ADAM_CONFIG,
    ),
)

# Shows your speedrun configuration, model FLOPs, model size and (training) hardware FLOPs
speedrun_config.print_run_info()

if __name__ == "__main__":
    executor_main(steps=default_speedrun("llama_130m_adamw_4096", speedrun_config))
