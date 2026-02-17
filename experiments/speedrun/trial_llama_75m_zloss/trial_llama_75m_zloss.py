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

"""
Trial speedrun code for a 75M parameter model based on the Llama architecture, and the Adamax optimizer.
This is a shorter trial version with num_train_steps=3000 instead of 6000.
"""

import logging
from dataclasses import dataclass

import optax
from levanter.optim import OptimizerConfig

from experiments.llama import llama_75m
from experiments.simple_train_config import SimpleTrainConfig
from marin.execution.executor import executor_main
from marin.resources import GpuConfig
from marin.speedrun.speedrun import Author, SpeedrunConfig, default_speedrun


def get_num_train_steps(param_count, batch_size, seq_len):
    """Compute the number of steps for Chinchilla optimal training (20x params tokens)."""
    total_tokens = param_count * 20
    tokens_per_step = batch_size * seq_len
    return total_tokens // tokens_per_step

# --------------------- Trial speedrun Using Adamax ------------------------

logger = logging.getLogger("ray")
PARAM_COUNT = 75_000_000
BATCH_SIZE = 64
MODEL_CONFIG = llama_75m
SEQ_LEN = MODEL_CONFIG.seq_len
NUM_TRAIN_STEPS = get_num_train_steps(PARAM_COUNT, BATCH_SIZE, SEQ_LEN)
# NUM_TRAIN_STEPS = 7600


speedrun_config = SpeedrunConfig(
    author=Author(
        name="Rice",  # TODO: Update with your name
        affiliation="Rice Factory",  # TODO: Update with your affiliation
        url="N/A",  # TODO: Update with your URL
    ),
    description="Trial: 75M parameter model with z_loss",
    model_config=llama_75m,
    train_config=SimpleTrainConfig(
        GpuConfig(gpu_count=1, accelerator_type="H200"),
        train_batch_size=BATCH_SIZE,
        num_train_steps=NUM_TRAIN_STEPS,  # Trial: 3000 steps instead of 6000
        learning_rate=3e-3,
        weight_decay=0.1,
        steps_per_eval=1000,  # Adjusted: evaluate every 1000 steps for 3000 total steps
        z_loss_weight=1e-4,
    ),
)

speedrun_config.print_run_info()

if __name__ == "__main__":
    executor_main(steps=default_speedrun("trial_llama_75m_zloss_int8", speedrun_config))
