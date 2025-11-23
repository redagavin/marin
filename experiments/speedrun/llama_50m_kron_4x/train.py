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
Phase 1: 50M Llama with Kron optimizer at 4× Chinchilla-optimal data.

First Kron experiment at any scale in Marin speedrun.
Expected: 1.34-1.38 BPB
"""

from levanter.optim import KronConfig

from experiments.llama import llama_50m
from experiments.simple_train_config import SimpleTrainConfig
from marin.execution.executor import executor_main
from marin.resources import GpuConfig
from marin.speedrun.speedrun import Author, SpeedrunConfig, default_speedrun


kron_config = KronConfig(
    learning_rate=0.010,
    beta1=0.9,
    beta2=0.98,
    weight_decay=0.1,
    warmup=2000,
    max_grad_norm=1,
    preconditioner_update_probability=0.05,
    preconditioner_lr=0.1,
    update_prob_flat_start=1000,
)

# Calculate steps for 4× Chinchilla (4B tokens)
# 50M params × 4 × 20 = 4B tokens
# 4B tokens / (128 batch × 1024 seq_len) = 30,518 steps
num_train_steps = 30518

speedrun_config = SpeedrunConfig(
    author=Author(
        name="redagavin",
        affiliation="Northeastern University",
        url="https://redagavin.github.io/"
    ),
    description="Phase 1: 50M Llama with Kron optimizer at 4× Chinchilla (4B tokens). First Kron experiment anywhere.",
    model_config=llama_50m,
    train_config=SimpleTrainConfig(
        GpuConfig(gpu_count=1, accelerator_type="H200"),
        train_batch_size=128,
        num_train_steps=num_train_steps,
        learning_rate=kron_config.learning_rate,
        optimizer_config=kron_config,
        weight_decay=kron_config.weight_decay,
        steps_per_eval=2000,
    ),
)

speedrun_config.print_run_info()

if __name__ == "__main__":
    executor_main(steps=default_speedrun("llama_50m_kron_4x", speedrun_config))
