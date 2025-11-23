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
Phase 1: 30M Llama with Kron optimizer at 1× Chinchilla-optimal data.

Tests Kron optimizer effectiveness on smaller models.
Expected: 1.48-1.52 BPB
"""

from levanter.optim import KronConfig

from experiments.llama import llama_30m
from experiments.simple_train_config import SimpleTrainConfig
from marin.execution.executor import executor_main
from marin.resources import GpuConfig
from marin.speedrun.speedrun import Author, SpeedrunConfig, default_speedrun


kron_config = KronConfig(
    learning_rate=0.012,
    beta1=0.9,
    weight_decay=0.1,
    warmup=1000,
    max_grad_norm=1,
    preconditioner_update_probability=0.05,
    preconditioner_lr=0.1,
    update_prob_flat_start=500,
)

# Calculate steps for 1× Chinchilla (600M tokens)
# 30M params × 1 × 20 = 600M tokens
# 600M tokens / (128 batch × 1024 seq_len) = 4,577 steps
num_train_steps = 4577

speedrun_config = SpeedrunConfig(
    author=Author(
        name="redagavin",
        affiliation="Northeastern University",
        url="https://redagavin.github.io/"
    ),
    description="Phase 1: 30M Llama with Kron optimizer at 1× Chinchilla (600M tokens). Tests Kron on smaller models.",
    model_config=llama_30m,
    train_config=SimpleTrainConfig(
        GpuConfig(gpu_count=1, accelerator_type="H200"),
        train_batch_size=128,
        num_train_steps=num_train_steps,
        learning_rate=kron_config.learning_rate,
        optimizer_config=kron_config,
        weight_decay=kron_config.weight_decay,
        steps_per_eval=500,
    ),
)

speedrun_config.print_run_info()

if __name__ == "__main__":
    executor_main(steps=default_speedrun("llama_30m_kron_1x", speedrun_config))
