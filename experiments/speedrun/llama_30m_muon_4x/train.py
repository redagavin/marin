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
Phase 1: 30M Llama with Muon optimizer at 4× Chinchilla-optimal data.

Expected best 30M result: 1.42-1.47 BPB
"""

from levanter.optim import MuonConfig

from experiments.llama import llama_30m
from experiments.simple_train_config import SimpleTrainConfig
from marin.execution.executor import executor_main
from marin.resources import GpuConfig
from marin.speedrun.speedrun import Author, SpeedrunConfig, default_speedrun


muon_config = MuonConfig(
    learning_rate=0.024,
    adam_lr=0.0048,
    momentum=0.95,
    beta1=0.8,
    beta2=0.98,
    epsilon=1e-15,
    muon_epsilon=1e-5,
    max_grad_norm=1,
    warmup=0,
    min_lr_ratio=0,
    lr_schedule="linear",
    decay=0.8,
)

# Calculate steps for 4× Chinchilla (2.4B tokens)
# 30M params × 4 × 20 = 2.4B tokens
# 2.4B tokens / (128 batch × 1024 seq_len) = 18,311 steps
num_train_steps = 18311

speedrun_config = SpeedrunConfig(
    author=Author(
        name="redagavin",
        affiliation="Northeastern University",
        url="https://redagavin.github.io/"
    ),
    description="Phase 1: 30M Llama with Muon at 4× Chinchilla (2.4B tokens). Best 30M expected.",
    model_config=llama_30m,
    train_config=SimpleTrainConfig(
        GpuConfig(gpu_count=1, accelerator_type="H200"),
        train_batch_size=128,
        num_train_steps=num_train_steps,
        learning_rate=muon_config.learning_rate,
        optimizer_config=muon_config,
        steps_per_eval=1500,
    ),
)

speedrun_config.print_run_info()

if __name__ == "__main__":
    executor_main(steps=default_speedrun("llama_30m_muon_4x", speedrun_config))
