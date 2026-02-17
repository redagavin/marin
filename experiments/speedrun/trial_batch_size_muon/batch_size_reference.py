#!/usr/bin/env python3
"""
Reference table for batch size ablation study hyperparameters.
Shows the calculated learning rates, steps, and tokens for each batch size.
"""

import math

PARAM_COUNT = 50_000_000
SEQ_LEN = 1024
BASE_BATCH_SIZE = 128
BASE_LR = 0.020
TOTAL_TOKENS = PARAM_COUNT * 40  # 2x Chinchilla optimal


def calculate_hp(batch_size):
    """Calculate hyperparameters for a given batch size."""
    tokens_per_step = batch_size * SEQ_LEN
    num_steps = TOTAL_TOKENS // tokens_per_step
    learning_rate = BASE_LR * math.sqrt(batch_size / BASE_BATCH_SIZE)
    return {
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "num_steps": num_steps,
        "tokens_per_step": tokens_per_step,
    }


if __name__ == "__main__":
    # Note: Batch size 512 may OOM on a single H200 (141GB).
    # Tested feasible range: 1-256. Max depends on model size, optimizer, and memory overhead.
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]

    print("\n" + "="*100)
    print("Batch Size Ablation Study - Hyperparameter Reference (50M Model, 2x Chinchilla)")
    print("="*100)
    print(
        f"{'Batch Size':<12} {'Learning Rate':<16} {'Training Steps':<20} {'Tokens/Step':<15} {'Total Tokens':<15}"
    )
    print("-" * 100)

    for batch_size in batch_sizes:
        hp = calculate_hp(batch_size)
        print(
            f"{hp['batch_size']:<12} {hp['learning_rate']:<16.6f} {hp['num_steps']:<20,} {hp['tokens_per_step']:<15,} {TOTAL_TOKENS:<15,}"
        )

    print("-" * 100)
    print(f"\nBase configuration: batch_size={BASE_BATCH_SIZE}, learning_rate={BASE_LR}")
    print(f"Total tokens: {TOTAL_TOKENS:,} (2x Chinchilla optimal = 40x params)")
    print(f"Sequence length: {SEQ_LEN}")
    print(f"Model parameters: {PARAM_COUNT:,}")
    print("\n")
