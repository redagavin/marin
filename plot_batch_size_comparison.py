#!/usr/bin/env python3
"""
Plot batch size experiment results to analyze:
1. Critical batch size (optimal point)
2. Effect of learning rate scaling
3. Effect of warmup ratio
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_batch_size_comparison(
    eval_losses_lr_scaled: list[float],
    eval_losses_lr_scaled_warmup: list[float],
    eval_losses_fixed_lr_warmup: list[float],
    batch_sizes: list[int] = [16, 32, 64, 128, 256],
    save_path: str = "batch_size_comparison.png",
):
    """
    Plot evaluation loss vs batch size for three experimental settings.

    Args:
        eval_losses_lr_scaled: Losses for LR scaled with batch size (no warmup)
        eval_losses_lr_scaled_warmup: Losses for LR scaled + warmup ratio 0.05
        eval_losses_fixed_lr_warmup: Losses for fixed LR=0.02 + warmup ratio 0.05
        batch_sizes: Batch sizes tested (default: [16, 32, 64, 128, 256])
        save_path: Path to save the figure
    """
    # Validate input lengths
    assert len(eval_losses_lr_scaled) == len(batch_sizes), "Mismatch in data length"
    assert len(eval_losses_lr_scaled_warmup) == len(batch_sizes), "Mismatch in data length"
    assert len(eval_losses_fixed_lr_warmup) == len(batch_sizes), "Mismatch in data length"

    # Create figure with larger size for clarity
    plt.figure(figsize=(12, 7))

    # Plot three lines with distinct colors and markers
    plt.plot(
        batch_sizes,
        eval_losses_lr_scaled,
        marker="o",
        linewidth=2.5,
        markersize=8,
        label="LR Scaled (no warmup)",
        color="#2E86AB",  # Blue
    )
    plt.plot(
        batch_sizes,
        eval_losses_lr_scaled_warmup,
        marker="s",
        linewidth=2.5,
        markersize=8,
        label="LR Scaled + Warmup (0.05)",
        color="#A23B72",  # Purple
    )
    plt.plot(
        batch_sizes,
        eval_losses_fixed_lr_warmup,
        marker="^",
        linewidth=2.5,
        markersize=8,
        label="Fixed LR=0.02 + Warmup (0.05)",
        color="#F18F01",  # Orange
    )

    # Find critical batch sizes for summary statistics
    min_idx_1 = np.argmin(eval_losses_lr_scaled)
    min_idx_2 = np.argmin(eval_losses_lr_scaled_warmup)
    min_idx_3 = np.argmin(eval_losses_fixed_lr_warmup)

    # Formatting
    plt.xlabel("Batch Size", fontsize=14, fontweight="bold")
    plt.ylabel("Evaluation Loss", fontsize=14, fontweight="bold")
    plt.title("Effect of Batch Size, LR Scaling, and Warmup on Evaluation Loss", fontsize=16, fontweight="bold", pad=20)

    # Use log scale for x-axis since batch sizes are powers of 2
    plt.xscale("log", base=2)
    plt.xticks(batch_sizes, labels=[str(bs) for bs in batch_sizes])

    # Grid for easier reading
    plt.grid(True, alpha=0.3, linestyle="--", linewidth=0.8)

    # Legend with larger font
    plt.legend(fontsize=12, loc="best", framealpha=0.9)

    # Tight layout to prevent label cutoff
    plt.tight_layout()

    # Save figure
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"✓ Figure saved to {save_path}")

    # Print summary statistics
    print("\n" + "=" * 60)
    print("CRITICAL BATCH SIZES:")
    print("=" * 60)
    print(f"LR Scaled (no warmup):         Batch Size = {batch_sizes[min_idx_1]}, Loss = {eval_losses_lr_scaled[min_idx_1]:.4f}")
    print(
        f"LR Scaled + Warmup:            Batch Size = {batch_sizes[min_idx_2]}, Loss = {eval_losses_lr_scaled_warmup[min_idx_2]:.4f}"
    )
    print(
        f"Fixed LR=0.02 + Warmup:        Batch Size = {batch_sizes[min_idx_3]}, Loss = {eval_losses_fixed_lr_warmup[min_idx_3]:.4f}"
    )

    print("\n" + "=" * 60)
    print("EFFECT ANALYSIS:")
    print("=" * 60)

    # Effect of warmup (compare setting 1 vs 2)
    warmup_effect = np.array(eval_losses_lr_scaled) - np.array(eval_losses_lr_scaled_warmup)
    avg_warmup_effect = np.mean(warmup_effect)
    print(f"\nEffect of Warmup (LR Scaled w/o warmup vs w/ warmup):")
    print(f"  Average improvement: {avg_warmup_effect:.4f}")
    print(f"  {'Warmup helps' if avg_warmup_effect > 0 else 'Warmup hurts'} on average")

    # Effect of LR scaling (compare setting 2 vs 3, both have warmup)
    lr_scaling_effect = np.array(eval_losses_fixed_lr_warmup) - np.array(eval_losses_lr_scaled_warmup)
    avg_lr_scaling_effect = np.mean(lr_scaling_effect)
    print(f"\nEffect of LR Scaling (Fixed LR vs Scaled LR, both with warmup):")
    print(f"  Average improvement from scaling: {avg_lr_scaling_effect:.4f}")
    print(f"  {'LR scaling helps' if avg_lr_scaling_effect > 0 else 'LR scaling hurts'} on average")

    # Overall best configuration
    all_losses = (
        list(eval_losses_lr_scaled) + list(eval_losses_lr_scaled_warmup) + list(eval_losses_fixed_lr_warmup)
    )
    best_loss = min(all_losses)
    if best_loss in eval_losses_lr_scaled:
        best_config = "LR Scaled (no warmup)"
        best_bs = batch_sizes[eval_losses_lr_scaled.index(best_loss)]
    elif best_loss in eval_losses_lr_scaled_warmup:
        best_config = "LR Scaled + Warmup"
        best_bs = batch_sizes[eval_losses_lr_scaled_warmup.index(best_loss)]
    else:
        best_config = "Fixed LR=0.02 + Warmup"
        best_bs = batch_sizes[eval_losses_fixed_lr_warmup.index(best_loss)]

    print(f"\nOverall Best Configuration:")
    print(f"  {best_config} with Batch Size = {best_bs}, Loss = {best_loss:.4f}")
    print("=" * 60 + "\n")

    plt.show()


if __name__ == "__main__":
    # Example usage - replace these with your actual results
    # Format: [loss_bs16, loss_bs32, loss_bs64, loss_bs128, loss_bs256]

    # Setting 1: LR scaled with batch size (no warmup)
    ## replace the last number!!
    eval_losses_lr_scaled = [1.4345, 1.3912, 1.3733, 1.3670, 1.3665]

    # Setting 2: LR scaled + warmup ratio 0.05
    ## replace the first two numbers!!
    eval_losses_lr_scaled_warmup = [1.4132, 1.3870, 1.3709, 1.3639, 1.3635]

    # Setting 3: Fixed LR=0.02 + warmup ratio 0.05
    eval_losses_fixed_lr_warmup = [1.4143, 1.3874, 1.3714, 1.3641, 1.3637]

    # Generate plot
    plot_batch_size_comparison(
        eval_losses_lr_scaled=eval_losses_lr_scaled,
        eval_losses_lr_scaled_warmup=eval_losses_lr_scaled_warmup,
        eval_losses_fixed_lr_warmup=eval_losses_fixed_lr_warmup,
    )
