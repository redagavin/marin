#!/usr/bin/env python3
"""
Analyze differences between experimental settings without multiple runs.
Provides various metrics to help make defensible claims about effect sizes.
"""

import numpy as np


def analyze_differences(
    eval_losses_lr_scaled: list[float],
    eval_losses_lr_scaled_warmup: list[float],
    eval_losses_fixed_lr_warmup: list[float],
    batch_sizes: list[int] = [16, 32, 64, 128, 256],
):
    """
    Analyze differences between settings and provide metrics for comparison.

    Without multiple runs, we can't claim statistical significance, but we can:
    1. Report absolute and relative differences
    2. Compare effect sizes across different interventions
    3. Compare to typical loss ranges and magnitudes
    """

    print("=" * 80)
    print("EFFECT SIZE ANALYSIS (Single Run - No Statistical Significance Claims)")
    print("=" * 80)

    # Convert to numpy arrays for easier computation
    lr_scaled = np.array(eval_losses_lr_scaled)
    lr_scaled_warmup = np.array(eval_losses_lr_scaled_warmup)
    fixed_lr_warmup = np.array(eval_losses_fixed_lr_warmup)

    print("\n" + "─" * 80)
    print("1. PAIRWISE DIFFERENCES: LR Scaled+Warmup vs Fixed LR+Warmup")
    print("─" * 80)
    print("(Both use warmup=0.05, comparing effect of LR scaling)\n")

    diff_scaling_effect = fixed_lr_warmup - lr_scaled_warmup

    print(f"{'Batch Size':<12} {'LR Scaled+W':<14} {'Fixed LR+W':<14} {'Difference':<14} {'% Difference':<14}")
    print("─" * 80)
    for i, bs in enumerate(batch_sizes):
        pct_diff = (diff_scaling_effect[i] / fixed_lr_warmup[i]) * 100
        sign = "+" if diff_scaling_effect[i] < 0 else ""
        print(f"{bs:<12} {lr_scaled_warmup[i]:<14.4f} {fixed_lr_warmup[i]:<14.4f} "
              f"{sign}{diff_scaling_effect[i]:<13.4f} {sign}{pct_diff:<13.3f}%")

    mean_abs_diff = np.mean(np.abs(diff_scaling_effect))
    mean_pct_diff = np.mean(np.abs(diff_scaling_effect / fixed_lr_warmup)) * 100
    max_abs_diff = np.max(np.abs(diff_scaling_effect))

    print("\n" + "Summary Statistics:")
    print(f"  Mean absolute difference:    {mean_abs_diff:.4f}")
    print(f"  Mean percentage difference:  {mean_pct_diff:.3f}%")
    print(f"  Max absolute difference:     {max_abs_diff:.4f}")
    print(f"  Range of differences:        [{diff_scaling_effect.min():.4f}, {diff_scaling_effect.max():.4f}]")

    print("\n" + "─" * 80)
    print("2. EFFECT SIZE COMPARISON: Warmup vs LR Scaling")
    print("─" * 80)
    print("(Comparing magnitude of different interventions)\n")

    # Effect of warmup (LR scaled without vs with warmup)
    warmup_effect = lr_scaled - lr_scaled_warmup

    # Effect of LR scaling (fixed LR vs scaled LR, both with warmup)
    lr_scaling_effect = fixed_lr_warmup - lr_scaled_warmup

    print(f"{'Batch Size':<12} {'Warmup Effect':<16} {'LR Scaling Effect':<20} {'Ratio':<12}")
    print("─" * 80)
    for i, bs in enumerate(batch_sizes):
        ratio = warmup_effect[i] / lr_scaling_effect[i] if lr_scaling_effect[i] != 0 else float('inf')
        print(f"{bs:<12} {warmup_effect[i]:<16.4f} {lr_scaling_effect[i]:<20.4f} {ratio:<12.2f}x")

    avg_warmup_effect = np.mean(warmup_effect)
    avg_lr_scaling_effect = np.mean(lr_scaling_effect)

    print("\n" + "Summary:")
    print(f"  Average warmup effect:       {avg_warmup_effect:.4f} (improvement from adding warmup)")
    print(f"  Average LR scaling effect:   {avg_lr_scaling_effect:.4f} (improvement from scaling LR)")
    print(f"  Effect ratio:                {avg_warmup_effect/avg_lr_scaling_effect:.2f}x")
    print(f"\n  → Warmup effect is {abs(avg_warmup_effect/avg_lr_scaling_effect):.1f}x {'larger' if abs(avg_warmup_effect) > abs(avg_lr_scaling_effect) else 'smaller'} than LR scaling effect")

    print("\n" + "─" * 80)
    print("3. PRACTICAL SIGNIFICANCE")
    print("─" * 80)

    # Compare to total loss range
    all_losses = np.concatenate([lr_scaled, lr_scaled_warmup, fixed_lr_warmup])
    loss_range = all_losses.max() - all_losses.min()

    print(f"\nTotal loss range across all experiments: {loss_range:.4f}")
    print(f"  (from {all_losses.min():.4f} to {all_losses.max():.4f})")

    print(f"\nLR Scaling vs Fixed LR (both with warmup):")
    print(f"  Mean difference: {mean_abs_diff:.4f}")
    print(f"  As % of total range: {(mean_abs_diff/loss_range)*100:.2f}%")

    # Compare to improvement from batch size scaling
    best_loss_each_setting = [lr_scaled.min(), lr_scaled_warmup.min(), fixed_lr_warmup.min()]
    worst_loss_each_setting = [lr_scaled.max(), lr_scaled_warmup.max(), fixed_lr_warmup.max()]
    avg_batch_size_effect = np.mean(np.array(worst_loss_each_setting) - np.array(best_loss_each_setting))

    print(f"\nFor context - Effect of batch size (BS 16 → BS 256):")
    print(f"  Average improvement: {avg_batch_size_effect:.4f}")
    print(f"  LR scaling effect is {(mean_abs_diff/avg_batch_size_effect)*100:.1f}% of batch size effect")

    print("\n" + "─" * 80)
    print("4. SUGGESTED CLAIMS (Without Statistical Significance)")
    print("─" * 80)

    print("\nOption 1 - Conservative (Recommended for single runs):")
    print('  "LR scaling and fixed LR with warmup show similar performance')
    print(f'   (mean difference of {mean_abs_diff:.4f} or {mean_pct_diff:.2f}%), suggesting that')
    print('   with warmup, the learning rate schedule has minimal impact on final')
    print('   evaluation loss. However, these results are from single runs and')
    print('   statistical significance cannot be established."')

    if mean_abs_diff < 0.001:
        strength = "negligible"
        confidence = "very high"
    elif mean_abs_diff < 0.005:
        strength = "minimal"
        confidence = "high"
    elif mean_abs_diff < 0.01:
        strength = "small"
        confidence = "moderate"
    else:
        strength = "noticeable"
        confidence = "moderate"

    print(f"\nOption 2 - Effect size based:")
    print(f'  "The difference between LR scaling and fixed LR (both with warmup)')
    print(f'   is {strength} (mean: {mean_abs_diff:.4f}, {mean_pct_diff:.2f}%), representing')
    print(f'   only {(mean_abs_diff/avg_batch_size_effect)*100:.1f}% of the improvement gained from')
    print(f'   optimal batch size selection. This suggests {confidence} confidence that')
    print('   LR scaling provides limited additional benefit when warmup is used."')

    if avg_warmup_effect > 10 * abs(avg_lr_scaling_effect):
        print(f"\nOption 3 - Comparative:")
        print(f'  "Warmup provides a substantially larger benefit')
        print(f'   ({avg_warmup_effect:.4f}) than LR scaling ({abs(avg_lr_scaling_effect):.4f}),')
        print(f'   being {avg_warmup_effect/abs(avg_lr_scaling_effect):.1f}x more effective.')
        print('   The difference between scaled and fixed LR (both with warmup)')
        print(f'   is minimal ({mean_abs_diff:.4f}), suggesting warmup may subsume')
        print('   the benefits of LR scaling."')

    print("\n" + "─" * 80)
    print("5. RECOMMENDATIONS FOR HONEST REPORTING")
    print("─" * 80)

    print("\n✓ DO:")
    print("  • Report exact numbers and let readers judge")
    print("  • Compare effect sizes (warmup vs LR scaling vs batch size)")
    print("  • Use terms like 'minimal difference', 'similar performance'")
    print("  • Be transparent that these are single runs")
    print("  • Focus on practical significance (cost/benefit)")

    print("\n✗ DON'T:")
    print("  • Claim 'no significant difference' without statistical tests")
    print("  • Claim 'equivalence' without proper testing")
    print("  • Ignore the differences entirely")
    print("  • Overstate confidence in conclusions")

    print("\n" + "=" * 80)

    return {
        'mean_abs_diff': mean_abs_diff,
        'mean_pct_diff': mean_pct_diff,
        'warmup_effect': avg_warmup_effect,
        'lr_scaling_effect': avg_lr_scaling_effect,
        'effect_ratio': avg_warmup_effect / avg_lr_scaling_effect,
    }


if __name__ == "__main__":
    # Your actual data

    # Setting 1: LR scaled with batch size (no warmup)
    ## replace the last number!!
    eval_losses_lr_scaled = [1.4345, 1.3912, 1.3733, 1.3670, 1.3665]

    # Setting 2: LR scaled + warmup ratio 0.05
    ## replace the first two numbers!!
    eval_losses_lr_scaled_warmup = [1.4132, 1.3870, 1.3709, 1.3639, 1.3635]

    # Setting 3: Fixed LR=0.02 + warmup ratio 0.05
    eval_losses_fixed_lr_warmup = [1.4143, 1.3874, 1.3714, 1.3641, 1.3637]

    results = analyze_differences(
        eval_losses_lr_scaled=eval_losses_lr_scaled,
        eval_losses_lr_scaled_warmup=eval_losses_lr_scaled_warmup,
        eval_losses_fixed_lr_warmup=eval_losses_fixed_lr_warmup,
    )
