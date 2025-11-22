# Phase 1 AdamH Speedrun

**Author**: redagavin
**Affiliation**: Northeastern University
**URL**: https://redagavin.github.io/

## Description

Phase 1 AdamH experiment using 75M Llama model with AdamH optimizer and hybrid normalization. AdamH maintains constant norm with dual learning rates, combined with architectural improvements for scale invariance.

## Configuration

- **Model**: llama_75m with hybrid_norm=True, use_qk_norm=True (75M parameters)
- **Hardware**: 1x A100 GPU
- **Training Steps**: 3000
- **Batch Size**: 128
- **Main Learning Rate**: 0.02
- **Adam Learning Rate**: 0.008
- **Weight Decay**: 0.1
- **Optimizer**: AdamH with cosine schedule
- **Special Techniques**:
  - hybrid_norm for scale invariance
  - use_qk_norm for attention stability
  - Dual learning rates (main + Adam)

## Approach

Uses AdamH optimizer which maintains constant parameter norm throughout training. The hybrid normalization and query-key normalization architectural changes work synergistically with AdamH. Reference 300M submission achieved BPB=1.064 with this approach.

## Expected Results

Target BPB: ~1.35-1.38
