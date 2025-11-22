# Phase 1 Combined Speedrun

**Author**: redagavin
**Affiliation**: Northeastern University
**URL**: https://redagavin.github.io/

## Description

Phase 1 Combined experiment combining the two most promising techniques from prior research: z-loss regularization (z_loss_weight=1e-4) for numerical stability and extended sequence length (seq_len=4096) for better quality. This is expected to be the best performing experiment in Phase 1.

## Configuration

- **Model**: llama_75m with seq_len=4096 (75M parameters)
- **Hardware**: 1x A100 GPU
- **Training Steps**: 3000
- **Batch Size**: 128
- **Learning Rate**: 3e-3
- **Weight Decay**: 0.1
- **Optimizer**: AdamW (default)
- **Special Techniques**:
  - z_loss_weight=1e-4 (numerical stability)
  - seq_len=4096 (4x longer context)

## Approach

Combines two validated techniques:
1. **Z-loss** (z_loss_weight=1e-4) - Prevents loss spikes and improves training stability
2. **Long context** (seq_len=4096) - Better quality through longer context windows

These techniques are complementary: z-loss provides stability while long context provides quality. Most competitive submissions use both.

## Expected Results

Target BPB: ~1.28-1.32 (best expected in Phase 1)

This should outperform individual techniques:
- Better than baseline (~1.40)
- Better than z-loss alone (~1.31)
- Better than long context alone (~1.38-1.40)
