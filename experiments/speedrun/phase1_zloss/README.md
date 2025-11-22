# Phase 1 Z-Loss Speedrun

**Author**: redagavin
**Affiliation**: Northeastern University
**URL**: https://redagavin.github.io/

## Description

Phase 1 Z-Loss experiment using 75M Llama model with z-loss regularization for improved numerical stability and training efficiency.

## Configuration

- **Model**: llama_75m (75M parameters)
- **Hardware**: 1x A100 GPU
- **Training Steps**: 3000
- **Batch Size**: 128
- **Learning Rate**: 3e-3
- **Weight Decay**: 0.1
- **Optimizer**: AdamW (default)
- **Special Technique**: z_loss_weight=1e-4

## Approach

Standard training with z-loss regularization. Z-loss improves numerical stability during training and has shown significant efficiency gains. Reference submission achieved BPB=1.31 with this technique on TPU.

## Expected Results

Target BPB: ~1.31 (based on reference submission)
