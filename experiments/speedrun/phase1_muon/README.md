# Phase 1 Muon Speedrun

**Author**: redagavin
**Affiliation**: Northeastern University
**URL**: https://redagavin.github.io/

## Description

Phase 1 Muon experiment using 75M Llama model with Muon optimizer. Muon is a momentum-based optimizer with strong empirical performance across model scales.

## Configuration

- **Model**: llama_75m (75M parameters)
- **Hardware**: 1x A100 GPU
- **Training Steps**: 3000
- **Batch Size**: 128
- **Main Learning Rate**: 0.016
- **Adam Learning Rate**: 0.0032
- **Weight Decay**: 0.1
- **Optimizer**: Muon with linear LR schedule
- **Special Techniques**:
  - Dual learning rates (main + Adam)
  - Linear LR schedule with decay=0.8
  - No warmup (warmup=0)
  - High momentum (0.95)

## Approach

Uses Muon optimizer which is a momentum-based approach with dual learning rates. Configuration adapted from the 130M Muon reference submission. Muon has shown strong scaling characteristics across model sizes (130M-1.2B).

## Expected Results

Target BPB: ~1.32-1.36
