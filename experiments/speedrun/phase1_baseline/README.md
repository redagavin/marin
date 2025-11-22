# Phase 1 Baseline Speedrun

**Author**: redagavin
**Affiliation**: Northeastern University
**URL**: https://redagavin.github.io/

## Description

Phase 1 Baseline experiment using 75M Llama model with standard AdamW optimizer. This establishes the performance baseline for all Phase 1 optimization experiments.

## Configuration

- **Model**: llama_75m (75M parameters)
- **Hardware**: 1x A100 GPU
- **Training Steps**: 3000
- **Batch Size**: 128
- **Learning Rate**: 3e-3
- **Weight Decay**: 0.1
- **Optimizer**: AdamW (default)

## Approach

Standard training configuration with no special optimizations. This serves as the reference point for evaluating the effectiveness of various optimization techniques (z-loss, AdamH, Muon, etc.) tested in other Phase 1 experiments.

## Expected Results

Target BPB: ~1.40
