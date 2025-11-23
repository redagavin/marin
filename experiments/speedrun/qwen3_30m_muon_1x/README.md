# qwen3_30m_muon_1x

## Overview
Phase 1 speedrun experiment: 30M Qwen3 model trained with Muon optimizer at 1× Chinchilla-optimal data scale.

## Approach
This experiment explores **Qwen3 architecture advantages at 30M scale** with QK normalization:
- **Model**: 30M parameter Qwen3 architecture (with QK normalization)
- **Optimizer**: Muon with tuned hyperparameters
- **Data Scale**: 1× Chinchilla-optimal (600M tokens)
- **Hardware**: 1× H200 GPU

## Key Hyperparameters
- **Learning Rate**: 0.024 (Muon learning rate)
- **Adam LR**: 0.0048
- **Momentum**: 0.95
- **Batch Size**: 128
- **Sequence Length**: 1024
- **Total Steps**: 4,577
- **Evaluation Frequency**: Every 500 steps

## Expected Performance
- **Target BPB**: 1.46-1.50 BPB (better than Llama baseline)
- **Training Time**: ~15 minutes on H200
- **Total Tokens**: 600M

## Architecture Differences
Qwen3 differs from Llama with:
- **QK Normalization**: Normalizes query/key projections for improved attention stability
- **Improved Gradient Flow**: Better numerical stability during training
- Benefits expected to yield ~0.02 BPB improvement over Llama at same scale

## Comparison Rationale
This experiment tests whether architectural improvements (QK norm) provide tangible benefits at the 30M scale, validating the advantages of the Qwen3 design.
