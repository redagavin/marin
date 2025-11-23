# llama_30m_muon_1x

## Overview
Phase 1 speedrun experiment: 30M Llama model trained with Muon optimizer at 1× Chinchilla-optimal data scale.

## Approach
This experiment serves as a **baseline validation** to test if Muon's benefits extend to standard scaling at 30M:
- **Model**: 30M parameter Llama architecture
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
- **Target BPB**: 1.48-1.52 BPB
- **Training Time**: ~15 minutes on H200
- **Total Tokens**: 600M

## Rationale
Tests whether Muon maintains its performance advantage at standard Chinchilla scaling ratio, providing a baseline for comparison with the 4× variant.
