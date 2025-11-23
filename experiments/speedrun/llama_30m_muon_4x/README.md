# llama_30m_muon_4x

## Overview
Phase 1 speedrun experiment: 30M Llama model trained with Muon optimizer at 4× Chinchilla-optimal data scale.

## Approach
This experiment targets the **best possible 30M model result** by combining:
- **Model**: 30M parameter Llama architecture
- **Optimizer**: Muon with optimized hyperparameters
- **Data Scale**: 4× Chinchilla-optimal (2.4B tokens)
- **Hardware**: 1× H200 GPU

## Key Hyperparameters
- **Learning Rate**: 0.024 (Muon learning rate)
- **Adam LR**: 0.0048
- **Momentum**: 0.95
- **Batch Size**: 128
- **Sequence Length**: 1024
- **Total Steps**: 18,311
- **Evaluation Frequency**: Every 1500 steps

## Expected Performance
- **Target BPB**: 1.42-1.47 BPB
- **Training Time**: ~45 minutes on H200
- **Total Tokens**: 2.4B

## Implementation Notes
- Uses Muon optimizer which has shown superior performance on larger data scales
- Slightly higher learning rate compared to 50M baseline (0.024 vs 0.020) adjusted for model size
- Evaluation frequency set to 1500 steps to balance monitoring with training overhead
