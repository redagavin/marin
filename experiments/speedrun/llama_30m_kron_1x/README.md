# llama_30m_kron_1x

## Overview
Phase 1 speedrun experiment: 30M Llama model trained with Kron optimizer at 1× Chinchilla-optimal data scale.

## Approach
This experiment explores the effectiveness of **Kron optimizer on smaller 30M models**:
- **Model**: 30M parameter Llama architecture
- **Optimizer**: Kron with tuned preconditioner settings
- **Data Scale**: 1× Chinchilla-optimal (600M tokens)
- **Hardware**: 1× H200 GPU

## Key Hyperparameters
- **Learning Rate**: 0.012
- **Beta1**: 0.9
- **Beta2**: 0.98
- **Weight Decay**: 0.1
- **Warmup Steps**: 1000
- **Batch Size**: 128
- **Sequence Length**: 1024
- **Total Steps**: 4,577
- **Preconditioner Update Probability**: 0.05
- **Evaluation Frequency**: Every 500 steps

## Expected Performance
- **Target BPB**: 1.48-1.52 BPB
- **Training Time**: ~15 minutes on H200
- **Total Tokens**: 600M

## Comparison Rationale
Kron is known to be more memory-efficient than some alternatives, making it a good candidate for smaller models. This experiment tests its effectiveness at 30M scale compared to Muon.
