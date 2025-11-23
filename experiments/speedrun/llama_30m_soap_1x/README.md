# llama_30m_soap_1x

## Overview
Phase 1 speedrun experiment: 30M Llama model trained with SOAP optimizer at 1× Chinchilla-optimal data scale.

## Approach
This experiment explores the effectiveness of **SOAP (Second-Order Shampoo) optimizer on smaller 30M models**:
- **Model**: 30M parameter Llama architecture
- **Optimizer**: SOAP with Shampoo-based preconditioning
- **Data Scale**: 1× Chinchilla-optimal (600M tokens)
- **Hardware**: 1× H200 GPU

## Key Hyperparameters
- **Learning Rate**: 0.012
- **Beta1**: 0.95
- **Beta2**: 0.95
- **Shampoo Beta**: 0.95
- **Weight Decay**: 0.1
- **Warmup Steps**: 1000
- **Batch Size**: 128
- **Sequence Length**: 1024
- **Total Steps**: 4,577
- **Precondition Frequency**: 10
- **Max Precondition Dimension**: 10000
- **Evaluation Frequency**: Every 500 steps

## Expected Performance
- **Target BPB**: 1.48-1.52 BPB
- **Training Time**: ~15 minutes on H200
- **Total Tokens**: 600M

## Comparison Rationale
SOAP combines Shampoo preconditioning with momentum for improved convergence. This experiment tests its effectiveness at 30M scale and provides a comparison point with Kron and Muon optimizers.
