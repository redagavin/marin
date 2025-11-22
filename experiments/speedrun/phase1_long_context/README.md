# Phase 1 Long Context Speedrun

**Author**: redagavin
**Affiliation**: Northeastern University
**URL**: https://redagavin.github.io/

## Description

Phase 1 Long Context experiment using 75M Llama model with extended sequence length (seq_len=4096 vs standard 1024). Most competitive submissions use longer sequence lengths for better quality.

## Configuration

- **Model**: llama_75m with seq_len=4096 (75M parameters)
- **Hardware**: 1x A100 GPU
- **Training Steps**: 3000
- **Batch Size**: 128 (same batch size, more compute per step)
- **Learning Rate**: 3e-3
- **Weight Decay**: 0.1
- **Optimizer**: AdamW (default)
- **Special Technique**: Extended sequence length (4096 vs 1024)

## Approach

Standard AdamW training with 4x longer sequence length. This requires more computation per step but generally achieves better BPB scores due to longer context windows. Most competitive submissions on the leaderboard use seq_len=4096.

## Expected Results

Target BPB: ~1.38-1.40 (similar to baseline, but tests if longer context helps on A100 with this training budget)
