# Phase 1 Test Run

**Author**: redagavin
**Affiliation**: Northeastern University
**URL**: https://redagavin.github.io/

## Description

Minimal test run with only 10 training steps to verify the pipeline works correctly before launching full experiments.

## Configuration

- **Model**: llama_75m (75M parameters)
- **Hardware**: 1x H200 GPU
- **Training Steps**: 10 (minimal for testing)
- **Batch Size**: 128
- **Learning Rate**: 3e-3
- **Weight Decay**: 0.1
- **Optimizer**: AdamW (default)

## Usage

Run with srun for interactive testing:

```bash
cd /scratch/yang.zih/marin_speedrun/marin/experiments/speedrun/phase1_test
srun --partition=gpu --gres=gpu:h200:1 --mem=32G --cpus-per-task=8 --time=00:30:00 ./test_srun.sh
```

This should complete in ~2-3 minutes.

## Purpose

This test verifies:
- Environment setup works correctly
- JAX/GPU initialization succeeds
- Training pipeline runs without errors
- Data loading is functional
- Checkpointing works

If this test passes, the full experiments should work correctly.
