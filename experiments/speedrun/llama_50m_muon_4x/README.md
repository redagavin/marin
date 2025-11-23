# 50M Llama with Muon Optimizer (4× Chinchilla)

**Model:** 50M parameter Llama
**Optimizer:** Muon
**Data Scale:** 4× Chinchilla-optimal (4B tokens)
**Expected BPB:** 1.32-1.36
**Target:** Beat current 50M record (1.353 BPB)

## Rationale

This is our highest-priority Phase 1 experiment. Muon optimizer has proven to win at all scales (130M-1.2B), and the paper shows Muon's sweet spot is 1×-4× Chinchilla data. We're targeting the current 50M record of 1.353 BPB achieved with 10× Chinchilla and Adam optimizer.

## Hyperparameters

- **Learning rate:** 0.020 (scaled between 30M and 130M)
- **Adam LR:** 0.004
- **Momentum:** 0.95
- **Batch size:** 128
- **Training steps:** 30,518 (4B tokens / 128 batch / 1024 seq_len)
- **Warmup:** 0 (Muon characteristic)
- **LR schedule:** Linear decay to 0
- **Weight decay:** 0.1

## Running

```bash
sbatch submit_slurm.sh
```

## Results

After completion, results will be in:
- Checkpoint: `/scratch/yang.zih/marin_speedrun/output/checkpoints/speedrun/llama_50m_muon_4x-*`
- Results JSON: `speedrun_results.json`
- WandB: `https://wandb.ai/marin-speedrun/marin-speedrun/runs/llama_50m_muon_4x-*`
