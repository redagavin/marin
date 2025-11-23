# 50M Qwen3 with Muon Optimizer (4× Chinchilla)

**Model:** 50M parameter Qwen3
**Optimizer:** Muon
**Data Scale:** 4× Chinchilla-optimal (4B tokens)
**Expected BPB:** 1.30-1.34
**Goal:** Best overall Phase 1 result

## Rationale

Combines two proven winners:
1. **Qwen3 architecture:** Beats Llama at 130M+ scale (1.166 vs 1.170 BPB)
2. **Muon optimizer:** Wins across all scales tested (130M-1.2B)
3. **4× Chinchilla:** Muon's optimal data range per paper

This is our best shot at achieving the lowest BPB in Phase 1.

## Architecture Differences from Llama

Qwen3 adds QK normalization which improves training stability and final performance at larger scales. Testing if benefits transfer to 50M.

## Hyperparameters

Same as llama_50m_muon_4x - only architecture changes.

## Running

```bash
sbatch submit_slurm.sh
```
