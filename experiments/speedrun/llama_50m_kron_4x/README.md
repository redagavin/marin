# 50M Llama with Kron Optimizer (4× Chinchilla)

**Model:** 50M parameter Llama
**Optimizer:** Kron (Kronecker-factored preconditioner)
**Data Scale:** 4× Chinchilla-optimal (4B tokens)
**Expected BPB:** 1.34-1.38

## Rationale

First Kron experiment at any scale in the Marin speedrun benchmark. Kronecker-factored optimization methods like Kron have shown promise in second-order optimization. This experiment tests whether Kron can deliver similar or better performance than Muon at the 4× data scale, while maintaining reasonable memory and computation costs.

## Hyperparameters

- **Learning rate:** 0.010
- **Batch size:** 128
- **Training steps:** 30,518 (4B tokens / 128 batch / 1024 seq_len)
- **Optimizer:** Kron with beta1=0.9, beta2=0.98
- **Preconditioner:** Updated with 5% probability, warmup at 2000 steps
- **Evaluation:** Every 2,000 steps

## What is Kron?

Kron (Kronecker) is a second-order optimizer that approximates the Hessian using Kronecker factorization. This reduces memory overhead compared to full second-order methods while capturing curvature information better than first-order methods.

## Running

```bash
sbatch submit_slurm.sh
```

## Expected Results

If Kron successfully leverages second-order information, we expect BPB in the 1.34-1.38 range, potentially matching or beating Muon's 4× performance.
