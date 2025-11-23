# 50M Llama with SOAP Optimizer (4× Chinchilla)

**Model:** 50M parameter Llama
**Optimizer:** SOAP (Shampoo-based Optimizer)
**Data Scale:** 4× Chinchilla-optimal (4B tokens)
**Expected BPB:** 1.34-1.38

## Rationale

First SOAP experiment at any scale in the Marin speedrun benchmark. SOAP combines ideas from Shampoo (a Hessian-based method) with modern optimization to provide a balance between second-order curvature information and computational efficiency. This experiment evaluates whether SOAP can achieve comparable or superior performance to Muon at 4× scale.

## Hyperparameters

- **Learning rate:** 0.010
- **Batch size:** 128
- **Training steps:** 30,518 (4B tokens / 128 batch / 1024 seq_len)
- **Optimizer:** SOAP with beta1=0.95, beta2=0.95
- **Preconditioner:** Frequency=10, max_dim=10000, merge_small_dims=True
- **Evaluation:** Every 2,000 steps

## What is SOAP?

SOAP (Shampoo-based Optimizer) is a preconditioned gradient method that uses block-diagonal approximations of the Hessian. It maintains low-rank statistics of squared gradients to improve convergence without the full memory cost of storing Hessian-type matrices.

## Running

```bash
sbatch submit_slurm.sh
```

## Expected Results

If SOAP successfully balances second-order information and efficiency, we expect BPB in the 1.34-1.38 range. This would indicate that Shampoo-based methods are competitive with momentum-based approaches on this scale.
