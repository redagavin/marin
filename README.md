aka Arya's Notes:

# Experiments Logs:

## 11/22:
### Pretraining Data
**Try curriculum learning with 9TB token steps(much more than chinchilla)**
This turns out to be extremely hard to do. There's three ways of preparing the dataset:
1. create a new dataset sorted by length and go through the tokenize steps.
Tokenization is extremely slow / gets stuck with 9 TB of raw sequences. Not practical for datasets of this scale.
2. sort the pretokenized dataset.
It is in zarr format which makes it hard to operate on. Zarr is designed for chunked storage, not in-place global sorting. Moving hundreds of billions of tokens across chunks is very slow.
3. do the sorting on the fly through marin/levanter framework:
Marin currently only supports shuffling permutations (linear, feistel) applied inside shuffle_ds(). adding a sort is tricky to integrate without touching low-level dataset internals.

### New Training Trick from Kimi K2:
**Try Muon-Clip. Note that its not really a new "optimizer".**
MuonClip = Muon + an extra “clip” or preprocessing step applied to certain parameters (like QK matrices in attention layers).
So it cant be implemented in optimizer layer. 
QK-Clip should be applied separately after the optimizer step in your training loop, as it requires access to attention logits.

I can't find a way to patch with the flexibity Marin gives:
1. Patch the model instance directly using callback hooks: Problem is Marin’s SpeedrunConfig does not accept a ready-made model.
2. Custom train_step_fn to Intercept the forward pass and clip attention logits: Problem is Marin’s SpeedrunConfig / executor also does not takein a custom training step logic.


## 11/5 - 11/6: Ideas and what's in scope:
### optimizers and sweep
In hello_word_lion_sweep: A sweep across llama model sizes with lion optimizer, with Chinchilla optimal steps.
Comparing 30m results with the speedrun_results in dir llama_30m:
| Metric                  | Lion                | Baseline         |
| ----------------------- | ------------------- | ---------------- |
| BPB (eval/paloma/c4_en) | 1.5646              | 1.5264           |
| Model FLOPs             | 7.47 × 10¹⁶         | 8.70 × 10¹⁶      |
So, slightly worse BPB but lower Model FLOPs → slightly more efficient algorithm. (see x-axis understanding for why we can't compare Training Hardware Flops)

Do Note that optimizers have been thoroughly tested. See https://github.com/marin-community/marin/issues/1290. 

### context size and sliding window attention:
There's this trick that they do for smaller models:
```python
model_config = dataclasses.replace(model_cfgs[size], seq_len=4096)
```
They bypass the default, which is 512 for nano and 1024 for the others.

We might consider making the seq len bigger and then do sliding window attention.
If you double/triple the sequence length without sliding windows, model FLOPs blow up quadratically. But with sliding windows, it’s roughly linear in sequence length, so model FLOPs increase moderately.

With sliding window attention:
    BPB: likely improves (more context per forward pass).
    Model FLOPs per token: increases moderately.
    Training hardware FLOPs: increases moderately.

But, why dont we just do Mistral then? The only visible difference between mistral and llama is their GQA and Sliding Window Attention.

### MoE:
Similar question goes for MoE and Mixtral.
In moe_300m_activated, they do ```from levanter.models.mixtral import MixtralConfig```, and config 32 experts and only 4 activated by specifying ```n_routed_experts=32,num_experts_per_tok=4```.
Comparing the moe_300m_activated with original llama_300m: MoE is worse.
- llama_300m has slightly lower BPB → better compression/prediction.
- MoE had better Model Flops: Itonly activates a subset of experts per token (e.g., 4 out of 32). So mathematically, the FLOPs counted per token are lower than a dense model of the same “full size,” because not all parameters are used.
- MoE had worse Hardware Flops: MoE trained on fewer total tokens (3.1B vs 6.3B), but routing, expert communication, and TPU inefficiencies inflate hardware utilization. Sparse activation reduces FLOPs per token, but hardware can be underutilized or have overhead, so total wall-clock/hardware FLOPs increase.
So the conclusion they draw is that MoE at this scale is hardware-inefficient and doesn’t justify its use for a 300M model in the Marin Speedrun context. It would be better at a much larger scale.


### Quantization:
A note on mixed-precision training: I think JAX does mixed-percision bf16 internally? i.e. the baseline is 32/16 mixed precision training. But double check me.
Marin's own conclusion: "Int8 training is much faster on the right hardware, but might lead to worse performance in terms of time-to-loss except in the early stages."
In hello_world_int8 we run int8 with default optimizer on nano, 30m, 50m, and 75m. We see that the Int8 run was slower and consumed more total training hardware FLOPs. It also leads to slightly higher bpb (worse perplexity) suggests Int8 quantization cost a little accuracy.
This is because on small models like 75M params, training is memory-bound.



# Basic Understanding:
## Instantiate Initial Model: Llama v.s. Mistral/Mixtral
In Marin Speedrun, pretraining always starts from scratch; no weights are loaded from any existing checkpoint. Instead, the model is initialized purely from its architecture using Levanter’s ModelConfig system.
Question: If i customize a Llama model (LlamaConfig) by increasing sequence length, adding sliding window attention, and applying optimizations like zMoE and GQPA, is this still doing a llama speedrun? Functionally, this creates a model that behaves very similarly to a Mistral model. How is this different from directly instantiate a MistralConfig? 
I think the answer is that they are the same. And that training from a MixtralConfig still counts for the speedrun.


## X-axis FLOPS understanding:
Regarding the two choice for x-axis, Model FLOPS and Training Hardare FLOPS:
- Model FLOPs = token-level compute cost.
- Training Hardware FLOPs = total hardware energy spent.
Algorithm improvements mostly reduce model FLOPs and indirectly reduce training hardware FLOPs.
Hardware/system improvements reduce training hardware FLOPs without changing model FLOPs.

This suggests we have two axis we can work on.
1. Algorithmic / Model-side efficiency:
This is everything about how smart your training recipe is, independent of the hardware. 
- model architecture
- optimizer
- data pipeline/curriculum
- training steps
- LR/Weight decay, etc.
- Techniques like LoRA, gradient checkpointing, sliding window attention, etc.
This affects Model FLOPS.
For the same training dataset, a lower model FLOPs model that achieves similar BPB is more efficient per token.

2. Hardware utilization/ System-side efficiency:
This is about how effectively you use the hardware’s peak performance — i.e., how close you get to theoretical FLOPs:
- Batch size tuning --> higher utilization
- mixed precision, faster attention ops --> more compute per second
- Efficient kernel fusion, JAX related optimization --> closer to peak FLOPS
A better algorithm will also affect Hardware FLOPS indirectly with by requiring less training time to go through the same amount of tokens. Since training hardware FLOPs = total time × device count × peak device FLOPs, reducing training time reduces training hardware FLOPs as well. This is why changing the x-axis metric does not affect the parento graph.

Note: This means if you optimize against the 1 A100 GPU you are using, you'd get better training_hardware_flops. 
This also means you cant compare a speedrun result on a A100 with a speedrun results on a bunch of TPUs, as in the orig repo.
But, you can roughly compare them if just looking at Model FLOPS. (Note you can't do that in cases like MoE) 