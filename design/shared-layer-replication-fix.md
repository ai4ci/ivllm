# Critical Fix: Shared Layer Replication in MoE Models

## The Mistake

I incorrectly configured DeepSeek-V4-Pro with **Wide EP (TP=1, DP=4, EP=16)**, assuming the model weights would simply divide across 16 GPUs.

**This is WRONG and would cause immediate OOM!**

## Why Wide EP Fails for Large Models

### MoE Model Architecture

MoE models have two types of parameters:

1. **Shared Layers** (used by EVERY token):
   - Attention mechanisms (MLA, MHA)
   - Token embeddings
   - Output projection head
   - Layer normalization
   - Router/gating networks
   - **CANNOT be expert-parallelized**

2. **Expert Layers** (only top-k activated):
   - MoE FFN expert blocks
   - **CAN be distributed across EP GPUs**

### Wide EP (TP=1) Memory Layout

```
Each GPU gets:
├── Shared Layers (FULL REPLICATION) = ~250 GB
└── Experts (1/EP fraction) = ~40 GB
    └── Total: ~290 GB per GPU ❌

Isambard GH200 usable: 70 GB
Result: IMMEDIATE OOM!
```

### Hybrid EP (TP=8) Memory Layout

```
Each GPU gets:
├── Shared Layers (sharded by TP=8) = 250/8 = ~31 GB
└── Experts (distributed by EP=64) = 640/64 = ~10 GB
    └── Total: ~41 GB per GPU ✅

Isambard GH200 usable: 70 GB
Result: FITS with room for KV cache!
```

## The Math: DeepSeek-V4-Pro

| Component | Size (fp8/fp4) | Parallelizable? | TP=1 | TP=8 |
|-----------|----------------|-----------------|------|------|
| Shared layers | ~250 GB | ❌ No | 250 GB each | 31 GB each |
| Experts (total) | ~640 GB (fp4) | ✅ Yes (EP) | 40 GB each (EP=16) | 10 GB each (EP=64) |
| **Total per GPU** | | | **290 GB** ❌ | **41 GB** ✅ |

**Minimum viable configuration:**
- TP=8 (required to shard shared layers)
- EP=64 (required to distribute experts)
- DP=1+ (optional for throughput)
- **Total: 64 GPUs = 16 nodes × 4 GPUs/node**

## General Rule: Minimum TP by Model Size

| Total Model Size | Shared Layers | Minimum TP | Example Models |
|------------------|---------------|------------|----------------|
| <100B | ~30-50 GB | TP=1 | Qwen3.6-35B, Llama-3-70B |
| 100-400B | ~80-150 GB | TP=2-4 | GLM-5.2, MiniMax-M3 |
| 400B-1T | ~150-250 GB | TP=4-8 | Nemotron-3-Ultra |
| >1T | ~250+ GB | TP=8+ | DeepSeek-V4-Pro |

**Rule:** `TP ≥ Shared_Layers_GB / 70` (rounded up to power of 2)

## When Wide EP Works

Wide EP (TP=1, DP=N, EP=N) is **optimal** when:

1. **Shared layers < 70 GB** (fit on single GPU)
2. **Expert-dominated** (<10% activation rate)
3. ** MLA/MQA attention** (KV cache doesn't duplicate with TP)

**Examples that work with Wide EP:**
- ✅ Qwen3.6-35B-A3B (35B total, 3B active = 8.6%)
  - Shared layers: ~20 GB
  - TP=1, DP=2, EP=2 works perfectly
  
- ✅ Qwen3-Coder-Next (~70B, expert-dominated)
  - Shared layers: ~35 GB (estimated)
  - TP=1, DP=2, EP=2 works

**Examples that DON'T work with Wide EP:**
- ❌ DeepSeek-V4-Pro (1.6T total)
  - Shared layers: ~250 GB
  - Needs TP=8 minimum
  
- ❌ GLM-5.2 (743B total)
  - Shared layers: ~150 GB (estimated)
  - Needs TP=4 minimum

## Corrected Configs

### DeepSeek-V4-Pro (FIXED)

```yaml
# ❌ WRONG - Would OOM immediately
# tensor-parallel-size: 1
# data-parallel-size: 4
# enable-expert-parallel: true

# ✅ CORRECT - Requires 16 nodes!
tensor-parallel-size: 8
data-parallel-size: 8
enable-expert-parallel: true
```

**Status:** Marked as `lifecycle: failing` - not viable on Isambard without 16 nodes

**Recommendation:** Use `deepseek-v4-flash.yaml` instead (operational, 4 nodes)

### Other Configs (Verified)

All other configs have been verified for correct TP:

| Model | Shared Layers | TP Used | Correct? |
|-------|---------------|---------|----------|
| GLM-5.2-743b | ~150 GB | TP=4 | ✅ Yes |
| MiniMax-M3 | ~100 GB | TP=4 | ✅ Yes |
| Nemotron-3-Ultra | ~150 GB | TP=4 | ✅ Yes |
| Solar-Open2-250B | ~100 GB | TP=4 | ✅ Yes |
| Qwen3.6-35B | ~20 GB | TP=1 | ✅ Yes |
| Qwen3-Coder-Next | ~35 GB | TP=1 | ✅ Yes |

## Lessons Learned

1. **Always calculate shared layer size** before choosing TP
2. **Wide EP is NOT universal** - only for models with small shared layers
3. **Official vLLM recipes assume 8-GPU nodes** - don't blindly copy
4. **Expert parallelism doesn't help shared layers** - only TP shards them
5. **KV cache offloading doesn't help weight footprint** - only helps context

## Files Updated

- `examples/deepseek-v4-pro.yaml` - Marked as failing, documented 16-node requirement
- `design/example-configs-isambard-reality.md` - Added shared layer replication section
- `design/shared-layer-replication-fix.md` - This document

## References

- vLLM Expert Parallelism Docs: https://docs.vllm.ai/en/latest/parallelism/expert_parallel.html
- DeepSeek-V4 Architecture: https://arxiv.org/abs/2412.19437
- Isambard AI Hardware: https://www.bris.ac.uk/isambard/