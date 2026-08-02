# MoE Parallelism Strategy Guide for Isambard AI

**Wide EP vs Hybrid EP: Choosing the Right Strategy for Multi-Node MoE Deployment**

This guide explains the critical architectural decision between **Wide EP** (DP+EP, TP=1) and **Hybrid EP** (TP+DP+EP) for deploying large MoE models across multiple nodes on Isambard AI.

**Sources:**
- [vLLM Expert Parallel Deployment](https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment/)
- [The vLLM MoE Playbook (ROCm)](https://rocm.blogs.amd.com/software-tools-optimization/vllm-moe-guide/README.html)
- [DeepSeek-V4-Flash Recipe](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4-Flash)
- Isambard AI testing on GH200/Slingshot 11

---

## The Core Problem: Minimizing Inter-Node Communication

On Isambard AI, you have **two tiers of interconnect** with very different performance characteristics:

| Interconnect | Bandwidth | Latency | Scope |
|--------------|-----------|---------|-------|
| **NVLink-C2C** | 900 GB/s | ~100 ns | Within 4-GH200 node |
| **HPE Slingshot 11** | 200 Gbps (25 GB/s) per NIC | ~5-10 μs | Between nodes |

**Key insight:** NVLink is **36× faster** than Slingshot for intra-node communication. The goal is to keep as much communication as possible within the NVLink domain.

---

## Two Parallelism Strategies

### Strategy 1: Wide EP (DP + EP, TP=1)

**Configuration:**
```bash
--tensor-parallel-size 1
--data-parallel-size <N>
--enable-expert-parallel
```

**How it works:**
- Each GPU holds **complete non-MoE layers** (attention, embeddings, norms)
- **Experts distributed** across all GPUs (EP_SIZE = DP_SIZE)
- **AllToAll communication** routes tokens to experts on different GPUs
- **KV cache partitioned** by request across DP ranks

**Memory layout per GPU:**
```
Non-MoE layers:    FULL replica (attention, embeddings, norms)
Routed experts:    1/EP_SIZE of total experts (complete experts)
Shared experts:    Sharded across DP ranks
KV cache:          1/DP_SIZE of total batch
```

**Communication pattern:**
```
Token → AllToAll (route to expert GPUs) → Expert FFN → AllToAll (return) → Attention (local) → Next layer
```

**When to use Wide EP:**
✅ **Expert layers dominate** the model (>80% of parameters)  
✅ **Shared layers fit** comfortably on a single GPU  
✅ **Example:** DeepSeek-V4-Flash (284B total, expert layers are ~95% of model)

**DeepSeek-V4-Flash Example (284B/13B active):**
- Non-MoE layers: ~15GB (attention, norms, embeddings)
- Routed experts: 256 experts, 8 activated per token
- With DP=8, EP=8: Each GPU holds 32 complete experts
- **Fits easily** on single GH200 (96GB HBM3e)
- **Recommended:** `--data-parallel-size 8 --enable-expert-parallel`

---

### Strategy 2: Hybrid EP (TP + DP + EP)

**Configuration:**
```bash
--tensor-parallel-size 4        # Within-node tensor parallel
--data-parallel-size <M>        # Across-node data parallel
--enable-expert-parallel
```

**How it works:**
- **Non-MoE layers sharded** by TP (TP_SIZE ways within node)
- **Experts distributed** across EP_SIZE = TP_SIZE × DP_SIZE
- **AllToAll communication** for expert routing (requires DP>1)
- **AllReduce within TP group** for non-MoE layers
- **KV cache partitioned** by DP rank

**Memory layout per GPU:**
```
Non-MoE layers:    1/TP_SIZE of full layers (sharded by TP)
Routed experts:    1/EP_SIZE of total experts (complete experts)
Shared experts:    Sharded across TP × DP
KV cache:          1/DP_SIZE of total batch
```

**Communication pattern:**
```
Token → AllToAll (expert routing) → Expert FFN → AllToAll (return) 
  → AllReduce (TP within node) → Attention (TP-sharded) → AllReduce → Next layer
```

**When to use Hybrid EP:**
✅ **Shared layers too large** for single GPU (need TP to shard)  
✅ **Multi-node deployment** where shared layers dominate memory  
✅ **Example:** Qwen3.5-397B-A17B (large attention/embedding layers)

**Example: 400B MoE with heavy shared layers:**
- Non-MoE layers: ~120GB (too big for single 96GB GPU!)
- Routed experts: 128 experts
- **Need TP=4** to shard non-MoE layers across 4 GPUs
- **Then DP+EP** across nodes for expert distribution
- **Recommended:** `--tensor-parallel-size 4 --data-parallel-size <M> --enable-expert-parallel`

---

## Decision Framework

### Step 1: Calculate Shared Layer Memory

**Shared layers include:**
- Token embeddings
- Position embeddings
- All attention layers (QKV projections, attention output)
- Layer norms
- Final lm_head

**Formula:**
```
shared_memory_GB = shared_params_B × 2  # at bf16
# or
shared_memory_GB = shared_params_B × 1  # at fp8
```

**Example calculations:**

| Model | Shared Params | fp16/bf16 | fp8 | Fits on 1 GPU? |
|-------|--------------|-----------|-----|----------------|
| DeepSeek-V4-Flash | ~7.5B | 15 GB | 7.5 GB | ✅ Yes (easily) |
| Qwen3.5-397B | ~60B | 120 GB | 60 GB | ❌ No (needs TP) |
| Qwen3-35B-A3B | ~10B | 20 GB | 10 GB | ✅ Yes |
| GLM-5.2-743B | ~100B | 200 GB | 100 GB | ❌ No (needs TP≥2) |

### Step 2: Calculate Expert Memory

**Routed experts:**
```
expert_memory_per_gpu_GB = (total_routed_experts / EP_SIZE) × expert_size_GB
```

**Example: DeepSeek-V4-Flash (256 experts, 284B total)**
- Expert params: ~277B of 284B total (~97.5%)
- At fp8: ~277GB total expert memory
- With EP=8: 277/8 = **34.6GB per GPU**

**Example: Qwen3.5-397B-A17B (128 experts, 397B total)**
- Expert params: ~380B of 397B (~95.7%)
- At fp8: ~380GB total expert memory
- With EP=8: 380/8 = **47.5GB per GPU**

### Step 3: Check Total Memory Per GPU

**Wide EP (TP=1, DP=N, EP=N):**
```
memory_per_gpu = shared_memory + (expert_memory / EP_SIZE) + kv_cache
```

**Hybrid EP (TP=4, DP=M, EP=4×M):**
```
memory_per_gpu = (shared_memory / TP_SIZE) + (expert_memory / EP_SIZE) + kv_cache
```

**Decision tree:**

```
Can shared layers fit on 1 GPU at fp8?
│
├─ YES → Use Wide EP (DP+EP, TP=1)
│   └─ Best for: DeepSeek-V4-Flash, Qwen3-35B-A3B
│
└─ NO → Use Hybrid EP (TP+DP+EP)
    └─ TP_SIZE = ceil(shared_memory_GB / 86.4)
    └─ Then add DP+EP for expert distribution
    └─ Best for: Qwen3.5-397B, GLM-5.2-743B
```

---

## Isambard AI Specific Guidance

### Single Node (4× GH200)

**For models with small shared layers (<80GB at fp8):**
```bash
# Wide EP: Maximum expert distribution
--tensor-parallel-size 1
--data-parallel-size 4
--enable-expert-parallel
--moe-backend triton
--gdn-prefill-backend triton
--attention-config '{"backend":"TRITON_ATTN"}'
--no-enable-flashinfer-autotune
```

**For models with large shared layers (>80GB at fp8):**
```bash
# Hybrid EP: Shard shared layers within node
--tensor-parallel-size 4
--data-parallel-size 1
--enable-expert-parallel
--moe-backend triton
--gdn-prefill-backend triton
--attention-config '{"backend":"TRITON_ATTN"}'
--no-enable-flashinfer-autotune
```

### Multi-Node (N × 4-GH200 nodes)

**For DeepSeek-V4-Flash type (expert-dominated):**
```bash
# Wide EP across nodes
--tensor-parallel-size 1
--data-parallel-size <N×4>
--enable-expert-parallel
--distributed-executor-backend ray
# Add Slingshot env vars
```

**For Qwen3.5-397B type (heavy shared layers):**
```bash
# Hybrid EP: TP=4 within node, DP+EP across nodes
--tensor-parallel-size 4
--pipeline-parallel-size <N>
--enable-expert-parallel
--distributed-executor-backend ray
--compilation-config '{"pass_config": {"fuse_allreduce_rms": false}}'
# Add Slingshot env vars
```

---

## Communication Overhead Comparison

### Wide EP (DP+EP, TP=1)

**AllToAll traffic per token:**
- Routes token to expert's GPU
- Volume: `hidden_size × sizeof(fp8)` per token
- **Cross-node:** If expert on different node, crosses Slingshot

**Optimization:** Expert placement matters!
- vLLM's `determine_expert_map` tries to balance expert load
- For static routing patterns, can manually assign experts to minimize cross-node traffic

### Hybrid EP (TP+DP+EP)

**Two types of communication:**
1. **AllToAll** (expert routing) - same as Wide EP
2. **AllReduce** (TP synchronization) - **stays within node** if TP≤4

**Key advantage:** TP AllReduce happens over NVLink (900 GB/s), not Slingshot!

**Trade-off:** Extra AllReduce overhead, but keeps TP communication local.

---

## Performance Benchmarks (from ROCm MoE Playbook)

### DeepSeek-R1 (671B, MLA attention, 256 experts)

**TP=8 + EP (not ideal for MLA):**
- KV cache duplicated 8× (wastes memory)
- 52% higher throughput at low concurrency (64 requests)
- **80% lower TTFT** than DP=8 + EP

**DP=8 + EP (ideal for MLA):**
- KV cache partitioned (8× more efficient)
- 47% higher throughput at high concurrency (1024 requests)
- **Enables 8× larger batches**

**Conclusion for MLA models:** Use DP+EP (Wide EP) for high throughput, TP+EP for low latency.

### Qwen3-235B-A22B (128 experts, 6.25% activation density)

**Crossover point:** 256-512 concurrent requests
- **<256 requests:** TP=8 + EP faster (lower latency)
- **>512 requests:** DP=8 + EP faster (higher throughput)

### Llama-4-Maverick-17B-128E (0.78% activation density - ultra-sparse)

**Surprising result:** EP=0 outperforms EP=1 by 7-12%

**Why:** AllToAll overhead exceeds benefit for ultra-sparse models

**Recommendation:** For models with <1% activation density, consider **not using EP** at all.

---

## Expert Activation Density Matters

**Formula:**
```
activation_density = (experts_per_token / total_routed_experts) × 100%
```

| Density | EP Recommendation | Why |
|---------|------------------|-----|
| **<1%** (ultra-sparse) | EP=0 (no EP flag) | AllToAll overhead > benefit |
| **1-3%** (sparse) | EP=1 optional | Marginal benefit |
| **>3%** (dense) | EP=1 recommended | AllToAll efficiency wins |
| **MLA/MQA models** | EP=1 required | KV cache handling |

**Examples:**
- Llama-4-Maverick-17B-128E: 1/128 = **0.78%** → EP=0
- DeepSeek-R1: 8/256 = **3.13%** → EP=1 (also MLA)
- Qwen3-235B-A22B: 8/128 = **6.25%** → EP=1

---

## Special Case: MLA/MQA Attention

**Models:** DeepSeek-V2/V3/R1, Qwen-MoE variants

**Problem:** Multi-Latent Attention (MLA) and Multi-Query Attention (MQA) use **single KV head**.

**With TP:**
- Cannot shard KV cache along head dimension (only 1 head!)
- **Result:** Full KV cache duplicated on every TP rank
- **Memory waste:** TP_SIZE × larger KV cache

**With DP+EP:**
- KV cache partitioned by request across DP ranks
- **Each GPU holds:** 1/DP_SIZE of total KV cache
- **Enables:** 8× larger batches with DP=8 vs TP=8

**Recommendation for MLA/MQA:**
- **Always use DP+EP** (Wide EP) for production throughput
- TP+EP only for low-latency, low-concurrency scenarios

---

## Configuration Examples

### DeepSeek-V4-Flash (284B/13B, expert-dominated)

**Single node (4× GH200):**
```bash
# Wide EP: TP=1, DP=4, EP=4
vllm serve deepseek-ai/DeepSeek-V4-Flash \
  --tensor-parallel-size 1 \
  --data-parallel-size 4 \
  --enable-expert-parallel \
  --dtype fp8 \
  --kv-cache-dtype fp8_e4m3 \
  --max-model-len 131072 \
  --gpu-memory-utilization 0.90 \
  --moe-backend triton \
  --gdn-prefill-backend triton \
  --attention-config '{"backend":"TRITON_ATTN"}' \
  --no-enable-flashinfer-autotune
```

**Multi-node (2 nodes, 8× GH200 total):**
```bash
# Wide EP: TP=1, DP=8, EP=8
vllm serve deepseek-ai/DeepSeek-V4-Flash \
  --tensor-parallel-size 1 \
  --data-parallel-size 8 \
  --enable-expert-parallel \
  --distributed-executor-backend ray \
  --dtype fp8 \
  ...
```

### Qwen3.5-397B-A17B-FP8 (heavy shared layers)

**Single node (4× GH200):**
```bash
# Hybrid EP: TP=4, DP=1, EP=4
vllm serve Qwen/Qwen3.5-397B-A17B-FP8 \
  --tensor-parallel-size 4 \
  --data-parallel-size 1 \
  --enable-expert-parallel \
  --dtype fp8 \
  --kv-cache-dtype fp8_e4m3 \
  --max-model-len 65536 \
  --gpu-memory-utilization 0.90 \
  --moe-backend triton \
  --gdn-prefill-backend triton \
  --attention-config '{"backend":"TRITON_ATTN"}' \
  --no-enable-flashinfer-autotune
```

**Multi-node (4 nodes, 16× GH200 total):**
```bash
# Hybrid EP: TP=4, PP=4, EP=4
# Note: EP activates because TP_SIZE × DP_SIZE = 4×1 = 4 > 1
vllm serve Qwen/Qwen3.5-397B-A17B-FP8 \
  --tensor-parallel-size 4 \
  --pipeline-parallel-size 4 \
  --enable-expert-parallel \
  --distributed-executor-backend ray \
  --compilation-config '{"pass_config": {"fuse_allreduce_rms": false}}' \
  --dtype fp8 \
  ...
```

---

## Troubleshooting

### EP Not Activating

**Symptom:** `--enable-expert-parallel` has no effect.

**Cause:** EP only activates when `TP_SIZE × DP_SIZE > 1`.

**Check:**
```bash
# These do NOT activate EP:
--tensor-parallel-size 1 --data-parallel-size 1 --enable-expert-parallel  # 1×1=1, EP inactive
--pipeline-parallel-size 4 --enable-expert-parallel  # TP=1, DP=1 per stage, EP inactive

# These DO activate EP:
--tensor-parallel-size 4 --enable-expert-parallel  # 4×1=4>1, EP active
--data-parallel-size 4 --enable-expert-parallel  # 1×4=4>1, EP active
--tensor-parallel-size 2 --data-parallel-size 2 --enable-expert-parallel  # 2×2=4>1, EP active
```

### AllToAll Kernel Failures

**Symptom:** Crash with AllToAll-related error.

**Cause:** AllToAll requires `dp_size > 1`.

**Fix:**
- Ensure `--data-parallel-size > 1` for AllToAll communication
- TP-only configurations use AllReduce, not AllToAll

### KV Cache OOM with TP

**Symptom:** OOM during KV cache allocation with TP>1.

**Cause:** MLA/MQA models duplicate KV cache across TP ranks.

**Fix:**
- Switch to DP+EP (Wide EP) for KV cache partitioning
- Or reduce `max-model-len` to decrease KV cache size

---

## Quick Reference Table

| Model Type | Shared Layers | Strategy | TP | DP | EP |
|------------|--------------|----------|----|----|----|
| **DeepSeek-V4-Flash** | Small (<20GB) | Wide EP | 1 | N | N |
| **DeepSeek-R1 (MLA)** | Medium | Wide EP (DP+EP) | 1 | N | N |
| **Qwen3-35B-A3B** | Small | Wide EP | 1 | N | N |
| **Qwen3.5-397B** | Large (>80GB) | Hybrid EP | 4 | M | 4×M |
| **GLM-5.2-743B** | Very Large | Hybrid EP | 4 | M | 4×M |
| **Llama-4-Maverick-17B** | Small | Wide EP (no EP flag) | 1 | N | 0 |

**Notes:**
- N = number of GPUs (single node: N=4, multi-node: N=4×nodes)
- M = DP_SIZE for hybrid (typically nodes count)
- For ultra-sparse models (<1% activation), consider EP=0

---

## Summary

**Wide EP (DP+EP, TP=1):**
- ✅ Best for expert-dominated models (DeepSeek-V4-Flash)
- ✅ Simpler communication (only AllToAll)
- ✅ KV cache partitioned (better for MLA/MQA)
- ❌ Requires shared layers to fit on single GPU

**Hybrid EP (TP+DP+EP):**
- ✅ Best for models with heavy shared layers
- ✅ TP keeps non-MoE communication within node (NVLink)
- ❌ Extra AllReduce overhead
- ❌ More complex configuration

**Decision rule:**
1. Calculate shared layer memory
2. If fits on 1 GPU → Wide EP
3. If doesn't fit → Hybrid EP with TP = ceil(shared_memory / 86.4GB)
4. For MLA/MQA → Prefer Wide EP (DP+EP) for KV cache efficiency
5. For ultra-sparse (<1% activation) → Consider no EP at all

---

**Last updated:** August 2026  
**Tested on:** Isambard AI Phase 2 (4× GH200 nodes, Slingshot 11)