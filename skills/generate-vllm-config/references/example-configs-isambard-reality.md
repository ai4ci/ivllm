# Example Configs - Isambard AI Reality Check

## Critical Hardware Constraints

**Isambard AI Phase 1 & 2 (GH200):**
- **4 GPUs per node** (NOT 8 like H200/H100 setups in vLLM recipes)
- **96 GB HBM3e per GPU**
- **~70 GB usable** (25 GB headroom required by BriCS policy)
- **NVLink-C2C intra-node** (900 GB/s)
- **Slingshot 11 inter-node** (200 Gbps, ~25 GB/s - NOT NVLink!)

**Total Usable VRAM:**
| Nodes | GPUs | Total Usable VRAM |
|-------|------|-------------------|
| 1 | 4 | 280 GB |
| 2 | 8 | 560 GB |
| 3 | 12 | 840 GB |
| 4 | 16 | 1,120 GB |

## Why Official vLLM Recipes Don't Apply

The official vLLM recipes (recipes.vllm.ai) assume:
1. ❌ **8 GPUs per node** (we have 4)
2. ❌ **Full 96 GB usable** (we have 70 GB)
3. ❌ **NVLink between all GPUs** (we have Slingshot inter-node)
4. ❌ **No KV cache offloading** (we need it for large models)

**Following official recipes verbatim will cause OOM errors on Isambard!**

---

## Model Requirements (Revised for Isambard)

### DeepSeek-V4-Pro (1.6T/49B, FP4+FP8)

**Weight Footprint:** ~960 GB
- MoE experts (FP4): ~640 GB
- Dense layers (FP8): ~320 GB
- **Shared layers (attention, embeddings, norms): ~250 GB** ⚠️

**CRITICAL CONSTRAINT:**
Shared layers CANNOT be expert-parallelized - they are used by EVERY token.
With Wide EP (TP=1), shared layers REPLICATE on every GPU.
**250 GB shared layers > 70 GB usable = IMPOSSIBLE with TP=1!**

**Minimum Viable Configuration:**
- **TP=8** (shard shared layers: 250/8 = ~31 GB per GPU) ✅
- **EP=64** (distribute experts: 640/64 = ~10 GB per GPU) ✅
- **DP=1** (or more for throughput)
- **Total: 64 GPUs = 16 nodes × 4 GPUs/node** 🚨

**Parallelism:** Hybrid EP (NOT Wide EP!)
- TP=8 shards the shared attention/embedding layers
- EP=64 distributes the 256 experts
- DP=1+ for throughput scaling

**KV Cache Offloading:** REQUIRED
- Mooncake connector essential for any practical context
- Weights alone use ~51 GB/GPU, leaving ~19 GB for KV

**Key Dependencies:**
- `humming-kernels` (doublewordAI fork) for W4A8 FP4×FP8
- DeepEP for all2all communication
- NCCL ≥2.30.4

**⚠️ RECOMMENDATION: Use DeepSeek-V4-Flash Instead**
- V4-Flash fits on 4 nodes (16 GPUs) vs 16 nodes for Pro
- V4-Flash has operational config: `examples/deepseek-v4-flash.yaml`
- ~95% of Pro capability at 40% the size
- Much more practical for Isambard AI

---

### GLM-5.2-FP8 (743B/39B)

**Weight Footprint:** ~743 GB (FP8)

**Minimum Configuration:**
- **3 nodes × 4 GPUs = 12 GPUs**
- 743 GB / 12 = **62 GB per GPU** ✅ (fits in 70 GB)

**Parallelism:** Hybrid EP
- TP=4 (within node), DP=3, EP=12
- TP=4 shards large shared attention layers

**KV Cache Offloading:** RECOMMENDED
- Enables full 1M context
- Mooncake connector

**Key Dependencies:**
- `deep_gemm` for FP8 MoE
- DeepEP for all2all
- MTP speculative decoding (5 tokens)

---

### MiniMax-M3 (427B/26B, AWQ INT4)

**Weight Footprint:** ~214 GB (INT4 = 0.5 bytes/param)

**Minimum Configuration:**
- **1 node × 4 GPUs = 4 GPUs** ✅
- 214 GB / 4 = **53.5 GB per GPU** ✅ (fits with room for KV!)

**Parallelism:** TP only (no EP needed)
- TP=4 (within node)
- INT4 weight-only quantization doesn't benefit from EP

**KV Cache Offloading:** NOT REQUIRED
- Full 256K context fits on-GPU
- 53.5 GB weights + 16.5 GB KV = 70 GB

**Key Dependencies:**
- `marlin` backend for INT4 dequantization
- MSA sparse attention requires `block-size=128`
- EAGLE3 speculative decoding (3 tokens)

---

### Nemotron-3-Ultra-550B (550B/55B, FP8)

**Weight Footprint:** ~550 GB (FP8)

**Minimum Configuration:**
- **2 nodes × 4 GPUs = 8 GPUs**
- 550 GB / 8 = **68.75 GB per GPU** ⚠️ (TIGHT FIT!)

**Parallelism:** Hybrid EP
- TP=4, DP=2, EP=8
- Hybrid architecture (Transformer + Mamba)

**KV Cache Offloading:** STRONGLY RECOMMENDED
- Only ~1.25 GB headroom without offloading
- Mooncake essential for practical context lengths

**Key Dependencies:**
- `deep_gemm` for FP8 MoE
- `flashinfer` for Mamba SSM layers
- Nemotron H-MTP speculative decoding (5 tokens)

**⚠️ WARNING:** This is a very tight fit. Monitor VRAM carefully!

---

### Solar-Open2-250B (250B/15B, FP8)

**Weight Footprint:** ~500 GB (estimated FP8)

**Minimum Configuration:**
- **2 nodes × 4 GPUs = 8 GPUs**
- 500 GB / 8 = **62.5 GB per GPU** ✅ (fits in 70 GB)

**Parallelism:** Hybrid EP
- TP=4, DP=2, EP=8

**KV Cache Offloading:** OPTIONAL
- ~7.5 GB headroom for KV cache
- Sufficient for moderate context (128K-256K)

**Key Dependencies:**
- **Requires Upstage vLLM fork** (not upstream!)
- `deep_gemm` for FP8 MoE
- Custom logits processors

---

### Qwen3.6-35B-A3B (35B/3B, FP8)

**Weight Footprint:** ~35 GB (FP8)

**Minimum Configuration:**
- **1 node × 2 GPUs = 2 GPUs** ✅
- 35 GB / 2 = **17.5 GB per GPU** ✅ (EASY FIT!)

**Parallelism:** Wide EP
- TP=1, DP=2, EP=2
- Expert-dominated (8.6% activation)

**KV Cache Offloading:** NOT REQUIRED
- Full 1M context fits on-GPU!
- 17.5 GB weights + 52.5 GB KV = 70 GB

**Key Dependencies:**
- `triton` MoE backend (stable, pre-compiled)
- No special dependencies

---

### Qwen3-Coder-Next-FP8 (~70B, FP8)

**Weight Footprint:** ~70 GB (estimated FP8)

**Minimum Configuration:**
- **1 node × 2 GPUs = 2 GPUs** ✅
- 70 GB / 2 = **35 GB per GPU** ✅ (fits with room for KV!)

**Parallelism:** Wide EP
- TP=1, DP=2, EP=2

**KV Cache Offloading:** NOT REQUIRED
- Full 262K context fits on-GPU
- 35 GB weights + 35 GB KV = 70 GB

**Key Dependencies:**
- `triton` MoE backend

---

## Critical: Shared Layer Replication in Wide EP

**The Problem:**

In MoE models, parameters fall into two categories:

1. **Shared Layers** (CANNOT be expert-parallelized):
   - Attention layers (MLA, MHA, GQA)
   - Token embeddings
   - Output head
   - Layer norms
   - Router/gating networks
   - **Used by EVERY token, on EVERY GPU**

2. **Expert Layers** (CAN be expert-parallelized):
   - MoE expert FFN blocks
   - **Only top-k experts activated per token**
   - Can be distributed across EP GPUs

**Wide EP (TP=1, DP=N, EP=M):**
```
GPU 0: [Shared Layers 250GB] + [Experts 1/16: 40GB] = 290GB ❌
GPU 1: [Shared Layers 250GB] + [Experts 1/16: 40GB] = 290GB ❌
... (replicated on every GPU)
```
**Result:** Shared layers replicate on EVERY GPU → IMMEDIATE OOM!

**Hybrid EP (TP=N, DP=M, EP=N×M):**
```
GPU 0: [Shared Layers 250GB/8: 31GB] + [Experts 1/64: 10GB] = 41GB ✅
GPU 1: [Shared Layers 250GB/8: 31GB] + [Experts 1/64: 10GB] = 41GB ✅
... (sharded across TP group)
```
**Result:** Shared layers sharded by TP → FITS!

**Rule of Thumb:**

| Model Size | Shared Layers | Minimum TP | Reason |
|------------|---------------|------------|--------|
| <100B total | ~50 GB | TP=1 | Fits on single GPU |
| 100-400B | ~100-150 GB | TP=2-4 | Need to shard shared layers |
| 400B-1T | ~150-250 GB | TP=4-8 | Large shared attention |
| >1T | ~250+ GB | TP=8+ | Massive shared layers |

**DeepSeek-V4-Pro Example:**
- Total: 1.6T params
- Shared layers: ~250 GB (fp8)
- **Minimum TP=8** (250/8 = 31 GB per GPU)
- **Wide EP (TP=1) is IMPOSSIBLE**

**Qwen3.6-35B Example:**
- Total: 35B params
- Shared layers: ~20 GB (fp8)
- **TP=1 works fine** (20 GB < 70 GB)
- **Wide EP is optimal** (expert-dominated)

---

## KV Cache Offloading Guide

### When to Use

**REQUIRED:**
- DeepSeek-V4-Pro (1.6T) - for 1M context
- GLM-5.2 (743B) - for 1M context
- Nemotron-3-Ultra (550B) - tight VRAM fit

**OPTIONAL:**
- Solar-Open2-250B - for longer context
- MiniMax-M3 - if you need >256K
- Qwen3.6-35B - not needed (fits easily)

### Offloading Connectors

**1. MooncakeStoreConnector (RECOMMENDED)**
```yaml
enable-kv-cache-offload: true
kv-cache-offload-config: '{"connector_type": "mooncake", "memory_budget": 64}'
```
- Pools CPU DRAM across multiple nodes
- Best for multi-node deployments
- Enables cross-node KV sharing

**2. SimpleCPUOffloadConnector**
```yaml
enable-kv-cache-offload: true
kv-cache-offload-config: '{"connector_type": "simple", "memory_budget": 64}'
```
- Single-node only
- Simpler setup
- No cross-node sharing

**3. LMCacheMPConnector**
```yaml
enable-kv-cache-offload: true
kv-cache-offload-config: '{"connector_type": "lmcache", "memory_budget": 64}'
```
- Requires companion `lmcache` server process
- Node-local pool
- Good performance but more complex setup

---

## Backend Selection Matrix (Isambard-Specific)

### MoE Backend by Quantization

| Quantization | Backend | Models | Notes |
|--------------|---------|--------|-------|
| **W4A8 (FP4×FP8)** | `humming` | DeepSeek-V4-Pro | Requires doublewordAI fork |
| **FP8 block-quantized** | `deep_gemm` | GLM-5.2, Nemotron, Solar | Best FP8 performance |
| **INT4 (AWQ/GPTQ)** | `marlin` | MiniMax-M3 | Weight-only dequant |
| **BF16/FP8 (unquantized)** | `triton` | Qwen3.6, Qwen3-Coder | Stable, pre-compiled |

### All2All Backend

| Backend | Use Case | Models | Notes |
|---------|----------|--------|-------|
| `allgather_reducescatter` | General purpose, most stable | Qwen3.6, Qwen3-Coder | Default choice |
| `deepep_low_latency` | Decode-optimized | DeepSeek, GLM, Nemotron, Solar | Requires DeepEP + NCCL ≥2.30.4 |

### Attention Backend

**ALL MoE models on Isambard:**
```yaml
attention-backend: TRITON_ATTN
enable-flashinfer-autotune: false
gdn-prefill-backend: triton
```

**Why:** FlashInfer autotune causes 2+ hour startup delays on MoE models. Triton kernels are pre-compiled and work immediately.

---

## Testing Priority

### Priority 1 (Test First - Easy Fits)

1. **qwen3.6-long-context.yaml** (2 GPUs, 1 node)
   - 17.5 GB per GPU - plenty of headroom
   - Full 1M context on-GPU
   - No special dependencies

2. **qwen3-coder-next-long-context.yaml** (2 GPUs, 1 node)
   - 35 GB per GPU - good headroom
   - Full 262K context on-GPU
   - No special dependencies

3. **minimax-m3.yaml** (4 GPUs, 1 node)
   - 53.5 GB per GPU - fits with KV cache room
   - Full 256K context on-GPU
   - Only requires marlin (included in vLLM)

### Priority 2 (Requires Additional Setup)

4. **solar-open2-250B.yaml** (8 GPUs, 2 nodes)
   - Requires Upstage vLLM fork
   - 62.5 GB per GPU - good fit
   - Test after Priority 1 configs work

5. **nemotron-3-ultra-550B-A55B-FP8.yaml** (8 GPUs, 2 nodes)
   - 68.75 GB per GPU - VERY TIGHT
   - Requires Mooncake offloading for practical use
   - Test with monitoring for OOM

### Priority 3 (Complex Dependencies)

6. **glm-5.2-743b.yaml** (12 GPUs, 3 nodes)
   - Requires 3 nodes (expensive!)
   - Requires DeepEP + deep_gemm
   - Test after confirming multi-node works

### ❌ NOT VIABLE ON ISAMBARD

7. **deepseek-v4-pro.yaml** - **REQUIRES 16 NODES (64 GPUs)**
   - Shared layers (~250 GB) require TP=8 minimum
   - Experts require EP=64 to fit
   - Total: 8 × 8 = 64 GPUs = 16 nodes
   - **Recommendation: Use deepseek-v4-flash.yaml instead** (4 nodes)
   - Kept for reference only, lifecycle: failing

---

## Common Issues & Fixes

### Issue: OOM on model load

**Symptoms:**
```
torch.cuda.OutOfMemoryError: CUDA out of memory.
```

**Fix:**
1. Verify node count matches config requirements
2. Check `gpu-memory-utilization` is ≤0.95
3. Enable KV cache offloading for large models
4. Reduce `max-num-seqs` or `max-num-batched-tokens`

### Issue: FlashInfer autotune hangs for 2+ hours

**Symptoms:**
- Startup takes forever
- Stuck at "Running FlashInfer autotune..."

**Fix:** Add these keys:
```yaml
attention-backend: TRITON_ATTN
enable-flashinfer-autotune: false
gdn-prefill-backend: triton
```

### Issue: All2All communication hangs

**Symptoms:**
- Job starts but hangs during first forward pass
- NCCL timeout errors

**Fix:** Fall back to stable backend:
```yaml
all2all-backend: allgather_reducescatter
```

### Issue: No module named 'humming'

**Symptoms:**
```
ModuleNotFoundError: No module named 'humming'
```

**Fix:**
```bash
uv pip install git+https://github.com/doublewordai/humming.git
```

### Issue: DeepEP not found

**Symptoms:**
```
ImportError: No module named 'deepep'
```

**Fix:**
```bash
# Install DeepEP with CXI provider for Slingshot 11
uv pip install git+https://github.com/deepseek-ai/DeepEP.git
```

### Issue: KV cache OOM with long context

**Symptoms:**
- Model loads fine
- OOM when processing long prompts

**Fix:** Enable Mooncake offloading:
```yaml
enable-kv-cache-offload: true
kv-cache-offload-config: '{"connector_type": "mooncake", "memory_budget": 64}'
```

---

## Slurm Job Script Template

```bash
#!/bin/bash
#SBATCH --job-name=vllm-test
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --gres=gpu:4
#SBATCH --time=12:00:00
#SBATCH --partition=standard
#SBATCH --qos=standard

# Load modules
module load brics/apptainer-multi-node
module load brics/nccl

# Set NCCL for Slingshot 11
export FI_PROVIDER=cxi
export NCCL_NET=aws-ofi-nccl
export NCCL_SOCKET_IFNAME=hsn0,hsn1,hsn2,hsn3

# Run vLLM
ivllm run examples/your-config.yaml \
  --model-path /path/to/model \
  --port 8000
```

---

## Files Updated

All configs updated with Isambard-specific constraints:

| File | Nodes | GPUs | Status | Notes |
|------|-------|------|--------|-------|
| `deepseek-v4-pro.yaml` | **16** | **64** | ❌ failing | NOT VIABLE - needs 16 nodes! |
| `glm-5.2-743b.yaml` | 3 | 12 | experimental | Requires DeepEP |
| `minimax-m3.yaml` | 1 | 4 | experimental | Best INT4 option |
| `nemotron-3-ultra-550B-A55B-FP8.yaml` | 2 | 8 | experimental | Tight VRAM fit |
| `solar-open2-250B.yaml` | 2 | 8 | experimental | Needs Upstage fork |
| `qwen3.6-long-context.yaml` | 1 | 2 | experimental | Easy fit! |
| `qwen3-coder-next-long-context.yaml` | 1 | 2 | experimental | Easy fit! |

**Files NOT changed (lifecycle: operational):**
- `deepseek-v4-flash.yaml` ✅ (4 nodes - use this instead of V4-Pro!)
- `gemma-4-31B-it.yaml` ✅
- `mistral-medium-3.5-128B.yaml` ✅
- `nemotron-3-super-120B-A12B-FP8.yaml` ✅
- `qwen3.5-397b-a17b-fp8.yaml` ✅
- `qwen3.6-35b-a3b-fp8.yaml` ✅

---

## Next Steps

1. **Test Priority 1 configs** on actual Isambard hardware
2. **Verify KV offloading** works with Mooncake on multi-node
3. **Document actual VRAM usage** from slurm-vllm logs
4. **Update lifecycle metadata** based on test results
5. **Create Slurm job templates** for each node count