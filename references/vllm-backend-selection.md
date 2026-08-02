# vLLM Backend Selection Guide for Isambard AI

**Complete guide to moe-backend, all2all-backend, attention-backend, and linear-backend choices with quantization compatibility and Isambard-specific limitations.**

**Sources:**
- [vLLM Fused MoE Kernel Features](https://docs.vllm.ai/en/latest/design/moe_kernel_features/)
- [vLLM Expert Parallel Deployment](https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment/)
- [vLLM Engine Arguments](https://docs.vllm.ai/en/stable/configuration/engine_args/)
- [Throughputmaxxing: DeepSeek-V4-Flash on Isambard-AI](https://fergusfinn.com/blog/throughputmaxxing-v4-flash-single-node/)
- [The vLLM MoE Playbook (ROCm)](https://rocm.blogs.amd.com/software-tools-optimization/vllm-moe-guide/README.html)
- Isambard AI testing on GH200/Slingshot 11

---

## Backend Types Overview

vLLM has **four distinct backend categories** that operate at different layers:

| Backend Type | Controls | CLI Flag | When It Matters |
|--------------|----------|----------|-----------------|
| **moe-backend** | Expert FFN computation kernels | `--moe-backend` | MoE models only |
| **all2all-backend** | Expert routing communication | `--all2all-backend` | EP-enabled MoE models |
| **attention-backend** | Attention mechanism kernels | `--attention-backend` | All models |
| **linear-backend** | Dense linear layer GEMMs | `--linear-backend` | Quantized models |

**Critical:** These backends are **independent** and must be configured separately. They interact with quantization formats and hardware capabilities.

---

## 1. MoE Backend (`--moe-backend`)

Controls how expert FFN (feed-forward network) layers compute. Only relevant for MoE models.

### Available Options

| Backend | Quantization Support | Hardware | Performance | Notes |
|---------|---------------------|----------|-------------|-------|
| `auto` | All | All | Varies | Default, auto-selects |
| `triton` | BF16, FP8, FP4 | All | Good | Pre-compiled, no JIT |
| `triton_unfused` | BF16, FP8, FP4 | All | Moderate | Unfused operators |
| `deep_gemm` | **FP8 block-quantized only** | Hopper (H100/H200/GH200) | Excellent | DeepSeek models |
| `deep_gemm_mega_moe` | FP8 block-quantized | Hopper | Best | DeepSeek-V4-Flash |
| `humming` | **W4A8** (FP4 weights, FP8 activations) | Hopper | Best for W4A8 | Ant Group kernels |
| `hpc` | **FP8 only** | Hopper (H20 optimized) | Excellent | Tencent Hy3 models |
| `cutlass` | FP8, BF16 | Hopper, Ada | Good | NVIDIA CUTLASS |
| `flashinfer_cutlass` | FP8, BF16, FP4 | Hopper, Ada | Very Good | FlashInfer + CUTLASS |
| `flashinfer_trtllm` | FP8, BF16 | Hopper | Very Good | FlashInfer + TRTLLM |
| `flashinfer_b12x` | FP4 | **SM120+ only** (B200, RTX Pro 6000) | Best for FP4 | Blackwell only |
| `flashinfer_cutedsl` | FP4 | Hopper, Blackwell | Excellent | CuTe-DSL kernels |
| `marlin` | **INT4, INT8** (weight-only) | All | Good for low-bit | Weight dequant on-the-fly |
| `aiter` | FP8, BF16 | **ROCm (MI300X)** only | Good | AMD AITER kernels |
| `flydsl` | FP8, BF16 | **ROCm** only | Good | AMD FlyDSL kernels |
| `emulation` | All | All | Slow | Dequant to BF16, testing only |

### Quantization Compatibility Matrix

| Model Quantization | Recommended moe-backend | Why |
|-------------------|------------------------|-----|
| **BF16 (unquantized)** | `triton`, `cutlass`, `flashinfer_cutlass` | Best performance for dense precision |
| **FP8 (E4M3/E5M2)** | `deep_gemm`, `hpc`, `flashinfer_cutlass` | Native FP8 tensor cores on Hopper |
| **FP8 block-quantized** | `deep_gemm`, `deep_gemm_mega_moe` | DeepSeek-specific optimization |
| **W4A8 (FP4 weights, FP8 activations)** | `humming` | Only backend supporting FP4×FP8 matmul |
| **NVFP4 (NVIDIA FP4)** | `flashinfer_cutedsl`, `flashinfer_b12x` (SM120+) | NVIDIA ModelOpt format |
| **INT4 (AWQ/GPTQ)** | `marlin` | Weight-only dequant, efficient |
| **INT8** | `marlin`, `triton` | Weight-only or full INT8 |

### Isambard AI Recommendations (GH200/Hopper)

**For DeepSeek-V4-Flash (FP4 weights, FP8 activations):**
```bash
# Best performance: humming-kernels (W4A8 FP8 tensor cores)
--moe-backend humming
export VLLM_HUMMING_MOE_GEMM_TYPE=indexed
export VLLM_HUMMING_INPUT_QUANT_CONFIG='{"dtype":"float8e4m3","input_scale_group_size":128}'

# Alternative: deep_gemm_mega_moe (if humming unavailable)
--moe-backend deep_gemm_mega_moe
```

**Performance data from Isambard-AI testing (4× GH200):**
- TP4 baseline (Marlin): **5,856 tok/s**
- DP4 + humming W4A8: **17,634 tok/s** (3.0× improvement!)

**For DeepSeek-R1/V3 (FP8):**
```bash
# Best: deep_gemm for block-quantized FP8
--moe-backend deep_gemm

# Alternative: triton (more stable, slightly slower)
--moe-backend triton
```

**For Qwen3.5-MoE (FP8):**
```bash
# Best: hpc (Tencent kernels, optimized for H20/GH200)
--moe-backend hpc

# Alternative: flashinfer_cutlass
--moe-backend flashinfer_cutlass
```

**For ultra-sparse MoE (<1% activation, e.g., Llama-4-Maverick-128E):**
```bash
# Disable EP entirely - AllToAll overhead exceeds benefit
# (no --enable-expert-parallel flag)
--moe-backend triton
```

### ⚠️ Critical Limitations on Isambard

**humming-kernels:**
- ✅ **Works on GH200** with FP4×FP8 (W4A8) models
- ❌ **Does NOT support** BF16 or pure FP8 models
- ❌ **Requires** `VLLM_HUMMING_MOE_GEMM_TYPE=indexed` env var
- ❌ **Requires** `VLLM_HUMMING_INPUT_QUANT_CONFIG` for activation quantization
- ⚠️ **Only available** from doublewordAI fork (not upstream vLLM)
- ✅ **Installed** by `ivllm setup` (cloned from `doublewordai/humming`)

**deep_gemm:**
- ✅ **Works on GH200** for FP8 block-quantized models
- ❌ **Does NOT support** BF16 or FP4
- ✅ **Used by** DeepSeek-V3/V4/R1 models
- ⚠️ **Requires** DeepSeek-specific model architecture

**hpc (Tencent HPC-OPS):**
- ✅ **Works on GH200** (optimized for H20, similar architecture)
- ✅ **Supports FP8** MoE kernels
- ❌ **Only tested** on Hy3-series models
- ⚠️ **May require** model-specific adaptations for other models

**flashinfer_b12x:**
- ❌ **Does NOT work on GH200** (requires SM120+, i.e., Blackwell B200/B300)
- ✅ **Only for** FP4 models on next-gen hardware

---

## 2. All2All Backend (`--all2all-backend`)

Controls expert routing communication in EP (Expert Parallel) mode. Only relevant when `--enable-expert-parallel` is active.

### Available Options

| Backend | Communication Pattern | Best For | Requirements | Notes |
|---------|----------------------|----------|--------------|-------|
| `allgather_reducescatter` | AllGather + ReduceScatter | **General purpose** | None | **Default**, works everywhere |
| `deepep_high_throughput` | DeepEP grouped GEMM | Prefill-dominated | DeepEP installed | Continuous layout, prefill optimization |
| `deepep_low_latency` | DeepEP CUDA graphs | Decode-dominated | DeepEP installed, NCCL ≥2.30.4 | Masked layout, decode optimization |
| `flashinfer_nvlink_one_sided` | FlashInfer one-sided | MNNVL systems | NVLink between nodes, FlashInfer | Multi-node NVLink only |
| `flashinfer_nvlink_two_sided` | FlashInfer two-sided | MNNVL systems | NVLink between nodes, FlashInfer | Multi-node NVLink only |
| `pplx` | Perplexity custom | PPLEX infrastructure | PPLX backend installed | Not for general use |
| `naive` | Simple AllToAll | Debugging | None | Slow, for testing only |

### Quantization Interactions

**Critical constraint from vLLM docs:**
> "deepep_high_throughput supports only block-quantized FP8 format"

| all2all-backend | Supported Quant Formats | Notes |
|-----------------|------------------------|-------|
| `allgather_reducescatter` | **All formats** (BF16, FP8, FP4, INT4) | Most flexible |
| `deepep_high_throughput` | **FP8 block-quantized only** | DeepSeek-V3/V4 format |
| `deepep_low_latency` | FP8, BF16 | More flexible than high_throughput |
| `flashinfer_*` | FP8, BF16, FP4 | Depends on FlashInfer version |
| `naive` | All formats | Slow but compatible |

### Isambard AI Recommendations

**For Slingshot 11 fabric (UCCL-EP):**
```bash
# Recommended: allgather_reducescatter (most stable)
--all2all-backend allgather_reducescatter

# Alternative for DeepSeek models (if DeepEP available)
--all2all-backend deepep_low_latency
```

**⚠️ UCCL-EP Compatibility:**
- ✅ `allgather_reducescatter` — **Works with UCCL-EP** on Slingshot 11
- ⚠️ `deepep_*` — **May work** but requires DeepEP compiled with CXI provider
- ❌ `flashinfer_nvlink_*` — **Does NOT work** (requires NVLink between nodes, Isambard has Slingshot)
- ⚠️ `pplx` — **Not available** (Perplexity-specific backend)

**Environment variables for all2all:**
```bash
# For allgather_reducescatter (recommended)
export VLLM_ALL2ALL_BACKEND=allgather_reducescatter

# For deepep_low_latency (if using DeepEP)
export VLLM_ALL2ALL_BACKEND=deepep_low_latency
export VLLM_USE_DEEP_EP=1

# NCCL upgrade required for DeepEP (PyTorch ships old NCCL)
# See: https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment/
```

### Multi-Node Configuration Example

```bash
# Node 1 (Primary)
vllm serve deepseek-ai/DeepSeek-V3-0324 \
  --all2all-backend deepep_low_latency \
  --tensor-parallel-size 1 \
  --enable-expert-parallel \
  --data-parallel-size 16 \
  --data-parallel-size-local 8 \
  --data-parallel-address 192.168.1.100 \
  --data-parallel-rpc-port 13345

# Node 2 (Secondary)
vllm serve deepseek-ai/DeepSeek-V3-0324 \
  --all2all-backend deepep_low_latency \
  --tensor-parallel-size 1 \
  --enable-expert-parallel \
  --data-parallel-size 16 \
  --data-parallel-size-local 8 \
  --data-parallel-start-rank 8 \
  --data-parallel-address 192.168.1.100
```

---

## 3. Attention Backend (`--attention-backend`)

Controls attention mechanism kernels. Relevant for all models.

### Available Options

| Backend | Hardware | Performance | Notes |
|---------|----------|-------------|-------|
| `auto` | All | Varies | Default |
| `FLASH_ATTN` | NVIDIA, AMD | Excellent | FlashAttention-3/4 |
| `FLASHINFER` | NVIDIA | Very Good | FlashInfer kernels |
| `TRITON_ATTN` | All | Good | Pre-compiled, no JIT |
| `TRTLLM_RAGGED` | NVIDIA | Very Good | TensorRT-LLM ragged attention |
| `HPC_ATTN` | Hopper (H20) | Best for Hy3 | Tencent HPC-OPS |
| `AITER` | ROCm (MI300X) | Good | AMD AITER kernels |

### MoE Model Special Requirements

**⚠️ Critical for MoE models on GH200:**

FlashInfer autotune causes **2+ hour startup delays** on MoE models. Must disable:

```bash
# For all MoE models with TP ≥ 2 on GH200
--attention-backend TRITON_ATTN
--no-enable-flashinfer-autotune
--gdn-prefill-backend triton
--moe-backend triton
```

**Why:** FlashInfer's autotune profiling loop runs for 2+ hours on MoE models. Triton kernels are pre-compiled and work immediately.

### Isambard AI Recommendations

**For MoE models (DeepSeek, Qwen3.5, Gemma):**
```bash
--attention-backend TRITON_ATTN
--no-enable-flashinfer-autotune
```

**For dense models (Llama, Qwen dense):**
```bash
--attention-backend FLASH_ATTN  # or auto
```

**For Hy3-series (Tencent):**
```bash
--attention-backend HPC_ATTN  # Tencent HPC-OPS attention
```

---

## 4. Linear Backend (`--linear-backend`)

Controls dense linear layer GEMM kernels. Most relevant for quantized models.

### Available Options

| Backend | Quantization Support | Hardware | Notes |
|---------|---------------------|----------|-------|
| `auto` | All | All | Default |
| `cutlass` | FP8, BF16 | NVIDIA | CUTLASS kernels |
| `deep_gemm` | FP8 | Hopper | DeepSeek-specific |
| `flashinfer_cutlass` | FP8, BF16, FP4 | Hopper, Ada | FlashInfer + CUTLASS |
| `flashinfer_cudnn` | FP8, BF16 | NVIDIA | FlashInfer + cuDNN |
| `marlin` | INT4, INT8 | All | Weight-only dequant |
| `machete` | Mixed-precision | NVIDIA | Machete kernels |
| `humming` | W4A8 (FP4×FP8) | Hopper | Ant Group kernels |
| `torch` | All | All | PyTorch native (slow) |
| `triton` | All | All | Triton kernels |
| `emulation` | All | All | Dequant to BF16 (testing) |

### Quantization Compatibility

| Model Quantization | Recommended linear-backend | Why |
|-------------------|---------------------------|-----|
| **BF16** | `cutlass`, `flashinfer_cutlass` | Best for dense precision |
| **FP8** | `deep_gemm`, `cutlass`, `flashinfer_cutlass` | Native FP8 tensor cores |
| **W4A8 (FP4×FP8)** | `humming` | Only backend supporting FP4 weights |
| **NVFP4** | `flashinfer_cutedsl` | NVIDIA ModelOpt format |
| **INT4 (AWQ/GPTQ)** | `marlin` | Efficient weight-only dequant |
| **INT8** | `marlin`, `triton` | Weight-only or full INT8 |

---

## Complete Backend Configurations by Model

### DeepSeek-V4-Flash (284B/13B, FP4 weights, FP8 activations)

**Single node (4× GH200):**
```bash
vllm serve deepseek-ai/DeepSeek-V4-Flash \
  --tensor-parallel-size 1 \
  --data-parallel-size 4 \
  --enable-expert-parallel \
  --moe-backend humming \
  --linear-backend humming \
  --attention-backend TRITON_ATTN \
  --all2all-backend allgather_reducescatter \
  --no-enable-flashinfer-autotune \
  --gdn-prefill-backend triton \
  --kv-cache-dtype fp8_e4m3 \
  --dtype fp8 \
  --max-model-len 131072 \
  --gpu-memory-utilization 0.95 \
  --max-num-batched-tokens 8192 \
  --max-num-seqs 1024 \
  --compilation-config '{"max_cudagraph_capture_size":1024}'

# Required environment variables
export VLLM_HUMMING_MOE_GEMM_TYPE=indexed
export VLLM_HUMMING_INPUT_QUANT_CONFIG='{"dtype":"float8e4m3","input_scale_group_size":128}'
```

**Expected performance:** ~17,600 tok/s (4× GH200)

### DeepSeek-R1 (671B, FP8, MLA attention)

**Multi-node (2 nodes, 8× GH200 total):**
```bash
vllm serve deepseek-ai/DeepSeek-R1 \
  --tensor-parallel-size 1 \
  --data-parallel-size 8 \
  --enable-expert-parallel \
  --moe-backend deep_gemm \
  --linear-backend deep_gemm \
  --attention-backend TRITON_ATTN \
  --all2all-backend allgather_reducescatter \
  --no-enable-flashinfer-autotune \
  --gdn-prefill-backend triton \
  --kv-cache-dtype fp8_e4m3 \
  --dtype fp8 \
  --distributed-executor-backend ray
```

**Note:** MLA attention requires DP+EP for KV cache efficiency (avoids 8× duplication).

### Qwen3.5-397B-A17B-FP8

**Single node (4× GH200):**
```bash
vllm serve Qwen/Qwen3.5-397B-A17B-FP8 \
  --tensor-parallel-size 4 \
  --data-parallel-size 1 \
  --enable-expert-parallel \
  --moe-backend hpc \
  --linear-backend flashinfer_cutlass \
  --attention-backend HPC_ATTN \
  --all2all-backend allgather_reducescatter \
  --no-enable-flashinfer-autotune \
  --gdn-prefill-backend triton \
  --kv-cache-dtype fp8_e4m3 \
  --dtype fp8
```

### Qwen3-35B-A3B (FP8)

**Single node (4× GH200):**
```bash
vllm serve Qwen/Qwen3-35B-A3B \
  --tensor-parallel-size 4 \
  --data-parallel-size 1 \
  --enable-expert-parallel \
  --moe-backend triton \
  --linear-backend cutlass \
  --attention-backend TRITON_ATTN \
  --all2all-backend allgather_reducescatter \
  --no-enable-flashinfer-autotune \
  --gdn-prefill-backend triton \
  --kv-cache-dtype fp8_e4m3
```

### Llama-4-Maverick-17B-128E-FP8 (ultra-sparse, 0.78% activation)

**Single node (4× GH200):**
```bash
# NO EP flag - AllToAll overhead exceeds benefit for ultra-sparse
vllm serve meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8 \
  --tensor-parallel-size 4 \
  --data-parallel-size 1 \
  # NO --enable-expert-parallel \
  --moe-backend triton \
  --linear-backend cutlass \
  --attention-backend FLASH_ATTN \
  --kv-cache-dtype fp8_e4m3
```

---

## Troubleshooting Backend Issues

### humming-kernels Not Working

**Symptom:** `ModuleNotFoundError: No module named 'humming'` or MoE falls back to Marlin.

**Causes:**
1. humming not installed
2. Wrong quantization format (not W4A8)
3. Missing environment variables

**Fix:**
```bash
# Verify installation
uv pip show humming-kernels

# Install if missing
uv pip install git+https://github.com/doublewordai/humming.git

# Set required env vars
export VLLM_HUMMING_MOE_GEMM_TYPE=indexed
export VLLM_HUMMING_INPUT_QUANT_CONFIG='{"dtype":"float8e4m3","input_scale_group_size":128}'

# Verify model is W4A8 quantized (FP4 weights, FP8 activations)
```

### All2All Hangs with DeepEP

**Symptom:** Job hangs during warmup with `deepep_low_latency` or `deepep_high_throughput`.

**Causes:**
1. NCCL version too old (<2.30.4)
2. DeepEP not compiled with CXI provider for Slingshot
3. Missing UCCL-EP integration

**Fix:**
```bash
# Fall back to allgather_reducescatter (most stable)
--all2all-backend allgather_reducescatter

# Or upgrade NCCL (if using DeepEP)
uv pip install nvidia-nccl-cu12>=2.30.4
```

### FlashInfer Autotune Taking 2+ Hours

**Symptom:** Job stuck in `initialising` for hours on MoE models.

**Cause:** FlashInfer autotune profiling loop.

**Fix:**
```bash
--attention-backend TRITON_ATTN
--no-enable-flashinfer-autotune
--gdn-prefill-backend triton
--moe-backend triton
```

### MoE Backend Auto-Selects Wrong Kernel

**Symptom:** vLLM auto-selects `flashinfer_cutlass` instead of `deep_gemm` for FP8 model.

**Cause:** vLLM auto-selection logic bug (GitHub issue #34249).

**Fix:**
```bash
# Explicitly set moe-backend
--moe-backend deep_gemm

# Or disable FlashInfer for FP8 MoE
export VLLM_USE_FLASHINFER_MOE_FP8=0
```

---

## Quick Reference Tables

### Backend Selection by Model Type

| Model | moe-backend | linear-backend | attention-backend | all2all-backend |
|-------|-------------|----------------|-------------------|-----------------|
| **DeepSeek-V4-Flash (W4A8)** | `humming` | `humming` | `TRITON_ATTN` | `allgather_reducescatter` |
| **DeepSeek-R1/V3 (FP8)** | `deep_gemm` | `deep_gemm` | `TRITON_ATTN` | `allgather_reducescatter` |
| **Qwen3.5-MoE (FP8)** | `hpc` | `flashinfer_cutlass` | `HPC_ATTN` | `allgather_reducescatter` |
| **Qwen3-35B-A3B (FP8)** | `triton` | `cutlass` | `TRITON_ATTN` | `allgather_reducescatter` |
| **Llama-4-Maverick (ultra-sparse)** | `triton` (no EP) | `cutlass` | `FLASH_ATTN` | N/A (no EP) |
| **Hy3-series (FP8)** | `hpc` | `hpc` | `HPC_ATTN` | `allgather_reducescatter` |
| **Gemma-MoE (BF16)** | `triton` | `cutlass` | `TRITON_ATTN` | `allgather_reducescatter` |

### Quantization → Backend Mapping

| Quantization | moe-backend | linear-backend | Notes |
|--------------|-------------|----------------|-------|
| **BF16** | `triton`, `cutlass` | `cutlass`, `flashinfer_cutlass` | Default for unquantized |
| **FP8 (E4M3)** | `deep_gemm`, `hpc`, `flashinfer_cutlass` | `deep_gemm`, `cutlass` | Hopper native |
| **FP8 block-quant** | `deep_gemm`, `deep_gemm_mega_moe` | `deep_gemm` | DeepSeek format |
| **W4A8 (FP4×FP8)** | `humming` | `humming` | Only humming supports |
| **NVFP4** | `flashinfer_cutedsl` | `flashinfer_cutedsl` | NVIDIA ModelOpt |
| **INT4 (AWQ/GPTQ)** | `marlin` | `marlin` | Weight-only dequant |
| **INT8** | `marlin`, `triton` | `marlin`, `triton` | Weight or full INT8 |

---

## Summary

**Key principles:**

1. **Match backend to quantization** — FP4 needs humming, FP8 needs deep_gemm/hpc, INT4 needs marlin
2. **Disable FlashInfer autotune for MoE** — Causes 2+ hour startup on GH200
3. **Use allgather_reducescatter for Slingshot** — Most stable all2all backend for UCCL-EP
4. **Wide EP for expert-dominated models** — DeepSeek-V4-Flash: DP+EP, TP=1
5. **Hybrid EP for heavy shared layers** — Qwen3.5-397B: TP=4 within node, DP across nodes
6. **No EP for ultra-sparse models** — Llama-4-Maverick (<1% activation): EP adds overhead

**Isambard-specific stack:**
- humming-kernels (doublewordAI fork) for W4A8
- deep_gemm for DeepSeek FP8
- hpc (Tencent) for Hy3-series
- UCCL-EP for expert parallel on Slingshot 11
- allgather_reducescatter for stable all2all communication

---

**Last updated:** August 2026  
**Tested on:** Isambard AI Phase 2 (4× GH200 nodes, Slingshot 11)