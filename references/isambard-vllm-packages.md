# Isambard AI vLLM Dependencies & Packages Reference

This document summarizes the key packages, libraries, and dependencies installed by `ivllm setup` for running vLLM on Isambard AI HPC. Understanding these components helps when optimizing configs for specific models and hardware.

---

## Hardware Context: Isambard AI GH200 Nodes

**Node Architecture:**
- **4× NVIDIA GH200 Grace Hopper Superchips** per node
- Each GH200: Arm Grace CPU (120 GB LPDDR5X) + Hopper GPU (96 GB HBM3e)
- **NVLink-C2C**: 900 GB/s coherent CPU-GPU interconnect (7× PCIe Gen 5)
- **Unified Memory Address Space**: CPU + GPU share page tables (460 GB CPU + 384 GB GPU = 844 GB total per node)
- **Usable GPU memory**: ~86.4 GB per GPU (96 GB × 0.90 utilization)
- **Usable CPU memory**: ~100 GB per CPU (115 GB × 0.90)

**Interconnect:**
- **HPE Slingshot 11** fabric with Cassini ASICs
- 4× 200 Gbps NICs per node (one per GH200)
- **Not InfiniBand**: Requires libfabric CXI provider, not IBGDA/NVSHMEM

---

## Core Dependencies (Installed by `slurm-vllm-setup.sh`)

### 1. **NVIDIA HPC SDK 26.3** (CUDA 12.9/13.1)

**What it is:** NVIDIA's compiler toolchain providing CUDA, cuDNN, and NCCL libraries.

**Why it's needed:**
- Provides **CUDA 12.9 forward compatibility** driver support
- Required for compiling vLLM wheels and CUDA kernels
- Includes NCCL (NVIDIA Collective Communications Library) for multi-GPU communication

**Installation path:** `$PROJECTDIR/engine/nvhpc/26.3/`

**Key components:**
- `nvcc`: CUDA compiler
- `libcudart`: CUDA runtime
- `libnccl.so`: NCCL library for GPU collectives
- `libcudnn.so`: cuDNN deep learning primitives

---

### 2. **rdma-core** (Compiled from Source)

**What it is:** Userspace RDMA (Remote Direct Memory Access) core libraries for Linux.

**Why compiled from source:**
- Isambard's SLES (SUSE Linux Enterprise Server) bare-metal **lacks InfiniBand development headers** (`infiniband/verbs.h`)
- Required by UCCL-EP for compilation (even though it uses LIBFABRIC backend)
- Provides `libibverbs` stub libraries and headers

**Installation path:** `$PROJECTDIR/engine/rdma-core/`

**Build flags:**
```bash
cmake -DIN_PLACE=1 -DNO_PROVIDERS=ON -DENABLE_VALGRIND=OFF -DENABLE_LOG_ERRORS=OFF -DNO_MAN_PAGES=ON -DENABLE_PYTHON=OFF ..
```

**Runtime usage:**
- Added to `LD_LIBRARY_PATH` via `vllm-env.sh`
- Enables UCCL-EP to compile with proper header paths

**Sources:**
- [rdma-core GitHub](https://github.com/linux-rdma/rdma-core)
- [Linux Kernel RDMA Verbs Docs](https://docs.kernel.org/infiniband/user_verbs.html)

---

### 3. **vLLM** (Versioned Installation)

**What it is:** High-throughput LLM inference engine with PagedAttention and OpenAI-compatible API.

**Installation:**
```bash
uv pip install vllm=="$version" ray[default] \
  --torch-backend=auto \
  --extra-index-url https://wheels.vllm.ai/$version/cu129
```

**Installation path:** `$PROJECTDIR/engine/vllm/<version>/`

**Key features for Isambard:**
- **PagedAttention**: Efficient KV cache management
- **Continuous batching**: Dynamic request scheduling
- **OpenAI-compatible API**: Drop-in replacement for OpenAI API
- **Distributed serving**: Tensor/pipeline parallelism, expert parallelism
- **Quantization support**: FP8, INT4, AWQ, GPTQ

**GH200-specific patches:**
- H200 MoE configs copied to GH200 (same architecture, similar memory)
- CUDA 12.9 wheels from `wheels.vllm.ai`

**Version management:**
- Multiple versions can coexist in separate directories
- `min-vllm-version` in config ensures compatibility
- Check with: `python -c "import importlib.metadata; print('vllm', importlib.metadata.version('vllm'))"`

---

### 4. **FlashInfer** (Kernel Library)

**What it is:** High-performance kernel library for LLM serving, providing optimized attention and GEMM operations.

**Why it's needed:**
- Provides **FlashInfer JIT cache** for fast kernel compilation
- Alternative attention backend to FlashAttention/Triton
- Required for some model configurations (e.g., Gemma, certain MoE models)

**Installation:**
```bash
uv pip install flashinfer-jit-cache=="$FLASHINFER" --index-url https://flashinfer.ai/whl/cu129
```

**Key features:**
- **Customizable attention engine** for LLM inference
- Supports FP8, FP4, BF16 precisions
- Used by vLLM, SGLang, TensorRT-LLM, MLC-Engine
- **29-69% ITL improvements** reported

**⚠️ GH200/MoE Warning:**
FlashInfer autotune profiling can cause **2+ hour startup delays** on MoE models. For MoE models with `tp >= 2`, disable autotune:

```yaml
gdn-prefill-backend: "triton"
moe-backend: triton
attention-config: '{"backend":"TRITON_ATTN"}'
enable-flashinfer-autotune: false
```

**Sources:**
- [FlashInfer GitHub](https://github.com/flashinfer-ai/flashinfer)
- [FlashInfer Paper (arXiv:2501.01005)](https://arxiv.org/abs/2501.01005)
- [NVIDIA Technical Blog](https://developer.nvidia.com/blog/run-high-performance-llm-inference-kernels-from-nvidia-using-flashinfer/)

---

### 5. **DeepGEMM** (DeepSeek GEMM Library)

**What it is:** Unified high-performance tensor core kernel library from DeepSeek for modern LLM operations.

**Why it's needed:**
- Required for running **DeepSeek models** (V4, R1, etc.)
- Provides optimized **grouped GEMM** for MoE layers
- Better performance than generic PyTorch/CUTLASS for DeepSeek-specific shapes

**Installation:**
```bash
# Git reference extracted from vLLM's install_deepgemm.sh
git clone --recursive https://github.com/deepseek-ai/DeepGEMM.git
uv pip install --no-build-isolation .
```

**Key features:**
- **FP8/FP4/BF16 GEMMs** (general matrix multiplication)
- **Fused MoE** with overlapped communication (Mega MoE)
- **MQA scoring** and other LLM primitives
- Optimized for DeepSeek V3/R1 training and inference

**Usage in vLLM:**
```yaml
# Auto-enabled for DeepSeek models
# Can be forced with:
quantization: fp8
```

**Sources:**
- [DeepGEMM GitHub](https://github.com/deepseek-ai/DeepGEMM)
- [vLLM DeepGEMM Integration](https://docs.vllm.ai/en/stable/api/vllm/utils/deep_gemm/)

---

### 6. **UCCL-EP** (Unified Collective Communication Library - Expert Parallel)

**What it is:** Portable expert-parallel communication library that delivers DeepEP-level performance across heterogeneous GPU and NIC hardware.

**Why it's critical for Isambard:**
- **DeepEP** (NVIDIA's original EP library) requires **GPU-initiated networking** (IBGDA) which only works on NVIDIA InfiniBand NICs
- Isambard-AI uses **HPE Slingshot 11** interconnect with Cassini ASICs — **no GPU-initiated networking support**
- UCCL-EP solves this by implementing a **GPU-initiated, CPU-executed** proxy pattern

**How it works:**
1. GPU warps write 16-byte commands to pinned host memory rings
2. CPU proxy threads monitor rings and dispatch to NIC
3. NIC does direct GPU DMA (GPUDirect RDMA) for data path
4. Keeps DeepEP contract (one-sided write, ordered signal, quiet) but reimplements transport

**Installation:**
```bash
git clone --recursive -b main https://github.com/uccl-project/uccl.git
export TORCH_CUDA_ARCH_LIST="9.0a"
export USE_LIBFABRIC_CXI=1
export USE_DMABUF=1
python3 -m build --wheel --no-isolation
uv pip install --no-build-isolation dist/uccl-*.whl
cd ep && uv pip install --no-build-isolation -vvv ./deep_ep_wrapper
```

**Key components:**
- **P2P layer**: `libuccl_p2p.so` for point-to-point communication
- **EP layer**: Expert parallel kernels with `deep_ep_wrapper` drop-in replacement
- **CXI backend**: HPE Slingshot 11 support via libfabric

**Performance:**
- Comparable to DeepEP on NVIDIA NICs
- Up to **2.1× dispatch/combine throughput** on AWS EFA
- **40% more SGLang token throughput** on NVIDIA+EFA
- **45% more DeepSeek-V3 training throughput** on AMD+Broadcom clusters

**Usage in vLLM:**
```yaml
# Auto-enabled for MoE models requiring expert parallelism
enable-expert-parallel: true
expert-parallel-size: <N>
```

**Sources:**
- [UCCL-EP: Portable Expert-Parallel Communication (arXiv:2512.19849)](https://arxiv.org/abs/2512.19849)
- [UCCL-EP: An expert parallel communications kernel without owning the NIC](https://fergusfinn.com/blog/uccl-ep-without-owning-the-nic/)
- [UCCL Project](https://uccl-project.github.io/)

---

### 7. **NIXL** (NVIDIA Inference Xfer Library)

**What it is:** High-performance KV cache transfer connector for vLLM's **disaggregated prefilling** feature.

**Why it's needed:**
- Enables **disaggregated serving**: separate prefill and decode workers
- Prefill worker computes KV cache, transfers to decode worker via NIXL
- Decode worker uses transferred KV for token generation
- Allows independent scaling of prefill vs decode capacity

**Installation:**
```bash
git clone --recursive https://github.com/ai-dynamo/nixl.git
export UCCL_STAGING_DIR="$workingDir/uccl"
export CPATH="$UCCL_STAGING_DIR/include:$UCCL_STAGING_DIR/p2p:$UCCL_STAGING_DIR/p2p/util:$CPATH"
export LIBRARY_PATH="$UCCL_STAGING_DIR/uccl/lib:$UCCL_STAGING_DIR/p2p:$LIBRARY_PATH"

python3 -m build --wheel --no-isolation \
  -Csetup-args="-Dlibfabric_path=/opt/cray/libfabric/1.22.0" \
  -Csetup-args="-Denable_plugins=LIBFABRIC,UCCL,POSIX" \
  -Csetup-args="-Ddisable_gds_backend=true" \
  -Csetup-args="-Ddisable_mooncake_backend=true" \
  -Csetup-args="-Ddisable_infinia_backend=true" \
  -Csetup-args="-Dbuild_nixl_ep=false"

uv pip install --no-build-isolation dist/nixl*.whl
```

**Key features:**
- Supports multiple transport backends: **LIBFABRIC**, **UCCL**, **POSIX**, CUDA (GPUDirect), Mooncake
- For Isambard: uses **LIBFABRIC** backend (HPE Slingshot 11)
- Uses UCCL's P2P components for cross-process KV cache transfer
- Fully asynchronous send/receive operations

**Usage in vLLM:**
```yaml
kv-transfer-config:
  kv_connector: "NIXLConnector"
  kv_connector_extra_config:
    backends: ["LIBFABRIC"]
```

**Disaggregated prefilling workflow:**
1. Prefill instance processes prompt, generates KV cache
2. NIXL transfers KV cache to decode instance via Slingshot fabric
3. Decode instance continues token generation with transferred KV
4. Enables independent scaling: more decode instances for high-throughput serving

**Sources:**
- [NixlConnector Usage Guide - vLLM Docs](https://docs.vllm.ai/en/stable/features/nixl_connector_usage/)
- [NVIDIA NIXL and Disaggregated Inference](https://www.spheron.network/blog/nvidia-nixl-disaggregated-inference-guide/)
- [AI Dynamo Documentation](https://docs.dynamo.nvidia.com/dynamo/design-docs/disaggregated-serving/)

---

### 8. **humming-kernels** (doublewordAI Fork)

**What it is:** Fork of NVIDIA's humming kernels optimized for GH200/H200 GPUs, maintained by doublewordAI.

**Why the fork:**
- Original humming kernels didn't work out-of-the-box on Isambard
- Required kernel bug fixes and tuning for GH200
- Specifically optimized for **DeepSeek-V4-Flash** on Isambard-AI

**Key fixes:**
- **Group-scaled path**: Fixed per-128 input scales on accumulator
- **BlockM tile heuristic**: Prevents catastrophic register spilling
- **GH200-specific tuning**: Optimized for H100-like architecture

**Installation:**
```bash
uv pip install git+https://github.com/doublewordai/humming.git
```

**Usage in vLLM:**
```yaml
# Auto-selected for supported models
# Can be forced with:
quantization: fp8
moe-backend: humming
```

**Sources:**
- [Throughputmaxxing: DeepSeek-V4-Flash on Isambard-AI](https://fergusfinn.com/blog/throughputmaxxing-v4-flash-single-node/)
- [vLLM MoE Kernel Features](https://docs.vllm.ai/en/latest/design/moe_kernel_features/)

---

### 9. **Tencent HPC-OPS** (High-Performance Computing Operators)

**What it is:** Production operator library from Tencent Hunyuan AI Infra team, optimized for Hopper GPUs (especially H20).

**Why it's needed:**
- Optimized for **H20 GPUs** (similar architecture to GH200)
- Production-proven on Tencent Hunyuan Hy3 models
- **Reduces TTFT by ~24% and TPOT by ~17%** end-to-end

**Two main components:**

#### A. **Attention Backend (`HPC_ATTN`)**

**Optimizations:**
- **Dynamic load-balanced decode scheduler**: Handles mixed-length batches efficiently
- **Fused RoPE + QK-Norm + KV-write prologue**: Eliminates HBM round-trips
- **2.25× average speedup** over FlashInfer/FlashAttention on mixed-length decode

**Key innovation:**
- Replaces fixed split-KV schedule with flat, persistent design
- Tiles KV sequences into 64-token chunks, distributes evenly across CTAs
- Long sequences split across multiple CTAs; short sequences don't monopolize CTAs

#### B. **MoE Backend (`hpc`)**

**Optimizations:**
- **Fully fused FP8 MoE pipeline**: Eliminates intermediate HBM round-trips
- **1.59× faster** at TP8/EP1, **1.21× faster** at TP1/EP8 vs Triton/CUTLASS
- Occupancy-first design without warp specialization

**Installation:**
```bash
git clone https://github.com/Tencent/hpc-ops.git
cd hpc-ops && make wheel
uv pip install --no-build-isolation dist/*.whl
```

**Usage in vLLM:**
```yaml
# For Hy3-series models (currently the only supported models)
attention-backend: HPC_ATTN
moe-backend: hpc
kv-cache-dtype: fp8_e4m3  # for FP8 models
block-size: 64
```

**Performance on H20 (8× GPUs, Hy3 model):**
| Metric | Baseline | HPC-Ops | Improvement |
|--------|----------|---------|-------------|
| TTFT (batch=16) | 7807 ms | 5886 ms | **24.6% faster** |
| TPOT (batch=64) | 31.10 ms | 21.90 ms | **29.6% faster** |

**Sources:**
- [vLLM Blog: HPC-Ops Integration](https://vllm.ai/blog/2026-07-06-vllm-hpc-ops)
- [Tencent HPC-OPS GitHub](https://github.com/Tencent/hpc-ops)
- [Tencent Hy3 Model Recipe](https://recipes.vllm.ai/tencent/Hy3)

---

## Environment Variables (Set by `vllm-env.sh`)

These environment variables are automatically set by the `vllm-env.sh` script sourced before `vllm serve` starts:

```bash
# CPU threading (prevents oversubscription across 4 GPUs)
export OMP_NUM_THREADS=16

# NCCL/Slingshot tuning
export NCCL_NET_GDR_LEVEL=5          # Enforce full GPUDirect RDMA
export FI_PROVIDER="cxi"             # Cray Cassini fabric provider
export FI_CXI_DEFAULT_CQ_SIZE=131072 # Prevent dropped frames
export FI_MR_CACHE_MONITOR=userfaultfd  # Prevent Slingshot memhooks deadlocks
export NCCL_CROSS_NIC=1              # Stripe across 4 NICs per node
export NCCL_MIN_NCHANNELS=4

# CUDA/NCCL stability
export CUDA_DEVICE_MAX_CONNECTIONS=1  # Prevent race conditions in pipeline sync
export NCCL_CUMEM_ENABLE=0            # Prevent virtual memory fragmentation
export NCCL_IB_PCI_RELAXED_ORDERING=1 # Optimize CPU-GPU page migration

# vLLM-specific overrides
export VLLM_SKIP_CUSTOM_ALL_REDUCE=1        # Force standard NCCL (Slingshot stability)
export VLLM_ENGINE_ITERATION_TIMEOUT_S=300  # Prevent multi-node graph timeouts
export VLLM_ALLREDUCE_USE_SYMM_MEM=0        # Disable broken experimental allocator

# Compilation parallelism (prevents 256-core stampede)
export MAX_JOBS=4
export TORCHINDUCTOR_PARALLEL_COMPILE_THREADS=4
export FLASHINFER_NVCC_THREADS=32
```

**Override via config:**
```yaml
env:
  VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS: "1"
  VLLM_USE_DEEP_GEMM_FP8: "1"
```

---

## Package Dependency Graph

```
vLLM (main inference engine)
├── PyTorch (CUDA 12.9 wheels)
│   └── CUDA runtime (from NVHPC SDK 26.3)
├── Ray (distributed execution)
├── FlashInfer (attention kernels)
│   └── JIT cache (pre-compiled for GH200)
├── DeepGEMM (DeepSeek MoE kernels)
│   └── Custom CUDA kernels
├── UCCL-EP (expert parallel on Slingshot)
│   ├── P2P layer (libuccl_p2p.so)
│   ├── EP layer (deep_ep_wrapper)
│   └── rdma-core (compiled headers)
├── NIXL (KV cache transfer)
│   ├── LIBFABRIC backend (Slingshot 11)
│   └── UCCL P2P integration
├── humming-kernels (GH200-optimized MoE)
│   └── Group-scaled MoE kernels
└── HPC-OPS (Tencent attention/MoE backends)
    ├── HPC_ATTN (dynamic load-balanced decode)
    └── hpc (fused FP8 MoE pipeline)
```

---

## Installation Order & Dependencies

```
1. NVHPC SDK 26.3 (CUDA 12.9) ← Foundation
   └── Provides: nvcc, libcudart, libnccl, libcudnn

2. rdma-core (from source) ← Required for UCCL-EP compilation
   └── Provides: infiniband/verbs.h, libibverbs.so

3. vLLM + PyTorch + Ray ← Core inference engine
   └── From: wheels.vllm.ai (CUDA 12.9 wheels)

4. FlashInfer + JIT cache ← Attention kernels
   └── From: flashinfer.ai (CUDA 12.9 wheels)

5. DeepGEMM (from source) ← DeepSeek MoE kernels
   └── Git: deepseek-ai/DeepGEMM

6. UCCL-EP (from source) ← Expert parallel for Slingshot
   └── Git: uccl-project/uccl (main branch)
       └── deep_ep_wrapper (drop-in replacement)

7. NIXL (from source) ← KV cache transfer
   └── Git: ai-dynamo/nixl
       └── Depends on: UCCL-EP, libfabric (Cray CXI)

8. humming-kernels (doublewordAI fork) ← GH200 MoE optimization
   └── Git: doublewordai/humming

9. HPC-OPS (from source) ← Tencent attention/MoE backends
   └── Git: Tencent/hpc-ops
```

---

## When to Use Each Package

### By Model Type

| Model Type | Required Packages | Optional Packages |
|------------|------------------|-------------------|
| **Dense models** (Llama, Qwen dense) | vLLM, FlashInfer | HPC-OPS (for Hy3) |
| **DeepSeek MoE** (V4, R1) | vLLM, DeepGEMM, UCCL-EP | humming-kernels |
| **Qwen MoE** (Qwen3.5, Qwen3-MoE) | vLLM, UCCL-EP | humming-kernels |
| **Gemma MoE** | vLLM, UCCL-EP | - |
| **Hy3-series** (Tencent) | vLLM, HPC-OPS | - |
| **Disaggregated serving** | vLLM, NIXL, UCCL-EP | - |

### By Use Case

| Use Case | Required Packages | Config Flags |
|----------|------------------|--------------|
| **Single-node, dense** | vLLM, FlashInfer | `tensor-parallel-size: 2-4` |
| **Single-node, MoE** | vLLM, UCCL-EP, DeepGEMM | `enable-expert-parallel: true` |
| **Multi-node, MoE** | vLLM, UCCL-EP, DeepGEMM, Ray | `pipeline-parallel-size: N`, `enable-expert-parallel: true` |
| **Disaggregated prefill/decode** | vLLM, NIXL, UCCL-EP | `kv_connector: NIXLConnector` |
| **Maximum throughput (Hy3)** | vLLM, HPC-OPS | `attention-backend: HPC_ATTN`, `moe-backend: hpc` |
| **DeepSeek-V4-Flash** | vLLM, DeepGEMM, humming-kernels, UCCL-EP | `quantization: fp8` |

---

## Troubleshooting Package Issues

### Long Startup Times (>2 hours)

**Symptom:** vLLM takes 2+ hours to start, especially on MoE models.

**Cause:** FlashInfer autotune profiling loop running on every startup.

**Fix:**
```yaml
gdn-prefill-backend: "triton"
moe-backend: triton
attention-config: '{"backend":"TRITON_ATTN"}'
enable-flashinfer-autotune: false
```

### Multi-Node Crash: `Flashinfer allreduce is not supported`

**Symptom:** vLLM crashes immediately on startup with:
```
ValueError: Flashinfer allreduce is not supported for multi-node allreduce with 'trtllm' backend
```

**Cause:** FlashInfer's `trtllm` allreduce backend only supports single-node; `mnnvl` requires NVLink (not available on Slingshot).

**Fix:**
```yaml
compilation-config: '{"pass_config": {"fuse_allreduce_rms": false}}'
```

**Applies to:** ANY multi-node job (`pipeline-parallel-size > 1`), not just MoE models.

### UCCL-EP Compilation Fails

**Symptom:** `infiniband/verbs.h: No such file or directory`

**Cause:** Missing InfiniBand development headers on SLES bare-metal.

**Fix:** Already handled by `slurm-vllm-setup.sh` — compiles rdma-core from source. If still failing:
1. Check `$PROJECTDIR/engine/rdma-core/include/infiniband/verbs.h` exists
2. Verify `LD_LIBRARY_PATH` includes `$PROJECTDIR/engine/rdma-core/lib`
3. Consider using containers instead (has headers pre-installed)

### NIXL Backend Not Found

**Symptom:** `ImportError: No module named 'nixl'` or `NIXLConnector not found`

**Cause:** NIXL not compiled/installed, or wrong Python environment.

**Fix:**
1. Verify NIXL wheel installed: `uv pip show nixl`
2. Check installation path: `$PROJECTDIR/engine/vllm/<version>/lib/python3.12/site-packages/nixl/`
3. Ensure `LD_LIBRARY_PATH` includes UCCL and libfabric paths

### HPC-OPS Only Works on Hy3 Models

**Symptom:** HPC-OPS backends enabled but model fails to load.

**Cause:** HPC-OPS attention backend currently only supports Hy3-series models.

**Fix:**
- Use default backends for non-Hy3 models
- To use HPC-OPS on custom models: replace `rope_norm` with `HpcRopeNorm` in model's forward method (requires code modification)

---

## Performance Benchmarks

### MoE Backends (DeepSeek-V4, TP8/EP1, H20)

| Backend | Latency (µs, batch=64) | Relative Speed |
|---------|------------------------|----------------|
| **HPC-OPS** | 147.2 | **1.0× (baseline)** |
| Triton | 374.9 | 2.55× slower |
| CUTLASS | 330.3 | 2.24× slower |

### Attention Backends (Mixed-Length Decode, H20)

| Scenario | HPC-OPS Dynamic | HPC-OPS Static | FlashInfer | FlashAttention |
|----------|----------------|----------------|------------|----------------|
| 64×4K uniform | 0.033 ms | 0.043 ms | 0.221 ms | 0.095 ms |
| 1×128K + 31×4K skewed | **0.063 ms** | 0.186 ms | 0.220 ms | 0.097 ms |
| **Speedup vs best baseline** | **2.95×** | - | - | - |

### End-to-End (Hy3, 8× H20)

| Metric | Baseline | HPC-OPS | Improvement |
|--------|----------|---------|-------------|
| TTFT (batch=16, 8K input) | 7807 ms | 5886 ms | **24.6% faster** |
| TPOT (batch=64, 4K output) | 31.10 ms | 21.90 ms | **29.6% faster** |

---

## References

- **[Official vLLM Environment Variables (v0.25.1)](https://docs.vllm.ai/en/v0.25.1/configuration/env_vars/)** — Authoritative env var reference
- **[Official vLLM serve CLI](https://docs.vllm.ai/en/stable/cli/serve/)** — Complete CLI option reference
- **[Curated Isambard configs](references/vllm-official-configs.md)** — Verified settings from official docs
- **[vLLM backend selection guide](references/vllm-backend-selection.md)** — **Complete moe-backend, all2all-backend, attention-backend, linear-backend matrix with quantization support**
- **[MoE parallelism strategy](references/moe-parallelism-strategy.md)** — Wide EP vs Hybrid EP decision framework
- [vLLM Documentation](https://docs.vllm.ai/)
- [vLLM Model Recipes](https://recipes.vllm.ai/)
- [Isambard AI Hardware Specs](https://docs.isambard.ac.uk/specs/)
- [UCCL-EP Paper (arXiv:2512.19849)](https://arxiv.org/abs/2512.19849)
- [NIXL Connector Guide](https://docs.vllm.ai/en/stable/features/nixl_connector_usage/)
- [HPC-OPS Blog Post](https://vllm.ai/blog/2026-07-06-vllm-hpc-ops)
- [FlashInfer Paper (arXiv:2501.01005)](https://arxiv.org/abs/2501.01005)
- [DeepGEMM GitHub](https://github.com/deepseek-ai/DeepGEMM)

---

**Last updated:** August 2026  
**Maintained by:** Isambard AI vLLM team