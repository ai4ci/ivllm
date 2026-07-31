# Multi-Node vLLM Deployment on Isambard AI (GH200): Learnings & Best Practices

This document compiles the core technical findings, pitfalls, and solutions discovered during the deployment and tuning of multi-node Mixture of Experts (MoE) models (such as Qwen 3.5 397B and DeepSeek-V4-Pro) on **Isambard AI (NVIDIA GH200 Grace Hopper Superchip cluster)**. These learnings can be fed back to the Isambard HPC support team to help other users succeed with large-scale distributed inference.

---

## Executive Summary

Deploying extremely large models requires multi-node Tensor + Expert Parallel (TEP) or Pipeline Parallel configurations via **Ray** and **vLLM**. While single-node vLLM works relatively out-of-the-box, scaling to multiple nodes on a specialized Grace Hopper architecture connected via HPE Slingshot introduces unique challenges. 

Our key breakthroughs focused on **CUDA 12.9 forward compatibility on a CUDA 12.7 driver, JIT compiler cache isolation on Lustre, bypassing CUDA IPC custom all-reduce over Slingshot, and ensuring robust environment propagation to Ray worker actors.**

---

## 1. CUDA Version Mismatch and Forward Compatibility

### The Pitfall: GPU Driver Limits vs. vLLM CUDA Requirements
The NVIDIA GH200 GPUs on Isambard AI run driver **565.57.01**, which natively supports **CUDA 12.7** at most (as shown by `nvidia-smi`). However, recent vLLM releases (0.19.x+) require **CUDA 12.9+** — both for runtime libraries and for JIT kernel compilation during model warmup.

Attempting to install and run vLLM with the system CUDA produces:
- **Build-time failures**: `fastsafetensors` C++ extensions fail to compile against system GCC
- **Runtime failures**: CUDA library version mismatches (`libnvJitLink`, `libcuda`) even with the system forward-compat package
- **JIT compilation failures**: flashinfer, DeepGEMM, and Triton kernels require `nvcc` and C++20 support unavailable in the system toolchain

### The Solution: NVIDIA HPC SDK 26.3 Bare-Metal Forward Compatibility
We use the **NVIDIA HPC SDK 26.3** (aarch64 tarball, ~3 GB) installed once to shared project space. The SDK bundles its own CUDA 12.9 toolkit and forward-compatibility libraries. By placing the compat path first in `LD_LIBRARY_PATH`, the HPC SDK's `libcuda_compat` intercepts CUDA API calls and translates them for the installed driver — enabling CUDA 12.9 applications to run on the 12.7 driver.

Additionally, we install vLLM using **`cu129` wheels** from vLLM's extra index (`https://wheels.vllm.ai/<version>/cu129`), which are compiled against CUDA 12.9 and match the HPC SDK environment.

The full preamble required in every SLURM script (setup and inference) is:
```bash
# Load host compiler (required for flashinfer JIT and fastsafetensors)
module load brics/nccl gcc-native/14.2

# NVIDIA HPC SDK 26.3 — CUDA 12.9 forward compatibility
export NVHPC_ROOT=/projects/<your-project>/ivllm/nvhpc/Linux_aarch64/26.3
export CUDA_HOME=$NVHPC_ROOT/cuda/12.9
export PATH=$CUDA_HOME/bin:$PATH
export CPATH=$NVHPC_ROOT/math_libs/12.9/include:$NVHPC_ROOT/comm_libs/12.9/nccl/include:${CPATH:-}
export LD_LIBRARY_PATH=$NVHPC_ROOT/cuda/12.9/compat:$NVHPC_ROOT/cuda/12.9/lib64:$NVHPC_ROOT/compilers/lib:$NVHPC_ROOT/comm_libs/12.9/nccl/lib:$NVHPC_ROOT/comm_libs/12.9/nvshmem/lib:$NVHPC_ROOT/math_libs/12.9/lib64:${LD_LIBRARY_PATH:-}
export CC=gcc
export CXX=g++
```

Key points:
- **`LD_LIBRARY_PATH` ordering is critical** — the `compat` directory must appear *before* any system CUDA path so `libcuda.so` from the compat shim shadows the system stub.
- **`gcc-native/14.2` is required** — the system GCC is too old for C++20, which flashinfer JIT kernels require. Without it: `nvcc warning: -std=c++20 not supported with host compiler`.
- **`CPATH` must include `math_libs/12.9/include`** — NVHPC stores math library headers (cuBLAS, cuSPARSE, cublasLt) separately from the CUDA SDK headers. Without it: `cublasLt.h: No such file or directory`.
- **`PATH` must include `$CUDA_HOME/bin`** — provides `nvcc` for JIT compilation. Ray actors strip most environment variables, so this must be set in the `bash -c` wrapper that starts Ray workers.
- **`brics/nccl` is loaded alongside HPC SDK** — provides the Slingshot-optimised NCCL build required for multi-node communication.
- The venv activation follows this preamble: `source /projects/<your-project>/ivllm/<vllm-version>/bin/activate`

### Installation
The HPC SDK is installed once per project via a GPU SLURM job (the `ivllm setup` command automates this):
```bash
wget https://developer.download.nvidia.com/hpc-sdk/26.3/nvhpc_2026_263_Linux_aarch64_cuda_multi.tar.gz
NVHPC_SILENT=true NVHPC_INSTALL_DIR=/projects/<your-project>/ivllm/nvhpc NVHPC_INSTALL_TYPE=single ./install
```
Note: the **`aarch64`** tarball is required (Isambard AI uses ARM CPUs).

### Alternative: Singularity/Apptainer Containers
Singularity with NGC container images (`nvcr.io/nvidia/pytorch:<tag>`) is also viable and handles CUDA forward compatibility automatically via the NVIDIA container runtime (`--nv` flag). However, multi-node Ray inside containers requires the unproven `source /host/adapt.sh` pattern inside every `srun` task, and images are 10–15 GB. The bare-metal HPC SDK approach was preferred for multi-node stability and smaller footprint.

---

## 2. JIT Compiler Cache Races on NFS vs. Lustre

### The Pitfall: NFS file-locking (`ESTALE`) and directory rename races
vLLM utilizes multiple JIT-compilation engines (FlashInfer, DeepGEMM, OpenAI Triton, and PyTorch TorchInductor) to compile custom GPU kernels during model startup/warmup. 
- By default, these engines write their compiled caches to the user's NFS home directory (`~/.cache/flashinfer/`, `~/.deep_gemm/`, `~/.triton/`, `~/.cache/torchinductor/`).
- **NFS does not support reliable `fcntl.flock()`**, causing FlashInfer compilation lock attempts to fail with `ESTALE` (errno 116).
- On highly concurrent multi-node ranks, concurrent compilation threads write and atomically rename cache directories. On NFS, metadata caching propagation delays lead to ranks attempting to load partially compiled or non-existent cubin files, crashing with:
  ```
  RuntimeError: Assertion error (.../compiler.hpp:147): runtime != nullptr
  ```

### The Solution: Model-Scoped Symlinks to Lustre (`$SCRATCHDIR`)
Redirecting the caches via environment variables (e.g. `FLASHINFER_JIT_CACHE_DIR`, `DG_JIT_CACHE_DIR`) is **insufficient** because **vLLM's Ray worker agent launcher (`ray_env.py`) strips non-prefixed environment variables**, meaning spawned worker actors revert to default NFS paths.

We resolved this by creating **model-scoped** cache directories under `$SCRATCHDIR` and symlinking them in the SLURM preamble on all nodes before any Ray actor starts:
```bash
# Derive a model-specific slug (e.g. Qwen/Qwen3.5-397B → Qwen_Qwen3_5-397B)
MODEL_SLUG=$(echo "$VLLM_MODEL" | tr '/' '_' | tr '.' '_')

# FlashInfer JIT Cache
export FLASHINFER_JIT_CACHE_DIR=$SCRATCHDIR/flashinfer_cache_${MODEL_SLUG}
mkdir -p "$FLASHINFER_JIT_CACHE_DIR" ~/.cache
ln -sfn "$FLASHINFER_JIT_CACHE_DIR" ~/.cache/flashinfer

# DeepGEMM JIT Cache
export DG_JIT_CACHE_DIR=$SCRATCHDIR/deep_gemm_cache_${MODEL_SLUG}
mkdir -p "$DG_JIT_CACHE_DIR"
ln -sfn "$DG_JIT_CACHE_DIR" ~/.deep_gemm

# Triton JIT Cache
export TRITON_CACHE_DIR=$SCRATCHDIR/triton_cache_${MODEL_SLUG}
mkdir -p "$TRITON_CACHE_DIR"
ln -sfn "$TRITON_CACHE_DIR" ~/.triton

# TorchInductor JIT Cache
export TORCHINDUCTOR_CACHE_DIR=$SCRATCHDIR/torchinductor_cache_${MODEL_SLUG}
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" ~/.cache
ln -sfn "$TORCHINDUCTOR_CACHE_DIR" ~/.cache/torchinductor
```
**Benefits:** Lustre scratch space supports POSIX flock, is lightning fast, eliminates NFS metadata races, and caches persist across runs (~60-day retention). Using model-scoped directory names prevents kernel cache pollution when switching between different models (e.g. Gemma 4 vs. Qwen 3.5) — each model gets its own isolated cache with its own compiled kernels, avoiding bloat and potential stale-entry conflicts.

---

## 3. Custom All-Reduce CUDA Errors Over Slingshot

### The Pitfall: P2P IPC Handshaking Failures
vLLM utilizes its own custom all-reduce kernel for intra-node Tensor Parallel communication to bypass NCCL overhead. However:
- This custom all-reduce relies on CUDA IPC / symmetric memory handles which require peer-to-peer (P2P) GPU memory access.
- Across distinct compute nodes connected via HPE Slingshot, P2P memory access is physically unavailable.
- If vLLM attempts to use custom all-reduce in a multi-node topology, the communication handshake fails and crashes with:
  ```
  Failed: Cuda error /workspace/csrc/custom_all_reduce.cuh:434 'invalid argument'
  ```

### The Solution: Disable Custom All-Reduce in Multi-Node Preamble
We force vLLM to use standard, fully supported NCCL collectives across the network by exporting the bypass variable:
```bash
export VLLM_SKIP_CUSTOM_ALL_REDUCE=1
```
This, combined with NCCL tuning parameters for Slingshot:
```bash
export VLLM_ALLREDUCE_USE_SYMM_MEM=0
export NCCL_CROSS_NIC=1
export NCCL_FORCE_FLUSH=0
```
ensures robust, error-free multi-node distributed communication.

---

## 4. Environment Variable & Path Stripping on Workers

### The Pitfall: Binary and venv path mismatch
When starting worker processes via remote `srun` (such as Ray cluster initialization), the remote processes run in non-login environments and do not inherit login-node settings or binary paths. This leads to `ray: command not found` or `nvcc: command not found` errors, blocking kernel compilation.

### The Solution: Standardize Non-Login Wrapper Invocation
All Ray head and worker startup commands must be wrapped in `bash -c` and explicitly source the versioned virtual environment:
```bash
srun --nodelist "$WORKER" --nodes=1 --gpus=4 --mem=0 --cpus-per-task 72 --ntasks-per-node 1 \
  bash -c "source /projects/b6ax/ivllm/0.22.0/bin/activate && \
  VLLM_HOST_IP=$WORKER_IP ray start --block --address=$HEAD_NODE_IP:$RAY_PORT --node-ip-address=$WORKER_IP --object-store-memory=$RAY_OBJECT_STORE_MEMORY" &
```

---

## 5. Multi-User Directory Permissions & HF Offline Mode

### The Pitfall: Blocked Model Cache access and HF 429 Errors
In shared research team directory structures (e.g. `$PROJECTDIR/` / `/projects/` folder):
1. **umask 0022:** Standard umask creates directories and downloaded HuggingFace model cache weights with `drwxr-xr-x` permissions. A second user in the same project group trying to access or write to this cache hits `Permission denied`.
2. **HF API 429s:** Concurrent ranks query the HF hub to verify weight freshness even if the model is fully cached, triggering API rate-limiting blocks.

### The Solution: Enforce Group-Writable Umask and Offline Mode
- **umask 0002:** Prepend `umask 0002` to the login pre-downloads, bare-metal setups, and SLURM inference headers, combined with an explicit `chmod g+w` on the root model cache directories.
- **HF_HUB_OFFLINE=1:** Always download the model fully on a LOGIN node prior to job submission, then launch the inference SLURM script with:
  ```bash
  export HF_HUB_OFFLINE=1
  ```
  This completely prevents API calls at inference startup, saving several minutes and avoiding rate limits.

---

## 6. Memory Pressure Mitigation

### Ray Object Store Overhead
By default, Ray automatically configures its object store size relative to the total node RAM. On Isambard nodes with large host RAM, this consumes a massive footprint, leaving insufficient room for vLLM core engines.
- **Solution:** Cap Ray's object store memory to `RAY_OBJECT_STORE_MEMORY=64GB` (or less) during `ray start --object-store-memory=...`.

---

## Summary of Optimal SLURM Preamble for Isambard AI

Below is the definitive environment configuration block recommended for any multi-node vLLM deployment on Isambard AI Phase 2:

```bash
#!/bin/bash
#SBATCH --nodes=2
#SBATCH --gpus-per-node=4
#SBATCH --mem=0
#SBATCH --exclusive

# Enforce group-write capabilities for shared weights/caches
umask 0002

# Load host compiler (required for flashinfer JIT kernels and fastsafetensors)
# and Slingshot-optimised NCCL
module load brics/nccl gcc-native/14.2

# NVIDIA HPC SDK 26.3 — CUDA 12.9 forward compatibility on driver 565.57.01 (max 12.7)
export NVHPC_ROOT=/projects/<your-project>/ivllm/nvhpc/Linux_aarch64/26.3
export CUDA_HOME=$NVHPC_ROOT/cuda/12.9
export PATH=$CUDA_HOME/bin:$PATH
export CPATH=$NVHPC_ROOT/math_libs/12.9/include:$NVHPC_ROOT/comm_libs/12.9/nccl/include:${CPATH:-}
# compat must be FIRST to shadow system libcuda.so
export LD_LIBRARY_PATH=$NVHPC_ROOT/cuda/12.9/compat:$NVHPC_ROOT/cuda/12.9/lib64:$NVHPC_ROOT/compilers/lib:$NVHPC_ROOT/comm_libs/12.9/nccl/lib:$NVHPC_ROOT/comm_libs/12.9/nvshmem/lib:$NVHPC_ROOT/math_libs/12.9/lib64:${LD_LIBRARY_PATH:-}
export CC=gcc
export CXX=g++

# Activate the versioned vLLM environment (cu129 wheels)
source /projects/<your-project>/ivllm/<vllm-version>/bin/activate

# Symlink all major JIT engines to Lustre scratch to bypass NFS locking/metadata races.
# Model-scoped cache dirs prevent kernel pollution when switching between models.
# $SCRATCHDIR is always defined on Isambard AI and persists ~60 days.
MODEL_SLUG=$(echo "$VLLM_MODEL" | tr '/' '_' | tr '.' '_')

export FLASHINFER_JIT_CACHE_DIR=$SCRATCHDIR/flashinfer_cache_${MODEL_SLUG}
mkdir -p "$FLASHINFER_JIT_CACHE_DIR" ~/.cache
ln -sfn "$FLASHINFER_JIT_CACHE_DIR" ~/.cache/flashinfer

export DG_JIT_CACHE_DIR=$SCRATCHDIR/deep_gemm_cache_${MODEL_SLUG}
mkdir -p "$DG_JIT_CACHE_DIR"
ln -sfn "$DG_JIT_CACHE_DIR" ~/.deep_gemm

export TRITON_CACHE_DIR=$SCRATCHDIR/triton_cache_${MODEL_SLUG}
mkdir -p "$TRITON_CACHE_DIR"
ln -sfn "$TRITON_CACHE_DIR" ~/.triton

export TORCHINDUCTOR_CACHE_DIR=$SCRATCHDIR/torchinductor_cache_${MODEL_SLUG}
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" ~/.cache
ln -sfn "$TORCHINDUCTOR_CACHE_DIR" ~/.cache/torchinductor

# Network Tuning for Multi-Node over HPE Slingshot
export VLLM_ALLREDUCE_USE_SYMM_MEM=0
export NCCL_CROSS_NIC=1
export NCCL_FORCE_FLUSH=0
export VLLM_SKIP_CUSTOM_ALL_REDUCE=1 # Crucial to avoid 'invalid argument' CUDA IPC crashes

# Prevent HF API rate-limits during multi-gpu concurrent launch
export HF_HOME=/projects/<your-project>/hf
export HF_HUB_OFFLINE=1
```
