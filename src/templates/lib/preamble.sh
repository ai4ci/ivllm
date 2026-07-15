#!/bin/bash
# shellcheck disable=SC2155
#
# preamble.sh — Isambard GH200 environment setup for vLLM GPU workloads.
#
# Source this at the top of any SLURM script before starting vLLM. It sets
# all required environment variables for:
#   - NVIDIA HPC SDK (CUDA 12.9 forward compatibility)
#   - NCCL + Slingshot 11 (CXI fabric) tuning
#   - Compiler selection for JIT kernels (gcc-native)
#   - Triton/FlashInfer/DeepGEMM compilation flags
#   - vLLM runtime overrides
#
# Required environment variables (set by the SLURM script before sourcing):
#   NVHPC_ROOT     — Path to NVHPC installation (e.g. /projects/X/ivllm/nvhpc/Linux_aarch64/26.3)
#   VLLM_ENGINE_DIR — Root of the vLLM engine directory (for vllm_logs.json etc.)
#
# Optional overrides:
#   VLLM_LOGGING_CONFIG_PATH — Path to vLLM logging config JSON (default: $VLLM_ENGINE_DIR/vllm_logs.json)

# ── Module setup ──────────────────────────────────────────────────────
module purge 2>/dev/null || true
module load brics/nccl gcc-native 2>/dev/null || true

# ── CUDA and NVHPC paths ─────────────────────────────────────────────
export NVHPC_ROOT="${NVHPC_ROOT:?NVHPC_ROOT must be set}"
export CUDA_HOME="$NVHPC_ROOT/cuda/12.9"
export PATH="$CUDA_HOME/bin:$PATH"
export CUDA_VERSION=12.9

# NVHPC separates math library headers (cuBLAS, cuSPARSE) from CUDA SDK headers.
# flashinfer JIT kernels include cublasLt.h which is in math_libs, not cuda/include.
export CPATH="$NVHPC_ROOT/math_libs/12.9/include:${CPATH:-}"

# Library path: compat libs first (for forward compat), then CUDA, compilers, NCCL, NVSHMEM, math
export LD_LIBRARY_PATH="$NVHPC_ROOT/cuda/12.9/compat:$NVHPC_ROOT/cuda/12.9/lib64:$NVHPC_ROOT/compilers/lib:$NVHPC_ROOT/comm_libs/12.9/nccl/lib:$NVHPC_ROOT/comm_libs/12.9/nvshmem/lib:$NVHPC_ROOT/math_libs/12.9/lib64:${LD_LIBRARY_PATH:-}"

# vLLM CUDA forward compatibility
export VLLM_ENABLE_CUDA_COMPATIBILITY=1
export VLLM_CUDA_COMPATIBILITY_PATH="$NVHPC_ROOT/cuda/12.9/compat"

# ── vLLM logging config ──────────────────────────────────────────────
if [ -n "${VLLM_ENGINE_DIR:-}" ]; then
    export VLLM_LOGGING_CONFIG_PATH="${VLLM_LOGGING_CONFIG_PATH:-$VLLM_ENGINE_DIR/vllm_logs.json}"
fi

# ── Network interface selection ──────────────────────────────────────
export GLOO_SOCKET_IFNAME=hsn0
export NCCL_SOCKET_IFNAME=hsn
# Force PyTorch's internal TensorPipe layer to follow Gloo to the exact index
export TP_SOCKET_IFNAME=hsn0

# ── Compiler selection (JIT kernels) ────────────────────────────────
# Use gcc from gcc-native module for JIT compilation (flashinfer, torch.compile).
export CC=gcc
export CXX=g++

# Prevent torch from over-subscribing CPU cores across parallel workers.
# GH200 has 72 cores; 16 threads/worker is safe for the 4-GPU-per-node case.
export OMP_NUM_THREADS=16

# ── NCCL + Slingshot 11 tuning ──────────────────────────────────────
# Force NCCL to map over the Libfabric Cassini driver (Slingshot 11)
export NCCL_NET_GDR_LEVEL=5          # Enforce full GPUDirect RDMA across nodes
export FI_PROVIDER="cxi"             # Enforce Cray Cassini fabric provider
export FI_CXI_DEFAULT_CQ_SIZE=131072 # Expand Completion Queue size to prevent dropped frames

# Libfabric CXI Buffer Optimisations
export FI_CXI_DEFAULT_TX_SIZE=16384
export FI_CXI_DISABLE_HOST_REGISTER=1
export FI_CXI_RX_MATCH_MODE=software

# Prevent Slingshot Memory Hooks Deadlocks
# HPE Slingshot uses 'memhooks' by default, which clashes with vLLM memory allocation.
# Switching to userfaultfd guarantees stable collective communications.
export FI_MR_CACHE_MONITOR=userfaultfd

# Multi-NIC striping (4 Cassini NICs per node, one per GH200)
export NCCL_CROSS_NIC=1
export NCCL_MIN_NCHANNELS=4

# Prevent parallel GPU worker processes from overlapping data transfers,
# causing race conditions or kernel hangs during deep pipeline/tensor syncs.
export CUDA_DEVICE_MAX_CONNECTIONS=1

# Prevent catastrophic virtual memory fragmentation inside the unified space
export NCCL_CUMEM_ENABLE=0

# Relaxed ordering tells the PCIe root complex that memory pages migrating
# between LPDDR5 CPU memory and HBM3 GPU memory don't need strict serial locks.
export NCCL_IB_PCI_RELAXED_ORDERING=1

# ── vLLM networking and compilation overrides ───────────────────────
export VLLM_FLASHINFER_ALLREDUCE_BACKEND=trtllm  # Force standard NCCL for Slingshot stability
export VLLM_ENGINE_ITERATION_TIMEOUT_S=300       # Prevent timeouts during multi-node graph setup
export VLLM_ALLREDUCE_USE_SYMM_MEM=0             # Disable broken experimental symmetric memory allocator

# ── DeepGEMM / Triton compiler flags ─────────────────────────────────
export TRITON_CUDA_ARCH=90
export TRITON_PTXAS_PATH="$NVHPC_ROOT/cuda/12.9/bin/ptxas"
export TORCH_CUDA_ARCH_LIST="9.0a"
export DEEPGEMM_TARGET_ARCH="sm_90"
export NVCC_APPEND_FLAGS="-arch=sm_90a"

# 4. Limit runtime combinatorics
export FLASHINFER_HEAD_DIMS="128"
export FLASHINFER_POS_ENCODING_MODES="0"
