#!/bin/bash
# shellcheck disable=SC2155,SC1091
# common-env.sh — NVHPC/CUDA/Slingshot environment setup.
#
# Sources by run_head_vllm.sh and run_worker_vllm.sh. Resolves NVHPC root,
# sets CUDA_HOME, CC/CXX compilers, LD_LIBRARY_PATH, NCCL vars, and
# Slingshot 11 fabric tuning. Designed to be sourced multiple times safely.
# cuda.sh - Common settings for installation and runtime of vllm

# Sets up CUDA_HOME, CC, CXX, CUDA_VERSION, NVHPC_ROOT

[[ -v IVLLM_CUDA ]] && return
export IVLLM_CUDA=
source "$(dirname "${BASH_SOURCE[0]}")/utils.sh"

# ── Module setup ──────────────────────────────────────────────────────
module purge
module load brics/nccl

# 13.2 required by DeepEP when compiling against 12.9
module load gcc-native/13.2

# after this LD_LIBRARY_PATH is set to:
# /opt/cray/libfabric/1.22.0/lib64:/opt/cray/libfabric/1.22.0/lib:/tools/brics/apps/linux-sles15-neoverse_v2/gcc-12.3.0/aws-ofi-nccl-1.8.1-c47cd5ivrugm3jzlyqyis4igyflnydmo/lib

# ── GCC and compiler setup
# ─────────────────────────────────────────────
# 2. Find the exact paths to the newly loaded compilers
export CC=$(which gcc)
export CXX=$(which g++)
# Set arm64 compiler optimisation flags
# https://openbenchmarking.org/result/2402098-NE-NVIDIAGH291&sor&sgm=1
export CFLAGS="-mcpu=neoverse-v2 -mtune=neoverse-v2 -O3 ${CFLAGS:-}"
export CXXFLAGS="-mcpu=neoverse-v2 -mtune=neoverse-v2 -O3 ${CXXFLAGS:-}"

# ── Common target architecture flags
# ─────────────────────────────────────────────
export TORCH_CUDA_ARCH_LIST="9.0a"
export NVCC_APPEND_FLAGS="-arch=sm_90a"

# ── Compilation control
# ──────────────────────────────────────
# Prevent torch from over-subscribing CPU cores across parallel workers.
# GH200 has 72 cores; 16 threads/worker is safe for the 4-GPU-per-node case.
export OMP_NUM_THREADS=16
export TORCHINDUCTOR_COMPILE_THREADS=4
export VLLM_USE_PRECOMPILED=1
# These are probably vllm compile time only flags and amybe have no runtime effect:
export MAX_JOBS=8
export NVCC_THREADS=4

# ── CUDA and NVHPC paths
# ─────────────────────────────────────────────
export CUDA_VERSION=12.9
# The path at which the NVHPC SDK is installed to:

export NVHPC_ROOT=$(resolve_nvhpc_root)
export CUDA_HOME="$NVHPC_ROOT/cuda/$CUDA_VERSION"
export PATH="$CUDA_HOME/bin:$PATH"

# NVHPC separates math library headers (cuBLAS, cuSPARSE) from CUDA SDK headers.
# flashinfer JIT kernels include cublasLt.h which is in math_libs, not cuda/include.
export CPATH="$NVHPC_ROOT/math_libs/$CUDA_VERSION/include:${CPATH:-}"

# ── NVSHMEM slingshot support
# ──────────────────────────────────────
# From: https://docs.nvidia.com/nvshmem/archives/nvshmem-260/pdf/NVSHMEM-Release-Notes.pdf
# https://docs.nvidia.com/nvshmem/api/gen/env.html
# (Section moved here so NVSHMEM_DIR is defined *before* CMAKE_PREFIX_PATH,
# LD_LIBRARY_PATH etc. all use it — a v2/v3 ordering bug caught by the
# sandboxed test `common_env_sources` under `set -u`.)
export NVSHMEM_DIR="$NVHPC_ROOT/comm_libs/$CUDA_VERSION/nvshmem"
export CMAKE_PREFIX_PATH="$NVSHMEM_DIR/lib/cmake:${CMAKE_PREFIX_PATH:-}"
export FI_CXI_OPTIMIZED_MRS=false
export NVSHMEM_REMOTE_TRANSPORT="libfabric"
export NVSHMEM_LIBFABRIC_PROVIDER="cxi"
export NVSHMEM_DISABLE_CUDA_VMM=1

# Library path: brics/nccl libs first, then compat libs, then CUDA, compilers, NCCL, NVSHMEM, math.
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:$NVHPC_ROOT/cuda/$CUDA_VERSION/compat:$NVHPC_ROOT/cuda/$CUDA_VERSION/lib64:$NVHPC_ROOT/compilers/lib:$NVHPC_ROOT/comm_libs/$CUDA_VERSION/nccl/lib:$NVHPC_ROOT/comm_libs/$CUDA_VERSION/nvshmem/lib:$NVHPC_ROOT/math_libs/$CUDA_VERSION/lib64"

# vLLM CUDA forward compatibility
export VLLM_ENABLE_CUDA_COMPATIBILITY=1
export VLLM_CUDA_COMPATIBILITY_PATH="$NVHPC_ROOT/cuda/$CUDA_VERSION/compat"


