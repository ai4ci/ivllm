#!/bin/bash
# shellcheck disable=SC2155,SC1091
# common-env.sh — NVHPC/CUDA/Slingshot environment setup.
#
# Sources by run_head_vllm.sh and run_worker_vllm.sh. Resolves NVHPC root,
# sets CUDA_HOME, CC/CXX compilers, LD_LIBRARY_PATH, NCCL vars, and
# Slingshot 11 fabric tuning. Designed to be sourced multiple times safely.
# Core environment variables that are required for everything to work.
# Sets up CUDA_HOME, CC, CXX, CUDA_VERSION, NVHPC_ROOT.
# These flags should not need to know the vllm version, and are setup before
# vllm install.

[[ -v IVLLM_CUDA ]] && return
export IVLLM_CUDA=
source "$(dirname "${BASH_SOURCE[0]}")/utils.sh"

# ── Module setup ──────────────────────────────────────────────────────
module purge
module load brics/default
module load brics/userenv
module load brics/nccl
module load libfabric

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
# These are probably vllm compile time only flags and maybe have no runtime effect:
export MAX_JOBS=8
export NVCC_THREADS=4

# ── CUDA and NVHPC paths
# ─────────────────────────────────────────────
export CUDA_VERSION=12.9
# The path at which the NVHPC SDK is installed to:

export NVHPC_ROOT=$(resolve_nvhpc_root)
export CUDA_HOME="$NVHPC_ROOT/cuda/$CUDA_VERSION"
export PATH="$CUDA_HOME/bin:$PATH"
export CUDA_PATH="$CUDA_HOME"
export C_INCLUDE_PATH="$CUDA_HOME/include:${C_INCLUDE_PATH:-}"
export CPLUS_INCLUDE_PATH="$CUDA_HOME/include:${CPLUS_INCLUDE_PATH:-}"

# NVHPC separates math library headers (cuBLAS, cuSPARSE) from CUDA SDK headers.
# flashinfer JIT kernels include cublasLt.h which is in math_libs, not cuda/include.
export CPATH="$NVHPC_ROOT/math_libs/$CUDA_VERSION/include:${CPATH:-}"

# ── Triton flags:
# ─────────────────────────────────
export TRITON_PTXAS_PATH="$NVHPC_ROOT/cuda/12.9/bin/ptxas"

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

# ── Network interface selection
# ──────────────────────────────────────
export GLOO_SOCKET_IFNAME=hsn0
export NCCL_SOCKET_IFNAME=hsn
# Force PyTorch's internal TensorPipe layer to follow Gloo to the exact index
export TP_SOCKET_IFNAME=hsn0

# ── UCCL slingshot support
# ──────────────────────────────────────
export RDMA_ROOT=$(resolve_rdma_dir)
export USE_LIBFABRIC_CXI=1
export USE_DMABUF=1
export UCCL_SOCKET_IFNAME=hsn0
export EP_NIC_NAME="cxi0"

# TODO:
# NCCL INFO ENV/Plugin: Could not find: libnccl-env.so
# NCCL INFO PROFILER/Plugin: Could not find: libnccl-profiler.so


# To avoid possible hangs, we suggest setting env variables explicitly including NCCL_IB_GID_INDEX, UCCL_IB_GID_INDEX, NCCL_SOCKET_IFNAME, and UCCL_SOCKET_IFNAME:
# UCCL_IB_GID_INDEX should be the same as NCCL_IB_GID_INDEX like if you were using NCCL.
# UCCL_SOCKET_IFNAME should be the interface that you would use for the --master_addr in torchrun.

# Set up compilation environment variables if local rdma-core was successfully built or already present
# 2. Hardcode the definitive Cray libfabric absolute paths for Isambard AI
LIBFABRIC_INC_DIR="/opt/cray/libfabric/1.22.0/include"
LIBFABRIC_LIB_DIR="/opt/cray/libfabric/1.22.0/lib64"

# 3. Synchronize CPATH so nvcc and g++ grab both rdma-core and fabric headers
export CPATH="$RDMA_ROOT/include:${LIBFABRIC_INC_DIR}:${CPATH:-}"

# 4. Synchronize compiler/linker variables
export CFLAGS="-I$RDMA_ROOT/include -I${LIBFABRIC_INC_DIR} ${CFLAGS:-}"
export CPPFLAGS="-I$RDMA_ROOT/include -I${LIBFABRIC_INC_DIR} ${CPPFLAGS:-}"
export CXXFLAGS="-I$RDMA_ROOT/include -I${LIBFABRIC_INC_DIR} ${CXXFLAGS:-}"
export LDFLAGS="-L$RDMA_ROOT/lib64 -L$RDMA_ROOT/lib -L${LIBFABRIC_LIB_DIR} ${LDFLAGS:-}"
export LD_LIBRARY_PATH="$RDMA_ROOT/lib64:$RDMA_ROOT/lib:${LIBFABRIC_LIB_DIR}:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="$RDMA_ROOT/lib64:$RDMA_ROOT/lib:${LIBFABRIC_LIB_DIR}:${LIBRARY_PATH:-}"

# Library path: brics/nccl libs first, then compat libs, then CUDA, compilers, NCCL, NVSHMEM, math.
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:$RDMA_ROOT/lib:$NVHPC_ROOT/cuda/$CUDA_VERSION/compat:$NVHPC_ROOT/cuda/$CUDA_VERSION/lib64:$NVHPC_ROOT/compilers/lib:$NVHPC_ROOT/comm_libs/$CUDA_VERSION/nccl/lib:$NVHPC_ROOT/comm_libs/$CUDA_VERSION/nvshmem/lib:$NVHPC_ROOT/math_libs/$CUDA_VERSION/lib64"

# vLLM CUDA forward compatibility
export VLLM_ENABLE_CUDA_COMPATIBILITY=1
export VLLM_CUDA_COMPATIBILITY_PATH="$NVHPC_ROOT/cuda/$CUDA_VERSION/compat"

# ── HF Models flags:
# ─────────────────────────────────
# shellcheck disable=2119
export HF_HOME="$(resolve_model_dir)/hf"
