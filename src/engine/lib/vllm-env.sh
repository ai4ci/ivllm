#!/bin/bash
# shellcheck disable=SC2155,SC1091
# vllm-env.sh — vLLM-specific environment variables.
#
# Sets NCCL_CROSS_NIC, FI_PROVIDER, FI_CXI_DISABLE_CONNECTIONS,
# VLLM_ENGINE_ITERATION_TIMEOUT_S, and other tuning variables
# for optimal Slingshot 11 performance on Isambard GH200.
# preamble.sh — Isambard GH200 environment setup for vLLM GPU workloads.
#
# Source this at the top of any SLURM script before starting vLLM. It sets
# all required environment variables for:
#   - NVIDIA HPC SDK (CUDA 12.9 forward compatibility)
#   - NCCL + Slingshot 11 (CXI fabric) tuning
#   - Compiler selection for JIT kernels (gcc-native)
#   - Triton/FlashInfer/DeepGEMM compilation flags
#   - vLLM runtime overrides

source "$(dirname "${BASH_SOURCE[0]}")/utils.sh"

# ── Vllm native flags:
# ─────────────────────────────────
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

# ── vLLM logging config
# ──────────────────────────────────────────────
# N.B. this is needed to ensure that vllm logs time stamps for events so that we can query for inactivity.
export VLLM_LOGGING_CONFIG_PATH="$(dirname "${BASH_SOURCE[0]}")/vllm_logs.json"

# ── Network interface selection
# ──────────────────────────────────────
export GLOO_SOCKET_IFNAME=hsn0
export NCCL_SOCKET_IFNAME=hsn
# Force PyTorch's internal TensorPipe layer to follow Gloo to the exact index
export TP_SOCKET_IFNAME=hsn0

# ── NCCL + Slingshot 11 tuning
# ──────────────────────────────────────
# Force NCCL to map over the Libfabric Cassini driver (Slingshot 11)
export NCCL_NET_GDR_LEVEL=SYS          # Enforce full GPUDirect RDMA across nodes - bypasses the risk of NCCL miscalculating the path over the NVLink-C2C bridge and guarantees that GPUDirect RDMA stays locked on across your nodes
export FI_PROVIDER="cxi"             # Enforce Cray Cassini fabric provider

# Libfabric CXI Buffer Optimisations
export FI_CXI_DEFAULT_CQ_SIZE=131072 # Expand Completion Queue size to prevent dropped frames
export FI_CXI_DEFAULT_TX_SIZE=1024
export FI_CXI_DISABLE_HOST_REGISTER=1
export FI_CXI_RX_MATCH_MODE=software
# This is a specific override of isambard defaults - large language models processing complex multi-node requests can overflow the hardware queue instantly, leading to an application crash or throwing errors like LE resources not recovered during flow control.

# As recommended in UKGovernmentBEIS/isambard_containers
# GDRCopy is not needed with vLLM - vllm bypasses NCCL for intranode comms and between nodes is using RDMA over slingshot.
export NCCL_GDRCOPY_ENABLE=0
export FI_HMEM_CUDA_USE_GDRCOPY=0

# Prevent Slingshot Memory Hooks Deadlocks
# HPE Slingshot uses 'memhooks' by default, which clashes with vLLM memory allocation. Switching to userfaultfd guarantees stable collective communications.
export FI_MR_CACHE_MONITOR=userfaultfd

# Multi-NIC striping (4 Cassini NICs per node, one per GH200)
export NCCL_CROSS_NIC=1
export NCCL_MIN_NCHANNELS=4

# Prevent parallel GPU worker processes from overlapping data transfers,
# causing race conditions or kernel hangs during deep pipeline/tensor syncs.
export CUDA_DEVICE_MAX_CONNECTIONS=1

# Prevent catastrophic virtual memory fragmentation inside the unified space
# N.B. unproven value
export NCCL_CUMEM_ENABLE=0

# Relaxed ordering tells the PCIe root complex that memory pages migrating
# between LPDDR5 CPU memory and HBM3 GPU memory don't need strict serial locks.
# N.B. unproven value
export NCCL_IB_PCI_RELAXED_ORDERING=1

# ── vLLM networking and compilation overrides
# ──────────────────────────────────────────────
export VLLM_FLASHINFER_ALLREDUCE_BACKEND=trtllm  # Force standard NCCL for Slingshot stability
export VLLM_ENGINE_ITERATION_TIMEOUT_S=1800       # Prevent timeouts during multi-node graph setup
export VLLM_ALLREDUCE_USE_SYMM_MEM=0             # Disable broken experimental symmetric memory allocator

# ── DeepEP flags:
# ─────────────────────────────────
# See: https://github.com/deepseek-ai/DeepEP:
# /opt/cray/libfabric/1.22.0/bin/fi_info -p cxi on login.
export EP_NIC_NAME="cxi0"

# ── DeepGEMM flags:
# ─────────────────────────────────
# See: https://github.com/deepseek-ai/DeepGEMM
export DG_JIT_USE_RUNTIME_API=1

# ── FlashInfer flags:
# ─────────────────────────────────
# Limit runtime combinatorics
# These are unvalidated:
export FLASHINFER_HEAD_DIMS="128"
export FLASHINFER_POS_ENCODING_MODES="0"

# ── Triton flags:
# ─────────────────────────────────
export TRITON_PTXAS_PATH="$NVHPC_ROOT/cuda/12.9/bin/ptxas"

# ── Torch inductor flags:
# ─────────────────────────────────
# try and fix inductor autotuning hardcoded paths. We had problems with permissions
# when different users shared the same caches due to autotuning defaulting to a
# hard coded location owned by the original cacher. We now use user level caches
# that are not shared so these settings may not be strictly needed:
export VLLM_ENABLE_INDUCTOR_MAX_AUTOTUNE="0"
export TORCHINDUCTOR_AUTOTUNE_REMOTE_CACHE="0"
export TORCHINDUCTOR_FX_GRAPH_REMOTE_CACHE="0"
export TORCHINDUCTOR_AUTOGRAD_REMOTE_CACHE="0"

# ── Complier caches:
# ─────────────────────────────────
set_jit_caches
export VLLM_COMPILE_CACHE_SAVE_FORMAT="unpacked"

# ── HF Models flags:
# ─────────────────────────────────
export HF_HOME="$(resolve_model_dir)/hf"
export HF_HUB_OFFLINE=1

