#!/bin/bash
# slurm-vllm-serve.sh — Multi-node vLLM job launcher.
#
# Submitted by ivllm-serve.sh via sbatch. Resolves environment,
# starts vLLM with proper NCCL/Ray configuration, and runs the
# monitor triad (startup, head, worker).
#SBATCH --export=ALL
#SBATCH --partition=interactive
#SBATCH --reservation=interactive

# shellcheck disable=SC2155,SC1091
# N.b. set the log in the wrapper script (sbatch --output and --error flags)
# and most parameters

source "$(dirname "${BASH_SOURCE[0]}")/utils.sh"

IVLLM_JOB=${1?must supply job name}
IVLLM_GPUS_PER_NODE=${2?must specify gpus per node}
IVLLM_MEM_PER_NODE=${3?must specify mem per node}

export IVLLM_HEAD_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export IVLLM_HEAD_NODE_IP=$(dig +short "$IVLLM_HEAD_NODE")

# launch and background long running process
srun \
    --overlap \
    --nodelist="$IVLLM_HEAD_NODE" \
    --nodes=1 \
    --gpus="$IVLLM_GPUS_PER_NODE" \
    --mem="$IVLLM_MEM_PER_NODE" \
    --cpus-per-gpu=64 \
    --ntasks-per-node=1 \
    --export=ALL \
    ./run_head_vllm.sh \
    "$IVLLM_JOB" \
    "$IVLLM_HEAD_NODE_IP" \
    & IVLLM_PARENT_PID=$!

# Waits for user cancel via lockfile instruction, slurm timeout, or idle timeout:
# This will idle timeout eventually
echo "[serve] initialised head monitor for job $IVLLM_JOB"
monitor_head "$IVLLM_JOB" "$IVLLM_PARENT_PID" & IVLLM_MONITOR_PID=$!

declare -i count=1  # Declare as type integer

IVLLM_WORKER_NODES=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | tail -n+2)
for IVLLM_WORKER in $IVLLM_WORKER_NODES; do

    srun \
        --overlap \
        --nodelist="$IVLLM_WORKER" \
        --nodes=1 \
        --gpus="$IVLLM_GPUS_PER_NODE" \
        --mem="$IVLLM_MEM_PER_NODE" \
        --cpus-per-gpu=64 \
        --ntasks-per-node=1 \
        --export=ALL \
        ./run_worker_vllm.sh \
        "$IVLLM_JOB" \
        "$IVLLM_HEAD_NODE_IP" \
        "$count" \
        & IVLLM_WORKER_PARENT_PID=$!

    echo "[serve] initialised worker monitor for job $IVLLM_JOB, worker $count: $IVLLM_WORKER"
    monitor_worker "$IVLLM_JOB" "$IVLLM_WORKER_PARENT_PID" &
    count+=1          # Increment

done

echo "[serve] initialised startup monitor for job $IVLLM_JOB"
# poll health until api responding
monitor_startup "$IVLLM_JOB"

wait $IVLLM_PARENT_PID
wait $IVLLM_MONITOR_PID

