#!/bin/bash
# shellcheck disable=1091
# Setup a ray node for adding to a cluster
# Before the ray head or worker is added to the cluster the following need to be setup
# python venv
# jit cache restore
# common env vars
# vllm specific env vars
# vllm custom env vars from vllm.yaml config
# however nothing to do with the vllm shape at all.


if [[ -z $SLURM_SUBMIT_DIR ]]; then
    echo "ERROR: no slurm submit directory defined" >&2
    exit 1
fi

IVLLM_JOB=${1:?must set job name}
IVLLM_HEAD_NODE_IP=${2:?must set head node ip}
IVLLM_NODE_RANK=${3:-0}
IVLLM_WORKER_NODE_IP=${4:-$IVLLM_HEAD_NODE_IP}

# slurm srun node scripts are copied to a /var/run directory and executed from there
# so we can;t rely on the script location to find the libraries
source "$SLURM_SUBMIT_DIR/lib/utils.sh"

# The cleanup script for a local node.

cleanup() {
    echo "[shutdown-$IVLLM_NODE_RANK] cleaning up local Ray daemon..."
    ray stop --force
    clear_localdir "$IVLLM_JOB"
    exit 0
}

trap cleanup SIGINT SIGTERM

restore_cache "$IVLLM_JOB"
set_jit_caches "$IVLLM_JOB"

minVllmVersion=$(get_job_config_setting "$IVLLM_JOB" ".min-vllm-version")
vllmVersion=$(select_closest_version "$minVllmVersion")
vllmVersionDir=$(resolve_vllm_version_dir "$vllmVersion")

envExports=$(get_job_config_exports "$IVLLM_JOB")

source "$SLURM_SUBMIT_DIR/lib/common-env.sh"
source "$SLURM_SUBMIT_DIR/lib/vllm-env.sh"

# Common to all Ray enviroment variables
# 4Gb ray object store
export RAY_OBJECT_STORE_MEMORY=4294967296
export RAY_LOG_TO_DRIVER=1
export RAY_RUNTIME_ENV_LOG_TO_DRIVER=1
# Change the default temp storage location to a persistent cluster directory
# export RAY_TMPDIR="${ss.paths.remoteJobDir}/ray-logs"

eval "$envExports"
source "$vllmVersionDir/bin/activate"

RAY_ARGS=(
    --node-ip-address="$IVLLM_WORKER_NODE_IP"
)

export VLLM_HOST_IP="$IVLLM_WORKER_NODE_IP"
export RAY_PORT=6379

echo "[serve-$IVLLM_NODE_RANK] setting up ray node $IVLLM_NODE_RANK: $IVLLM_WORKER_NODE_IP"
if [[ "$IVLLM_NODE_RANK" == 0 ]]; then
    echo "this is ray head node: $IVLLM_HEAD_NODE_IP"
    RAY_ARGS+=(
        --head
        --port="$RAY_PORT"
    )

    echo "=== Selected environment exports ==="
    env | grep -E "VLLM_|RAY_|NCCL_|FI_|NVHPC|CUDA_|LD_CONFIG|CPATH|PATH|SLURM_|TRITON|_CACHE_" | sort


else
    RAY_ARGS+=(
        --address="$IVLLM_HEAD_NODE_IP:$RAY_PORT"
    )
fi

echo "==================================="
echo "[serve-$IVLLM_NODE_RANK] executing: ray start --block $(printf '%q ' "${RAY_ARGS[@]}")"
echo "node local job storage: $(resolve_localdir "$IVLLM_JOB")"
env | grep -E "_CACHE_" | sort
echo "==================================="

ray start "${RAY_ARGS[@]}" --block &
RAY_NODE_PID=$!

wait_report "$IVLLM_JOB" "$RAY_NODE_PID" "$IVLLM_NODE_RANK"
