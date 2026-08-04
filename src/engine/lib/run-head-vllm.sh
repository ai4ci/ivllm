#!/bin/bash
# shellcheck disable=1091
# run_head_vllm.sh — Head node vLLM launcher.
# Runs on a compute node

if [[ -z $SLURM_SUBMIT_DIR ]]; then
    echo "ERROR: no slurm submit directory defined" >&2
    exit 1
fi

# slurm srun node scripts are copied to a /var/run directory and executed from there
# so we can;t rely on the script location to find the libraries
source "$SLURM_SUBMIT_DIR/lib/utils.sh"

IVLLM_JOB=${1?must set job name}
IVLLM_HEAD_NODE_IP=${2?must set head node}

restore_cache "$IVLLM_JOB"
set_jit_caches "$IVLLM_JOB"

minVllmVersion=$(get_job_config_setting "$IVLLM_JOB" ".min-vllm-version")
vllmVersion=$(select_closest_version "$minVllmVersion")
vllmVersionDir=$(resolve_vllm_version_dir "$vllmVersion")

envExports=$(get_job_config_exports "$IVLLM_JOB")
strippedConfig=$(resolve_stripped_job_config "$IVLLM_JOB")

model=$(get_job_config_setting "$IVLLM_JOB" ".model")
serverPort=$(get_job_status_setting "$IVLLM_JOB" ".serverPort")

dp=$(get_job_config_setting "$IVLLM_JOB" ".data-parallel-size")

totalDp=${dp:-1}
nNodes=${SLURM_JOB_NUM_NODES:-1}

# Calculate how many DP ranks exist per node
# (e.g. if totalDp=16 on 16 nodes, localDp=1. If totalDp=64 on 16 nodes, localDp=4)
localDp=$(( totalDp / nNodes ))
if [ "$localDp" -eq 0 ]; then localDp=1; fi

# numaBindNodes="[${CUDA_VISIBLE_DEVICES:?...}]"

IVLLM_ARGS=(
#   --numa-bind-nodes "$numaBindNodes"
    --config "$strippedConfig"
    --port "${serverPort:-8000}"
    --served-model-name "$model" "default" "$IVLLM_JOB"
)

# Scenarios involving multiple physical machines
if [ "$nNodes" -gt 1 ]; then
    IVLLM_ARGS+=(
        --nnodes "$nNodes"
        --node-rank 0
        --master-addr "$IVLLM_HEAD_NODE_IP"
    )
fi

# Scenarios deploying Data Parallelism (including EP)
if [ "$totalDp" -gt 1 ]; then
    # data-parallel-size is passed in config.
    IVLLM_ARGS+=(
        --data-parallel-size-local "$localDp"
        --data-parallel-address "$IVLLM_HEAD_NODE_IP"
        --data-parallel-rpc-port "${IVLLM_DP_RPC_PORT:-13345}"
        --data-parallel-start-rank 0
    )
fi

source "$vllmVersionDir/bin/activate"
source "$SLURM_SUBMIT_DIR/lib/common-env.sh"
source "$SLURM_SUBMIT_DIR/lib/vllm-env.sh"

# Evaluate env blocks in yaml file last to override defaults.
eval "$envExports"

echo "=== Selected environment exports ==="
env | grep -E "VLLM_|RAY_|NCCL_|FI_|NVHPC|CUDA_|LD_CONFIG|CPATH|PATH|SLURM_|TRITON" | sort
echo "==================================="
echo "[serve-0] executing: vllm serve $(printf '%q ' "${IVLLM_ARGS[@]}")"
echo "==================================="

# make sure signals sent to this script via srun are propagated to vllm:
trap 'kill_pid "$IVLLM_PID" "vllm head"' SIGUSR2 SIGUSR1 SIGTERM
export PYTHONUNBUFFERED=1 # remove stdout buffering.
stdbuf -oL -eL vllm serve "${IVLLM_ARGS[@]}" &
    IVLLM_HEAD_NODE_PID=$!

sleep 1
echo "[serve-0] initialised head node $IVLLM_HEAD_NODE_IP - vllm pid: $IVLLM_HEAD_NODE_PID; jobid: $SLURM_JOB_ID"

wait_report "$IVLLM_JOB" "$IVLLM_HEAD_NODE_PID"
