#!/bin/bash
# shellcheck disable=1091
# run_worker_vllm.sh — Worker node vLLM launcher.
# Runs on a compute node

if [[ -z $SLURM_SUBMIT_DIR ]]; then
    echo "ERROR: no slurm submit directory defined" >&2
    exit 1
fi

IVLLM_JOB=${1:?must set job name}
IVLLM_HEAD_NODE_IP=${2:?must set head node ip}
IVLLM_NODE_RANK=${3:?must set node rank}
IVLLM_WORKER_NODE_IP=${4:?must set worker node ip}

# slurm srun node scripts are copied to a /var/run directory and executed from there
# so we can;t rely on the script location to find the libraries
source "$SLURM_SUBMIT_DIR/lib/utils.sh"

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

# Calculate the starting DP rank index for this specific node
startRank=$(( IVLLM_NODE_RANK * localDp ))

# TODO: node binding...
# Issue detecting nodes when running a 1 or 2 GPU job. In any other job the
# nodes will be full GPU node.
#numaBindNodes="[${CUDA_VISIBLE_DEVICES:?...}]"

IVLLM_ARGS=(
#    --numa-bind-nodes "$numaBindNodes"
    --headless
    --config "$strippedConfig"
    --port "$serverPort"
    --served-model-name "$model" "default" "$IVLLM_JOB"
)

# Scenarios involving multiple physical machines
# n.b. this maybe where --numa-bind lives?
if [ "$nNodes" -gt 1 ]; then
    IVLLM_ARGS+=(
        --nnodes "$nNodes"
        --node-rank "$IVLLM_NODE_RANK"
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
        --data-parallel-start-rank "$startRank"
    )
fi

source "$vllmVersionDir/bin/activate"
source "$SLURM_SUBMIT_DIR/lib/common-env.sh"
source "$SLURM_SUBMIT_DIR/lib/vllm-env.sh"

# Evaluate env blocks in yaml file last to override defaults.
eval "$envExports"

# echo "=== Selected environment exports ==="
# env | grep -E "VLLM_|RAY_|NCCL_|FI_|NVHPC|CUDA_|LD_CONFIG|CPATH|PATH|SLURM_|TRITON" | sort
echo "==================================="
echo "[serve-$IVLLM_NODE_RANK] executing: vllm serve $(printf '%q ' "${IVLLM_ARGS[@]}")"
echo "==================================="


# make sure signals sent to this script via srun are propagated to vllm:
trap 'kill_pid "$IVLLM_WORKER_NODE_PID" "vllm worker $IVLLM_NODE_RANK"' SIGUSR2 SIGUSR1 SIGTERM
export PYTHONUNBUFFERED=1 # remove stdout buffering.
export VLLM_HOST_IP="$IVLLM_WORKER_NODE_IP"
stdbuf -oL -eL vllm serve "${IVLLM_ARGS[@]}" &
    IVLLM_WORKER_NODE_PID=$!

# N.B. careful not to collide naming of IVLLM_WORKER_PID which is the slurm
# step host srun process

# --port $serverPort \
# --served-model-name "$model" "default" "$IVLLM_JOB" \

echo "[serve-$IVLLM_NODE_RANK] initialised worker $IVLLM_NODE_RANK for $IVLLM_JOB"

wait_report "$IVLLM_JOB" "$IVLLM_WORKER_NODE_PID" "$IVLLM_NODE_RANK"

