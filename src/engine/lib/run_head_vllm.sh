#!/bin/bash
# run_head_vllm.sh — Head node vLLM launcher.
#
# Sources utils.sh, resolves the environment, and runs the vLLM serve
# command with proper logging. Only runs on SLURM head node (NODEID=0).

source "$(dirname "${BASH_SOURCE[0]}")/utils.sh"

IVLLM_JOB=${1?must set job name}
IVLLM_HEAD_NODE_IP=${2?must set head node}

minVllmVersion=$(get_job_config_setting "$IVLLM_JOB" ".min-vllm-version")
vllmVersion=$(select_closest_version "$minVllmVersion")
vllmVersionDir=$(resolve_vllm_version_dir "$vllmVersion")

envExports=$(get_job_config_exports "$IVLLM_JOB")
strippedConfig=$(resolve_stripped_job_config "$IVLLM_JOB")

model=$(get_job_config_setting "$IVLLM_JOB" ".model")
serverPort=$(get_job_status_setting "$IVLLM_JOB" ".serverPort")

dp=$(get_job_config_setting "$IVLLM_JOB" ".data-parallel-size")

totalDp=${dp:-1}
nNodes=${SLURM_NNODES:-1}

# Calculate how many DP ranks exist per node
# (e.g. if totalDp=16 on 16 nodes, localDp=1. If totalDp=64 on 16 nodes, localDp=4)
localDp=$(( totalDp / nNodes ))
if [ "$localDp" -eq 0 ]; then localDp=1; fi

IVLLM_ARGS=(
    --numa-bind
    --config "$strippedConfig"
    --port "$serverPort"
    --served-model-name "$model" "default" "$IVLLM_JOB"
)

# Scenarios involving multiple physical machines
if [ "$nNodes" -gt 1 ]; then
    IVLLM_ARGS+=(
        --nnodes "$nNodes"
        --node-rank "$nodeRank"
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
source "$(dirname "${BASH_SOURCE[0]}")/common-env.sh"
source "$(dirname "${BASH_SOURCE[0]}")/vllm-env.sh"

echo "[serve-0] executing: vllm serve $(printf '%q ' "${IVLLM_ARGS[@]}")"

# Evaluate env blocks in yaml file last to override defaults.
eval "$envExports"

vllm serve "${IVLLM_ARGS[@]}" &
    IVLLM_PID=$!

sleep 1
echo "[serve-0] initialised head node $IVLLM_HEAD_NODE_IP - vllm pid: $IVLLM_PID; jobid: $SLURM_JOB_ID"
update_status_initialise "$IVLLM_JOB" "$IVLLM_PID"
setup_traps "$IVLLM_JOB"
echo "[serve-0] initialised job $IVLLM_JOB"

wait $IVLLM_PID
