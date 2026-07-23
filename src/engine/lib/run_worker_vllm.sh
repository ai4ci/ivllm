#!/bin/bash
# run_worker_vllm.sh — Worker node vLLM launcher.
#
# Sources utils.sh and runs the vLLM serve command on worker nodes.
# Only runs on SLURM worker nodes (NODEID > 0).

source "$(dirname "${BASH_SOURCE[0]}")/utils.sh"

IVLLM_JOB=${1?must set job name}
IVLLM_HEAD_NODE_IP=${2?must set head node ip}
IVLLM_NODE_RANK=${3?must set node rank}

minVllmVersion=$(get_job_config_setting "$IVLLM_JOB" ".min-vllm-version")
vllmVersion=$(select_closest_version "$minVllmVersion")
vllmVersionDir=$(resolve_vllm_version_dir "$vllmVersion")

envExports=$(get_job_config_exports "$IVLLM_JOB")
strippedConfig=$(resolve_stripped_job_config "$IVLLM_JOB")

# model=$(get_job_config_setting "$IVLLM_JOB" ".model")
# serverPort=$(get_job_status_setting "$IVLLM_JOB" ".serverPort")

source "$vllmVersionDir/bin/activate"
source "$(dirname "${BASH_SOURCE[0]}")/common-env.sh"
source "$(dirname "${BASH_SOURCE[0]}")/vllm-env.sh"

# Evaluate env blocks in yaml file last to override defaults.
eval "$envExports"

vllm serve \
    --nnodes "$SLURM_NNODES" \
    --node-rank "$IVLLM_NODE_RANK" \
    --master-addr "$IVLLM_HEAD_NODE_IP" \
    --headless \
    --config "$strippedConfig" \
    &
    IVLLM_WORKER_PID=$!

# --port $serverPort \
# --served-model-name "$model" "default" "$IVLLM_JOB" \

echo "[serve] initialised worker $IVLLM_NODE_RANK for $IVLLM_JOB"

wait $IVLLM_WORKER_PID


