#!/bin/bash

source ./utils.sh

IVLLM_JOB=${1?must set job name}
IVLLM_HEAD_NODE_IP=${2?must set head node}

minVllmVersion=$(get_job_config_setting "$IVLLM_JOB" ".min-vllm-version")
vllmVersion=$(select_closest_version "$minVllmVersion")
vllmVersionDir=$(resolve_vllm_version_dir "$vllmVersion")

envExports=$(get_job_config_exports "$IVLLM_JOB")
strippedConfig=$(resolve_stripped_job_config "$IVLLM_JOB")

model=$(get_job_config_setting "$IVLLM_JOB" ".model")
serverPort=$(get_job_status_setting "$IVLLM_JOB" ".serverPort")

source "$vllmVersionDir/bin/activate"
source ./common-env.sh
source ./vllm-env.sh
# Evaluate env blocks in yaml file last to override defaults.
eval "$envExports"

vllm serve \
    --nnodes "$SLURM_NNODES" \
    --node-rank 0 \
    --master-addr "$IVLLM_HEAD_NODE_IP" \
    --config "$strippedConfig" \
    --port "$serverPort" \
    --served-model-name "$model" "default" "$IVLLM_JOB" \
    &
    IVLLM_PID=$!

sleep 1
echo "[serve] initialised head node $IVLLM_HEAD_NODE_IP - vllm pid: $IVLLM_PID; jobid: $SLURM_JOB_ID"
update_status_initialise "$IVLLM_JOB" "$IVLLM_PID"
setup_traps "$IVLLM_JOB"
echo "[serve] initialised job $IVLLM_JOB"

wait $IVLLM_PID
