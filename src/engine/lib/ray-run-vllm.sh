#!/bin/bash
# shellcheck disable=1091
# run-ray-vllm.sh — Head ray node vLLM launcher.
# Runs on the head node of a ray cluster.
# apparently (possibly because of HF_HOME not being set) its a good idea to pass
# model by path to vllm
# before we got here all the envinroments this command will be run on have been
# setup. So this is a single vllm command no decisions need to be made.
# ray is in charge of deciding layout.

if [[ -z $SLURM_SUBMIT_DIR ]]; then
    echo "ERROR: no slurm submit directory defined" >&2
    exit 1
fi

IVLLM_JOB=${1:?must set job name}

# slurm srun node scripts are copied to a /var/run directory and executed from there
# so we can;t rely on the script location to find the libraries
# this has almost certainly been run.
source "$SLURM_SUBMIT_DIR/lib/utils.sh"

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


model=$(get_job_config_setting "$IVLLM_JOB" ".model")
# modelPath=$(resolve_model_dir "$model")
serverPort=$(get_job_status_setting "$IVLLM_JOB" ".serverPort")
strippedConfig=$(resolve_stripped_job_config "$IVLLM_JOB")

# make sure signals sent to this script via srun are propagated to vllm:
trap 'kill_pid "$IVLLM_RAY_HEAD_PID" "vllm head"' SIGUSR2 SIGUSR1 SIGTERM

echo "[vllm-serve] model home directory: $HF_HOME"
vllm serve \
    --config "$strippedConfig" \
    --port "${serverPort:-8000}" \
    --served-model-name "$model" "default" "$IVLLM_JOB" \
    --distributed-executor-backend ray \
    &
    IVLLM_RAY_HEAD_PID=$!

sleep 1
echo "[serve-0] initialised ray vllm pid: $IVLLM_RAY_HEAD_PID; jobid: $SLURM_JOB_ID"

wait $IVLLM_RAY_HEAD_PID
