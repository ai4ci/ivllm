#!/bin/bash

ivllm-serve_usage() {
    echo "Usage: $0 [-j job] [-t time]"
    echo ""
    echo "This assumes a job directory has been setup and minimally has a "
    echo "vllm.yaml file in it containing the vllm job configuration."
    echo ""
    echo "Options:"
    echo "  -j job      The name of the job to start."
    echo "  -t time     The maximum runtime of this job as HH:MM:SS"
    echo "  -h          Show this help message"
    exit 1
}

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$here/lib/utils.sh"

export IVLLM_JOB=""
export IVLLM_MAXTIME="08:00:00"

while getopts "j:th" opt; do
    case $opt in
        j) IVLLM_JOB="$OPTARG" ;;
        t) IVLLM_MAXTIME="$OPTARG" ;;
        h) ivllm-serve_usage ;;
        \?) echo "Error: Invalid option -$OPTARG" >&2; ivllm-serve_usage ;;
        :)  echo "Error: Option -$OPTARG requires an argument" >&2; ivllm-serve_usage ;;
    esac
done

JOB_FILE=$(resolve_job_status "$IVLLM_JOB")
if [[ ! -f $JOB_FILE ]]; then
    echo "No job file found for $IVLLM_JOB" >&2
    exit 1
fi

export IVLLM_LOG=$(resolve_job_log "$IVLLM_JOB")
minVllmVersion=$(get_job_config_setting "$IVLLM_JOB" ".min-vllm-version")
vllmVersion=$(select_closest_version "$minVllmVersion")

# redirect output to log
exec > >(tee "$IVLLM_LOG") 2>&1

idleTimeout=$(get_job_config_setting "$IVLLM_JOB" ".idle-timeout")
model=$(get_job_config_setting "$IVLLM_JOB" ".model")

# Create the lockfile & fail if cannot get a lockfile.
serverPort=$(create_status_pending "$IVLLM_JOB" "$model" "$idleTimeout") || exit 1

# Check the model requested exists and download if not & fail if model download fails
(source $here/ivllm-get-model.sh -m "$model") || exit 1

envExports=$(get_job_config_exports "$IVLLM_JOB")
strippedConfig=$(resolve_stripped_job_config "$IVLLM_JOB")

dp=$(get_job_config_setting "$IVLLM_JOB" ".data-parallel-size")
tp=$(get_job_config_setting "$IVLLM_JOB" ".tensor-parallel-size")
pp=$(get_job_config_setting "$IVLLM_JOB" ".pipeline-parallel-size")

# Calculate total GPUs using safe defaults (using 1 instead of -1 for neutral multiplication)
nGpus=$((${dp:-1} * ${tp:-1} * ${pp:-1}))

# Calculate nodes, rounding up to ensure a minimum of 1 node is allocated
nNodes=$(( (nGpus + 3) / 4 ))

# GPUS per nodes (multinode jobs grab whole node.)
nGpusPerNode=$(( nNodes > 1 ? 4 : nGpus ))

# GPUS per nodes (multinode jobs grab whole node (as do 4 GPU jobs).)
# Handle memory and exclusive execution flags
if (( nGpusPerNode >= 4 )); then
    memValue="0"
    exclusiveFlag="--exclusive"
else
    memValue="$(( nGpusPerNode * 115 ))G"
    exclusiveFlag=""
fi

maxTime=$(get_max_job_time "$IVLLM_MAXTIME")

echo "=================================="
echo "Starting job $IVLLM_JOB:"
echo "=================================="
echo "Model: $model"
echo "Port: $serverPort"
echo "Nodes: $nNodes"
echo "Log: $IVLLM_LOG"
echo "Gpus per node: $nGpusPerNode"
echo "Memory per node: $memValue"
echo "Max wall time: $maxTime"
echo "Target vllm version: $vllmVersion"
echo "=== Custom environment exports ==="
echo "$envExports"
echo "=== Stripped configuration ======="
cat "$strippedConfig"
echo "=================================="

slurmJobId=$(sbatch \
    --parsable \
    --job-name "$IVLLM_JOB" \
    --nodes=$nNodes \
    --gpus-per-node=$nGpusPerNode \
    --cpus-per-gpu=64 \
    --ntasks-per-node=1 \
    --partition=interactive \
    --reservation=interactive \
    --mem=$memValue ${exclusiveFlag} \
    --time="$maxTime" \
    --output="$IVLLM_LOG" \
    --error="$IVLLM_LOG" \
    $here/lib/slurm-vllm-serve.sh "$IVLLM_JOB" "$nGpusPerNode" "$memValue")

echo "Slurm Job ID: $slurmJobId"
echo "=================================="

update_status_slurm_id "$IVLLM_JOB" "$slurmJobId"

echo "Monitor start up progress:"
echo "Logfile: tail -f $IVLLM_LOG"
echo "Status: cat $(resolve_job_status "$IVLLM_JOB")"
echo "=================================="

