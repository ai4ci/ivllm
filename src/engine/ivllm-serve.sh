#!/bin/bash
# ivllm-serve.sh — Start or reconnect to a vLLM job.
#
# On the login node: submits a SLURM job via sbatch to run the compute-side
# vLLM launcher (slurm-vllm-serve.sh) on a compute node allocation.

ivllm-serve_usage() {
    echo "Usage: $0 [-j job] [-t time]"
    echo ""
    echo "This assumes a job directory has been setup and minimally has a "
    echo "vllm.yaml file in it containing the vllm job configuration."
    echo ""
    echo "Options:"
    echo "  -j job      The name of the job to start."
    echo "  -t time     The maximum runtime of this job as HH:MM:SS"
    echo "  -b          Run as a non-interactive batch"
    echo "  -h          Show this help message"
    echo "Additional flags passed unchanged to sbatch"
    exit 1
}

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$here/lib/utils.sh"

IVLLM_JOB=""
IVLLM_PARTITION="--partition=interactive --reservation=interactive"
IVLLM_MAXTIME="08:00:00"

OPTIND=1
while getopts "j:t:bh" opt; do
    case $opt in
        j) IVLLM_JOB="$OPTARG" ;;
        t) IVLLM_MAXTIME="$OPTARG" ;;
        b) unset IVLLM_PARTITION ;;
        h) ivllm-serve_usage ;;
        \?) echo "Error: Invalid option -$OPTARG" >&2; ivllm-serve_usage ;;
        :)  echo "Error: Option -$OPTARG requires an argument" >&2; ivllm-serve_usage ;;
    esac
done

shift $((OPTIND - 1))

if [[ -z "$IVLLM_JOB" ]]; then
    echo "[serve] ERROR: no job parameter supplied" >&2
    exit 1
fi

JOB_FILE=$(resolve_job_status "$IVLLM_JOB")
if [[ ! -f $JOB_FILE ]]; then
    echo "[serve] no existing job file found for $IVLLM_JOB"
fi

export IVLLM_LOG=$(resolve_job_log "$IVLLM_JOB")
minVllmVersion=$(get_job_config_setting "$IVLLM_JOB" ".min-vllm-version")
vllmVersion=$(select_closest_version "$minVllmVersion")

# redirect output to log
exec > >(tee "$IVLLM_LOG") 2>&1

idleTimeout=$(get_job_config_setting "$IVLLM_JOB" ".idle-timeout")
model=$(get_job_config_setting "$IVLLM_JOB" ".model")

echo "[serve] starting $model with vllm $vllmVersion: idle timeout: ${idleTimeout:-30} mins"

# Check the model requested exists and download if not & fail if model download fails
(source "$here/ivllm-get-model.sh" -m "$model" -l "$IVLLM_LOG") || exit 1

envExports=$(get_job_config_exports "$IVLLM_JOB")
strippedConfig=$(resolve_stripped_job_config "$IVLLM_JOB")

dp=$(get_job_config_setting "$IVLLM_JOB" ".data-parallel-size")
tp=$(get_job_config_setting "$IVLLM_JOB" ".tensor-parallel-size")
pp=$(get_job_config_setting "$IVLLM_JOB" ".pipeline-parallel-size")

echo "[serve] data parallel: ${dp:-1}, tensor parallel: ${tp:-1}, pipeline parallel: ${pp:-1}"

# Calculate total GPUs using safe defaults
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

echo "[serve] $nGpus GPUs; $nNodes nodes; $nGpusPerNode GPUs per node; memory: $memValue"

maxTime=$(get_max_job_time "$IVLLM_MAXTIME")

# Check existing status before starting job.
if is_status "$IVLLM_JOB" "pending"; then
    echo "[serve] ERROR: job $IVLLM_JOB is already submitted and waiting resources" >&2
    exit 1
fi
if is_status "$IVLLM_JOB" "initialising"; then
    echo "[serve] ERROR: job $IVLLM_JOB is already starting up" >&2
    exit 1
fi
if is_status "$IVLLM_JOB" "running"; then
    echo "[serve] ERROR: job $IVLLM_JOB is already running" >&2
    exit 1
fi
if is_status "$IVLLM_JOB" "cancel"; then
    echo "[serve] ERROR: job $IVLLM_JOB is in process of being cancelled" >&2
    exit 1
fi

# Create the lockfile & fail if cannot get a lockfile.
serverPort=$(create_status_pending "$IVLLM_JOB" "$model" "${idleTimeout:-30}") || exit 1

echo ""
echo "=================================="
echo "Starting job $IVLLM_JOB:"
echo "=================================="
echo "  Model: $model"
echo "  Port: $serverPort"
echo "  Nodes: $nNodes"
echo "  Status file: $(resolve_job_status "$IVLLM_JOB")"
echo "  Log: $IVLLM_LOG"
echo "  Gpus per node: $nGpusPerNode"
echo "  Memory per node: $memValue"
echo "  Max wall time: $maxTime"
echo "  Target vllm version: $vllmVersion"
echo "=== Custom environment exports ==="
echo "$envExports" | awk '{print "  " $0}'
echo "=== Stripped configuration ======="
cat "$strippedConfig" | awk '{print "  " $0}'
echo "=================================="

pushd "$here"

#shellcheck disable 2086
slurmJobId=$(sbatch \
    --parsable \
    --job-name "$IVLLM_JOB" \
    --nodes=$nNodes \
    --gpus-per-node=$nGpusPerNode \
    --cpus-per-gpu=64 \
    --ntasks-per-node=1 $IVLLM_PARTITION \
    --mem=$memValue $exclusiveFlag \
    --time="$maxTime" \
    --output="$IVLLM_LOG" \
    --error="$IVLLM_LOG" \
    $@ \
    $here/lib/slurm-vllm-serve.sh "$IVLLM_JOB" "$nGpusPerNode" "$memValue")

echo "  Slurm Job ID: $slurmJobId"
echo "  Submitted from: $here"
echo "=================================="

update_status_slurm_id "$IVLLM_JOB" "$slurmJobId"

echo "  To monitor start up progress:"
echo "  Tail log: ./ivllm-show-log.sh -j $IVLLM_JOB"
echo "  Check status: ./ivllm-status.sh -j $IVLLM_JOB"
echo "=================================="
echo ""

popd
