#!/bin/bash
# ivllm-setup.sh — Install or reinstall vLLM on the HPC.
#
# Submits an srun job to install the specified vLLM version into the
# shared project directory. Idempotent: if the version already exists,
# skips installation unless -f (force) is specified.

ivllm_setup_usage() {
    # Print usage instructions and exit with error code 1.
    echo ""
    echo "Options:"
    echo "  -v version  The vllm version to install"
    echo "  -f          Force (reinstallation)"
    echo "  -r          Retry failed installation"
    echo "  -l log      The log file location (optional)"
    echo "  -h          Show this help message"
    exit 1
}

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$here/lib/utils.sh"

export IVLLM_FORCE=0
export IVLLM_RETRY=0
export IVLLM_VERSION=""
export LOG=""

OPTIND=1
while getopts "v:frlh" opt; do
    case $opt in
        v) IVLLM_VERSION="$OPTARG" ;;
        f) IVLLM_FORCE=1 ;;
        r) IVLLM_RETRY=1 ;;
        l) LOG="$OPTARG" ;;
        h) ivllm_setup_usage ;;
        \?) echo "Error: Invalid option -$OPTARG" >&2; ivllm_setup_usage ;;
        :)  echo "Error: Option -$OPTARG requires an argument" >&2; ivllm_setup_usage ;;
    esac
done

if [[ -z $LOG ]]; then
    LOG=$(resolve_job_log "vllm-setup")
fi

if [[ -z "$IVLLM_VERSION" ]]; then
    echo "ERROR: No version supplied" >&2
    exit 1
fi

echo "logging to: $LOG"
# redirect output to log
exec > >(tee "$LOG") 2>&1

vllmVersionDir=$(resolve_vllm_version_dir "$IVLLM_VERSION")

echo "installing $IVLLM_VERSION to $vllmVersionDir"

if [[ $IVLLM_FORCE -eq 1 && -d "$vllmVersionDir" ]]; then
    echo "removing existing install: $vllmVersionDir"
    rm -rf "$vllmVersionDir"
fi

# Check if version is already installed
if [[ $IVLLM_RETRY -eq 0 && -f "$vllmVersionDir/bin/activate" ]]; then
    echo "$IVLLM_VERSION already installed in $vllmVersionDir use -f flag to force reinstall."
    exit 0
else
    echo "Installing $IVLLM_VERSION to $vllmVersionDir"

    # srun executes in the foreground, streams output live,
    # and automatically preserves/returns the exit code of the script.
    # --partition=interactive \

    srun \
        --partition=interactive \
        --reservation=interactive \
        --job-name=vllm-setup \
        --nodes=1 \
        --gpus=1 \
        --mem=115G \
        --cpus-per-gpu=64 \
        --ntasks-per-node=1 \
        --time=03:00:00 \
        --export=ALL \
        "$here/lib/slurm-vllm-setup.sh" "$IVLLM_VERSION"

    # Capture the direct exit code of the srun command
    exit_code=$?

    echo "Vllm install finished with exit code: $exit_code"
    exit "$exit_code"

fi
