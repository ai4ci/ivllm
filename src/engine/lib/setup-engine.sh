#!/bin/bash

setup_engine_usage() {
    echo "Usage: $0 [-v version] [-l log file]"
    echo ""
    echo "Options:"
    echo "  -v version  The vllm version to install"
    echo "  -f          Force (reinstallation"
    echo "  -l log      The log file location (optional)"
    echo "  -h          Show this help message"
    exit 1
}

source "./utils.sh"

export IVLLM_FORCE=0
export IVLLM_VERSION=""
export LOG=""

while getopts "v:flh" opt; do
    case $opt in
        v) IVLLM_VERSION="$OPTARG" ;;
        f) IVLLM_FORCE=1 ;;
        l) LOG="$OPTARG" ;;
        h) setup_engine_usage ;;
        \?) echo "Error: Invalid option -$OPTARG" >&2; setup_engine_usage ;;
        :)  echo "Error: Option -$OPTARG requires an argument" >&2; setup_engine_usage ;;
    esac
done

if [[ -z $LOG ]]; then
    LOG=$(resolve_job_log "vllm-setup")
fi

# redirect output to log
exec > >(tee "$LOG") 2>&1

vllmVersionDir=$(resolve_vllm_version_dir "$IVLLM_VERSION")

if [[ $IVLLM_FORCE -eq 1 && -d "${vllmVersionDir?directory does not exist}" ]]; then
    rm -rf "$vllmVersionDir"
fi

# Check if version is already installed
if [[ -f "$vllmVersionDir/bin/activate" ]]; then
    echo "$IVLLM_VERSION already installed in $vllmVersionDir use -f flag to force reinstall."
    exit 0
else
    echo "Installing $IVLLM_VERSION to $vllmVersionDir."

    # srun executes in the foreground, streams output live,
    # and automatically preserves/returns the exit code of the script.
    srun \
        --partition=interactive \
        --reservation=interactive \
        --job-name=vllm-setup \
        --nodes=1 \
        --gpus=1 \
        --mem=20G \
        --cpus-per-gpu=64 \
        --ntasks-per-node=1 \
        --time=03:00:00 \
        --export=ALL \
        ./slurm-vllm-setup.sh "$IVLLM_VERSION"

    # Capture the direct exit code of the srun command
    exit_code=$?

    echo "Vllm install finished with exit code: $exit_code"
    exit "$exit_code"

fi
