#!/bin/bash

hf_check_usage() {
    echo "Usage: $0 [-m model] [-t token]"
    echo ""
    echo "Options:"
    echo "  -m model    A huggingface model id"
    echo "  -t token    A huggingface toked"
    echo "  -l log      The log file location (optional)"
    echo "  -h          Show this help message"
    exit 1
}

source "./utils.sh"

export HF_MODEL=""
export HF_TOKEN=""
export LOG=""

while getopts "m:t:lh" opt; do
    case $opt in
        m) HF_MODEL="$OPTARG" ;;
        t) HF_TOKEN="$OPTARG" ;;
        l) LOG="$OPTARG" ;;
        h) hf_check_usage ;;
        \?) echo "Error: Invalid option -$OPTARG" >&2; hf_check_usage ;;
        :)  echo "Error: Option -$OPTARG requires an argument" >&2; hf_check_usage ;;
    esac
done

export HF_HOME="$(resolve_model_dir)/hf"
if [[ -z $LOG ]]; then
    LOG=$(resolve_job_log "hf-download")
fi

# redirect output to log
exec > >(tee "$LOG") 2>&1


# Check if it exists in the cache by adding the prefix for the match
if hf cache ls | grep -q "^model/${HF_MODEL}\b"; then
    echo "$HF_MODEL already cached in $HF_HOME."
    exit 0
else
    echo "Dowloading $HF_MODEL to $HF_HOME"

    # srun executes in the foreground, streams output live,
    # and automatically preserves/returns the exit code of the script.
    srun \
        --partition=interactive \
        --reservation=interactive \
        --export=ALL \
        ./slurm-hf-download.sh "$HF_MODEL"

    # Capture the direct exit code of the srun command
    exit_code=$?

    echo "Download job finished with exit code: $exit_code"
    exit "$exit_code"

fi
