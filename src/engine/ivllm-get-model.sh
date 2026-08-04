#!/bin/bash
# shellcheck disable=SC2155,1091
# ivllm-get-model.sh — Download or verify a model in the HuggingFace cache.
#
# Checks the shared HF cache and downloads if needed. Used by ivllm-serve.sh.

ivllm-get-model_usage() {
    echo "Usage: $0 [-m model] [-t token]"
    echo ""
    echo "Options:"
    echo "  -m model    A huggingface model id"
    echo "  -t token    A huggingface token"
    echo "  -l log      The log file location (optional)"
    echo "  -h          Show this help message"
    exit 1
}

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$here/lib/utils.sh"

export HF_MODEL=""
export LOG=""

OPTIND=1
while getopts "m:t:l:h" opt; do
    case $opt in
        m) HF_MODEL="$OPTARG" ;;
        t) export HF_TOKEN="$OPTARG" ;;
        l) LOG="$OPTARG" ;;
        h) ivllm-get-model_usage ;;
        \?) echo "Error: Invalid option -$OPTARG" >&2; ivllm-get-model_usage ;;
        :)  echo "Error: Option -$OPTARG requires an argument" >&2; ivllm-get-model_usage ;;
    esac
done

# shellcheck disable=2119
modelDir=$(resolve_model_dir)
export HF_HOME="$modelDir/hf"
if [[ -z $LOG ]]; then
    LOG=$(resolve_job_log "hf-download")
fi

if [[ -z "$HF_MODEL" ]]; then
    echo "[model] ERROR: No model supplied" >&2
    exit 1
fi

# redirect output to log
exec > >(tee "$LOG") 2>&1

echo "[model] checking for model: $HF_MODEL"

hfVenv="$modelDir/venv"

# Check if the virtual environment binary exists instead of checking the command path
if [ ! -f "$hfVenv/bin/hf" ]; then

  # Install uv if not already present
  if ! command -v uv &>/dev/null; then
    echo "[model] installing uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
  fi

  echo "[model] installing hf"
  # grab a local working directory for the "hf-download" job:
  workingDir=$(resolve_localdir "hf-download")
  echo "[model] working directory: $workingDir"

  export UV_CACHE_DIR=$workingDir/uv_cache
  uv venv "$hfVenv" --python 3.12
  source "$hfVenv/bin/activate"
  uv pip install huggingface_hub[cli]
  echo "[model] installed hf cli"

else
  echo "[model] hf virtual environment found. Activating..."
  source "$hfVenv/bin/activate"
fi

# Ensure hf is available in the current environment context
if ! command -v hf &>/dev/null; then
  echo "[model] ERROR: hf command not found even after activation." >&2
  exit 1
fi


# Check if it exists in the cache by adding the prefix for the match
if hf cache ls --format json | jq -e ".[] | select(.repo_id == \"$HF_MODEL\")" > /dev/null; then
    echo "[model] $HF_MODEL already cached in $HF_HOME."
    exit 0
else
    if [[ -z "$HF_TOKEN" ]]; then
        echo "[model] ERROR: No token parameter supplied or HF_TOKEN env var set" >&2
        exit 1
    fi
    echo "[model] downloading $HF_MODEL to $HF_HOME"

    # srun executes in the foreground, streams output live, which will end
    # up in the log due to the tee command above.
    # and automatically preserves/returns the exit code of the script.
    # HF_HOME and HF_TOKEN wil be resolved.
    # --partition=interactive \ seems unnecessary?

    srun \
        --reservation=interactive \
        --cpus-per-task=4 \
        --ntasks=1 \
        --mem=16G \
        --export=ALL \
        hf download "$HF_MODEL"

    # Capture the direct exit code of the srun command
    exit_code=$?

    echo "[model] download job finished with exit code: $exit_code"
    exit "$exit_code"

fi
