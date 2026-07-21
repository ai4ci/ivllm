#!/bin/bash
#SBATCH --job-name=hf-download
#SBATCH --nodes=1
#SBATCH --mem=20G
#SBATCH --ntasks-per-node=1
#SBATCH --time=03:00:00
#SBATCH --export=ALL
#SBATCH --partition=interactive
#SBATCH --reservation=interactive

# shellcheck disable=SC2155,SC1091
# N.b. set the log in the wrapper script (sbatch --output and --error flags)

set -euo pipefail
umask 0002

source "./utils.sh"


hfModel=${1?Must specify a model to download}

if [[ -z ${HF_TOKEN:-} ]]; then
  echo "Must set the HF_TOKEN env variable"
  exit 1
fi

modelDir=$(resolve_model_dir)

export HF_HOME="$modelDir/hf"
hfVenv="$modelDir/venv"



echo "=== hf-download ==="
echo "Downloading: ${hfModel} to ${HF_HOME}"

# Check if the virtual environment binary exists instead of checking the command path
if [ ! -f "$hfVenv/bin/hf" ]; then

  # Install uv if not already present
  if ! command -v uv &>/dev/null; then
    echo "installing uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
  fi

  echo "installing hf"
  # grab a local working directory for the "hf-download" job:
  workingDir=$(resolve_localdir "hf-download")
  echo "working directory: $workingDir"

  export UV_CACHE_DIR=$workingDir/uv_cache
  uv venv "$hfVenv" --python 3.12
  source "$hfVenv/bin/activate"
  uv pip install huggingface_hub[cli]
  echo "inistalled hf cli"

else
  echo "hf virtual environment found. Activating..."
  source "$hfVenv/bin/activate"
fi

# Ensure hf is available in the current environment context
if ! command -v hf &>/dev/null; then
  echo "error: hf command not found even after activation." >&2
  exit 1
fi

# download the model.
source "$hfVenv/bin/activate"
hf download "$hfModel"
