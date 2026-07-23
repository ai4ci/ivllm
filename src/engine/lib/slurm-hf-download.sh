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

source "$(dirname "${BASH_SOURCE[0]}")/utils.sh"

hfModel=${1?Must specify a model to download}

if [[ -z ${HF_TOKEN:-} ]]; then
  echo "[model] ERROR: must set the HF_TOKEN env variable" >&2
  exit 1
fi

modelDir=$(resolve_model_dir)

export HF_HOME="$modelDir/hf"
hfVenv="$modelDir/venv"

echo "[model] === hf-download ==="
echo "[model] downloading: ${hfModel} to ${HF_HOME}"

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

# download the model.
source "$hfVenv/bin/activate"
echo "[model] starting model download ($hfModel)"
hf download "$hfModel"
echo "[model] model download complete ($hfModel)"

