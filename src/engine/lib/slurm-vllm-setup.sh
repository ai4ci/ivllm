#!/bin/bash
#SBATCH --job-name=vllm-setup
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --mem=115G
#SBATCH --cpus-per-gpu=64
#SBATCH --ntasks-per-node=1
#SBATCH --time=03:00:00
#SBATCH --export=ALL
#SBATCH --partition=interactive
#SBATCH --reservation=interactive

set -euo pipefail
umask 0002

source "$(dirname "${BASH_SOURCE[0]}")/utils.sh"

vllmVersion=${1?must supply a vllm version to setup.sh}
vllmVersionDir=$(resolve_vllm_version_dir "$vllmVersion")

echo "=== ivllm-setup ==="
echo "Installing: $vllmVersion to $vllmVersionDir"

# Install uv if not already present
if ! command -v uv &>/dev/null; then
  echo "installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

# grab a local working directory for the "vllm-install" job:
workingDir=$(resolve_localdir "vllm-install")
# Use $workingDir (fast in-job scratch, wiped at job end)
trap 'rm -rf "$workingDir"' EXIT

echo "working directory: $workingDir"
nvhpcDir=$(resolve_nvhpc_dir)

# Phase A: Install NVIDIA HPC SDK 26.3 cuda_multi (provides CUDA 12.9 + 13.1)
if [ ! -d "$nvhpcDir/cuda" ]; then

  echo "=== Installing NVIDIA HPC SDK 26.3 (cuda_multi) ==="
  wget "https://developer.download.nvidia.com/hpc-sdk/26.3/nvhpc_2026_263_Linux_aarch64_cuda_multi.tar.gz" \
    -O "$workingDir/nvhpc.tar.gz"
  tar xpzf "$workingDir/nvhpc.tar.gz" -C "$workingDir"
  cd "$workingDir/nvhpc_2026_263_Linux_aarch64_cuda_multi"
  NVHPC_SILENT=true NVHPC_INSTALL_DIR=$nvhpcDir NVHPC_INSTALL_TYPE=single ./install
  rm -rf "$workingDir/nvhpc.tar.gz" "$workingDir/nvhpc_2026_263_Linux_aarch64_cuda_multi"
  echo "=== HPC SDK install complete ==="

else

  echo "=== HPC SDK already installed at ${nvhpcDir} — skipping ==="

fi

# sets LD_LIBRARY_PATH & other CUDA variables needed for install.
source "$(dirname "${BASH_SOURCE[0]}")/common-env.sh"

# Phase B: Install vLLM ${vllmVersion} into versioned venv
if [[ -f "$vllmVersionDir/bin/activate" ]]; then

  source "$vllmVersionDir/bin/activate"
  echo "=== vLLM ${vllmVersion} already installed at $vllmVersionDir — skipping ==="

else

  echo "=== Installing vLLM ${vllmVersion} ==="

  # UV_CACHE_DIR: use $workingDir (per-user in-job scratch) so multiple project
  # members don't share a single cache directory with conflicting permissions.
  # $workingDir is wiped at job end; the installed venv in $PROJECTDIR persists.
  export UV_CACHE_DIR=$workingDir/uv_cache

  uv venv "$vllmVersionDir" --python 3.12

  source "$vllmVersionDir/bin/activate"

  echo "Downloading and installing vLLM $vllmVersion wheels (may be slow — large download)..."
  uv pip install vllm=="$vllmVersion" ray[default] \
    --torch-backend=auto \
    --extra-index-url https://wheels.vllm.ai/$vllmVersion/cu129 \
    --extra-index-url https://pypi.org/simple/

  echo "vllm install complete."

  # Copy fused MoE for H200
  pushd "$vllmVersionDir/lib/python3.12/site-packages/vllm/model_executor/layers/fused_moe/configs"

  # copies all the H200 fused MoE configs as GH200
  for f in *device_name=NVIDIA_H200*; do
    cp "$f" "\${f//device_name=NVIDIA_H200/device_name=NVIDIA_GH200_120GB}";
  done

  popd

  echo "=== vLLM version installed ==="
  python -c "import importlib.metadata; print('vllm', importlib.metadata.version('vllm'))"

fi

export FLASHINFER=$(uv pip list --format=json | jq '.[] | select(.name == "flashinfer-python") | .version' -r)

if uv pip show flashinfer-jit-cache &>/dev/null; then
  echo "flashinfer-jit-cache already installed"
else
  echo "=== Installing flashinfer-jit-cache ($FLASHINFER) ==="
  uv pip install flashinfer-jit-cache=="$FLASHINFER" --index-url https://flashinfer.ai/whl/cu129
  echo "flashinfer-jit-cache ($FLASHINFER) install complete."
fi

# DeepGEMM

if uv pip show deep_gemm &>/dev/null; then
  echo "DeepGEMM already installed"
else
  deepGEMMRef=$(curl -fsSL https://raw.githubusercontent.com/vllm-project/vllm/refs/heads/releases/v${vllmVersion}/tools/install_deepgemm.sh | grep "DEEPGEMM_GIT_REF=" | head -n 1 | sed 's/.*="\(.*\)".*/\1/')

  if [[ -z ${deepGEMMRef:-} ]]; then
    echo "WARNING: no DeepGEMM git reference found to compile"
  else

    echo "=== compiling DeepGEMM from source ==="

    deepGEMMgit="https://github.com/deepseek-ai/DeepGEMM.git"
    mkdir -p "$workingDir/deepgemm"
    # Checkout the specific reference
    git clone --recursive --shallow-submodules "$deepGEMMgit" "$workingDir/deepgemm"
    pushd "$workingDir/deepgemm"

    # Checkout the specific reference
    git checkout "$deepGEMMRef"

    # COMPILE AND PIP INSTALL VIA UV
    echo "Compiling DeepGEMM C++/CUDA extensions directly into venv..."
    uv pip install --no-build-isolation -vvv .
    echo "DeepGEMM successfully compiled and installed from tmpfs."
    popd
    echo "DEEPGEMM_SETUP_SUCCESS"

  fi
fi

# DeepEP

export EP_NVSHMEM_ROOT_DIR="$NVSHMEM_DIR"
export EP_NCCL_ROOT_DIR="/tools/brics/apps/linux-sles15-neoverse_v2/gcc-12.3.0/aws-ofi-nccl-1.8.1-c47cd5ivrugm3jzlyqyis4igyflnydmo"

if uv pip show deepep &>/dev/null; then
  echo "DeepEP already installed"
else
  deepEpRef=$(curl -fsSL "https://raw.githubusercontent.com/vllm-project/vllm/refs/heads/releases/v$vllmVersion/tools/ep_kernels/install_python_libraries.sh" | grep "DEEPEP_COMMIT_HASH=" | head -n 1 | sed 's/.*-"\(.*\)".*/\1/')

  if [[ -z ${deepEpRef:-} ]]; then
    echo "WARNING: no DeepEP git reference found to compile"
  else

    echo "=== compiling DeepEP from source ==="

    deepEPgit="https://github.com/deepseek-ai/DeepEP.git"
    mkdir -p "$workingDir/deepep"
    # Checkout the specific reference
    git clone --recursive --shallow-submodules "$deepEPgit" "$workingDir/deepep"
    pushd "$workingDir/deepep"

    # Checkout the specific reference
    git checkout "$deepEpRef"

    # COMPILE AND PIP INSTALL VIA UV
    echo "Compiling DeepEP C++/CUDA extensions directly into venv..."
    uv pip install --no-build-isolation -vvv .
    echo "DeepEP successfully compiled and installed from tmpfs."
    popd
    echo "DEEPEP_SETUP_SUCCESS"

  fi
fi
