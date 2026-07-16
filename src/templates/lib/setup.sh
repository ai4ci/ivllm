#
#
#
#  * Parameters for generating the `{@link renderSetupScript}` SLURM script.
#  *
#  * Passed via `{@link ProcessState}` and consumed by the setup template
#  * to install NVIDIA HPC SDK and vLLM into a versioned virtualenv.
#  *
#  * | Field | Description |
#  * |-------|-------------|
#  * | `vllmVersion` | vLLM version to install (e.g. `'0.22.0'`) |
#  * | `hfToken` | Optional HuggingFace token for gated models |
#  * @see ProcessState
#  * @see renderSetupScript
#  */
# export interface SetupScriptOptions {
#   /** vLLM version string to install (e.g. `'0.22.0'`) */
#   vllmVersion: string;
#   /** Optional HuggingFace access token for gated models */
#   hfToken?: string;
# }
#
# /**
#  * Generates the `ivllm setup` SLURM batch script for installing
#  * NVIDIA HPC SDK and vLLM into a versioned virtualenv on the Isambard HPC.
#  *
#  * The script executes in two phases:
#  *
#  * 1. **Phase A** — Downloads and installs NVIDIA HPC SDK 26.3
#  *    (`cuda_multi` package with CUDA 12.9 + 13.1) into the
#  *    `{@link SimplePaths.nvhpcDir}` directory.
#  * 2. **Phase B** — Creates a Python 3.12 virtualenv and installs
#  *    vLLM and Ray via `uv pip`, using the vLLM CUDA 12.9 wheel index.
#  *
#  * If the NVHPC SDK or vLLM version is already installed, the
#  * corresponding phase is skipped.
#  *
#  * | Field | Description |
#  * |-------|-------------|
#  * | `vllmVersion` | vLLM version to install (e.g. `'0.22.0'`) |
#  * | `hfToken` | Optional HuggingFace token for gated models |
#  * @param ss - Process state containing paths, version, and NVHPC settings
#  * @param remoteSetupLog - Remote path for the SLURM stdout/stderr log file
#  * @returns Complete bash script string ready for `sbatch` submission
#  * @see ProcessState
#  * @see SetupScriptOptions
#  * @see renderInferenceScript
#  */
# export async function renderSetupScript(
#   ss: ProcessState,
#   remoteLogFile: string,
# ): Promise<string> {
#   const paths = ss.paths;
#   const vllmVersion = ss.vllmVersion;
#   const venvDir = paths.remoteProjectVllmVersionDir;
#   const nvhpcDir = paths.nvhpcDir;
#   const nvhpcRoot = paths.nvhpcRoot;
#   const ldLibPath = [
#     `$NVHPC_ROOT/cuda/12.9/compat`,
#     `$NVHPC_ROOT/cuda/12.9/lib64`,
#     `$NVHPC_ROOT/compilers/lib`,
#     `$NVHPC_ROOT/comm_libs/12.9/nccl/lib`,
#     `$NVHPC_ROOT/comm_libs/12.9/nvshmem/lib`,
#     `$NVHPC_ROOT/math_libs/12.9/lib64`,
#     `\${LD_LIBRARY_PATH:-}`,
#   ].join(':');
#   const installDeepGEMM = await renderInstallDeepGEMM(ss);
#
# #!/bin/bash
# exec > >(tee -a "${remoteLogFile}") 2>&1
# set -euo pipefail
#
# export vllmVersion="0.19.0"
#
# echo "=== ivllm-setup version ${__VERSION__} ==="
# echo "Installing: ${vllmVersion}"
#
# # Install uv if not already present
# if ! command -v uv &>/dev/null; then
#   curl -LsSf https://astral.sh/uv/install.sh | sh
#   export PATH="$HOME/.local/bin:$PATH"
# fi
#
# mkdir -p ${paths.remoteProjectVllmDir} ${paths.remoteProjectDir}/hf
# # Ensure group-writable so other project members can install their own versioned venvs.
# chmod g+w ${paths.remoteProjectVllmDir}
# # Ensure HuggingFace cache directory is also group-writable for shared downloads.
# chmod g+w ${paths.remoteProjectDir}/hf
#
# # Phase A: Install NVIDIA HPC SDK 26.3 cuda_multi (provides CUDA 12.9 + 13.1)
# if [ ! -d ${nvhpcDir} ]; then
#   echo "=== Installing NVIDIA HPC SDK 26.3 (cuda_multi) ==="
#   # Use $LOCALDIR (fast in-job scratch, wiped at job end) not /tmp (policy: never use /tmp)
#   wget https://developer.download.nvidia.com/hpc-sdk/26.3/nvhpc_2026_263_Linux_aarch64_cuda_multi.tar.gz \\
#     -O $LOCALDIR/nvhpc.tar.gz
#   tar xpzf $LOCALDIR/nvhpc.tar.gz -C $LOCALDIR
#   cd $LOCALDIR/nvhpc_2026_263_Linux_aarch64_cuda_multi
#   NVHPC_SILENT=true NVHPC_INSTALL_DIR=${nvhpcDir} NVHPC_INSTALL_TYPE=single ./install
#   rm -rf $LOCALDIR/nvhpc.tar.gz $LOCALDIR/nvhpc_2026_263_Linux_aarch64_cuda_multi
#   echo "=== HPC SDK install complete ==="
# else
#   echo "=== HPC SDK already installed at ${nvhpcDir} — skipping ==="
# fi
#
# module load gcc-native/14.2
#
# # 2. Find the exact paths to the newly loaded compilers
# export CC=$(which gcc)
# export CXX=$(which g++)
#
# export NVHPC_ROOT=${nvhpcRoot}
# export CUDA_HOME=$NVHPC_ROOT/cuda/12.9
# export CPATH=$NVHPC_ROOT/math_libs/12.9/include:\${CPATH:-}
# export MAX_JOBS=16
# export FLASHINFER_NVCC_THREADS=4
# export PATH=$CUDA_HOME/bin:$PATH
# export LD_LIBRARY_PATH=${ldLibPath}
#
# # Phase B: Install vLLM ${vllmVersion} into versioned venv
# if [ ! -d ${venvDir} ]; then
#   echo "=== Installing vLLM ${vllmVersion} ==="
#   # UV_CACHE_DIR: use $LOCALDIR (per-user in-job scratch) so multiple project
#   # members don't share a single cache directory with conflicting permissions.
#   # $LOCALDIR is wiped at job end; the installed venv in $PROJECTDIR persists.
#   export UV_CACHE_DIR=$LOCALDIR/uv_cache
#   uv venv ${venvDir} --python 3.12
#   source ${paths.remoteProjectVllmVenvActivate}
#   echo "Downloading and installing vLLM ${vllmVersion} wheels (may be slow — large download)..."
#   uv pip install vllm==${vllmVersion} ray[default] \\
#     --torch-backend=auto \\
#     --extra-index-url https://wheels.vllm.ai/${vllmVersion}/cu129 \\
#     --extra-index-url https://pypi.org/simple/
#   echo "uv install complete."
#   echo "=== vLLM version ==="
#   python -c "import importlib.metadata; print('vllm', importlib.metadata.version('vllm'))"
#   echo "IVLLM_SETUP_SUCCESS"
# else
#   source ${paths.remoteProjectVllmVenvActivate}
#   echo "=== vLLM ${vllmVersion} already installed at ${venvDir} — skipping ==="
#   echo "IVLLM_SETUP_SUCCESS"
# fi
#
# export FLASHINFER=$(uv pip list --format=json | jq '.[] | select(.name == "flashinfer-python") | .version' -r)
# echo "=== Installing flashinfer-jit-cache ($FLASHINFER) ==="
# uv pip install flashinfer-jit-cache==$FLASHINFER --index-url https://flashinfer.ai/whl/cu129
# echo "flashinfer-jit-cache ($FLASHINFER) install complete."
#
# ${installDeepGEMM}
#
# `.trimStart();
# }
#
# async function getDeepGemmRef(
#   vllmVersion: string,
# ): Promise<string | undefined> {
#   const url = `https://raw.githubusercontent.com/vllm-project/vllm/refs/heads/releases/v${vllmVersion}/tools/install_deepgemm.sh`;
#
#   try {
#     const response = await fetch(url);
#     if (!response.ok) throw new Error(`HTTP error! Status: ${response.status}`);
#
#     const text = await response.text();
#
#     // Matches DEEPGEMM_GIT_REF="any_value" or DEEPGEMM_GIT_REF='any_value'
#     const match = text.match(/DEEPGEMM_GIT_REF=["']([^"']+)["']/);
#
#     return match ? match[1]! : undefined;
#   } catch (error) {
#     console.error('Failed to fetch or parse the file:', error);
#     return undefined;
#   }
# }
#
# async function renderInstallDeepGEMM(ss: ProcessState): Promise<string> {
#   const vllmVersion = ss.vllmVersion;
#   const deepGEMMDir = `${ss.paths.remoteProjectVllmDir}/deepGEMM/${vllmVersion}`;
#   const deepGEMMRef = await getDeepGemmRef(vllmVersion);
#
#   if (deepGEMMRef) {
#     // if (semverGte(vllmVersion, '0.23.0')) return '';
#     return `
# source ${ss.paths.remoteProjectVllmVenvActivate}
#
# if [ ! -d ${deepGEMMDir} ]; then
#
#   echo "=== compiling DeepGEMM from source ==="
#
#   INSTALL_DIR=$(mktemp -d)
#   DEEPGEMM_GIT_REPO="https://github.com/deepseek-ai/DeepGEMM.git"
#   DEEPGEMM_GIT_REF=${deepGEMMRef!}
#
#   export CUDA_VERSION="12.9"
#
#   mkdir -p "$INSTALL_DIR/deepgemm"
#
#   trap 'rm -rf "$INSTALL_DIR"' EXIT
#   rm -rf -- build dist *.egg-info 2>/dev/null || true
#
#   # Checkout the specific reference
#   git clone --recursive --shallow-submodules "$DEEPGEMM_GIT_REPO" "$INSTALL_DIR/deepgemm"
#   pushd "$INSTALL_DIR/deepgemm"
#
#   # Checkout the specific reference
#   git checkout "$DEEPGEMM_GIT_REF"
#
#   # Clean previous build artifacts
#   # (Based on https://github.com/deepseek-ai/DeepGEMM/blob/main/install.sh)
#   rm -rf -- build dist *.egg-info 2>/dev/null || true
#
#   echo "🏗️  Building DeepGEMM wheel..."
#   python3 setup.py bdist_wheel
#
#   mkdir -p ${deepGEMMDir}
#   cp dist/*.whl ${deepGEMMDir}
#   echo "DeepGEMM Wheel built and copied to ${deepGEMMDir}"
#   popd
#
# fi
#
# echo "=== Installing precomplied DeepGEMM ==="
# uv pip install ${deepGEMMDir}/*.whl
# echo "DEEPGEMM_SETUP_SUCCESS"
# `;
#   } else {
#     return `
# # No DeepGEMM identified for ${vllmVersion}
# `;
#   }
# }
