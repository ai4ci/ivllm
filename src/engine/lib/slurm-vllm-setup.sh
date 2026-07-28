#!/bin/bash
# slurm-vllm-setup.sh — vLLM installation script.
#
# Installs the specified vLLM version into a shared versioned directory.
# Downloads NVHPC SDK if needed, creates venv, installs PyTorch/vLLM,
# and compiles CUDA kernels for JIT cache warmup.
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
if [ ! -d "$nvhpcDir/Linux_aarch64/26.3/cuda" ]; then

  echo "=== Installing NVIDIA HPC SDK 26.3 (cuda_multi) to $nvhpcDir ==="
  wget --progress=dot:giga "https://developer.download.nvidia.com/hpc-sdk/26.3/nvhpc_2026_263_Linux_aarch64_cuda_multi.tar.gz" \
    -O "$workingDir/nvhpc.tar.gz"
  tar xpzf "$workingDir/nvhpc.tar.gz" -C "$workingDir"
  cd "$workingDir/nvhpc_2026_263_Linux_aarch64_cuda_multi"
  NVHPC_SILENT=true NVHPC_INSTALL_DIR=$nvhpcDir NVHPC_INSTALL_TYPE=single ./install
  rm -rf "$workingDir/nvhpc.tar.gz" "$workingDir/nvhpc_2026_263_Linux_aarch64_cuda_multi"
  echo "=== HPC SDK install complete ==="

else

  echo "=== HPC SDK already installed at ${nvhpcDir} — skipping ==="

fi

# RDMA compile from source
# Compile rdma-core from source to provide the necessary libibverbs headers and libraries if missing
rdma_install_dir=$(resolve_rdma_dir)
if [[ ! -f "$rdma_install_dir/include/infiniband/verbs.h" ]]; then
  echo "infiniband/verbs.h not found in system or local headers. Compiling rdma-core from source to $rdma_install_dir..."
  rdmaCoreGit="https://github.com/linux-rdma/rdma-core.git"
  mkdir -p "$workingDir/rdma-core-src"
  if git clone --depth 1 "$rdmaCoreGit" "$workingDir/rdma-core-src"; then
    pushd "$workingDir/rdma-core-src"

    rm -rf build && mkdir build && cd build

    # 2. Configure using IN_PLACE and skip hardware providers
    if cmake -DIN_PLACE=1 \
              -DNO_PROVIDERS=ON \
              -DENABLE_VALGRIND=OFF \
              -DENABLE_LOG_ERRORS=OFF \
              -DNO_MAN_PAGES=ON \
              -DENABLE_PYTHON=OFF \
              .. && make -j16; then

      # 3. Create your custom installation directories manually
      mkdir -p "$rdma_install_dir/include/infiniband"
      mkdir -p "$rdma_install_dir/lib"

      # 4. Copy the compiled Infiniband Verb headers to your target include dir
      cp -rL include/infiniband/* "$rdma_install_dir/include/infiniband/"

      # 5. Optional: Copy the generated libibverbs stub libraries if your program links to them
      cp -dL lib/libibverbs* "$rdma_install_dir/lib/" 2>/dev/null || true

      echo "rdma-core headers safely extracted to $rdma_install_dir!"

    else
      echo "WARNING: Failed to compile rdma-core from source."
    fi
    popd
  fi
else
  if [[ -d "$rdma_install_dir" ]]; then
    echo "=== Local userspace rdma-core already installed at $rdma_install_dir — reusing. === "
  fi
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
  export UV_LINK_MODE=copy

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
    cp "$f" "${f//device_name=NVIDIA_H200/device_name=NVIDIA_GH200_120GB}";
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

# DeepEP or UCCL-EP (High-Performance MoE Collective transport)
# export EP_NVSHMEM_ROOT_DIR="$NVSHMEM_DIR"
# export EP_NCCL_ROOT_DIR="$NVHPC_ROOT/comm_libs/$CUDA_VERSION/nccl"


# Ensure build requirements match the host environment
uv pip install tomlkit nanobind wheel setuptools build pybind11

if uv pip show deepep &>/dev/null && uv pip show uccl &>/dev/null; then
    echo "UCCL already installed with DeepEP wrapper"
else
#   deepEpRef=$(curl -fsSL "https://raw.githubusercontent.com/vllm-project/vllm/refs/heads/releases/v$vllmVersion/tools/ep_kernels/install_python_libraries.sh" | grep "DEEPEP_COMMIT_HASH=" | head -n 1 | sed 's/.*-"\(.*\)".*/\1/')
#
#   compiled_any_ep=0
#
#   if [[ -n ${deepEpRef:-} ]]; then
#     echo "=== compiling DeepEP from source ==="
#
#     deepEPgit="https://github.com/deepseek-ai/DeepEP.git"
#     mkdir -p "$workingDir/deepep"
#     # Checkout the specific reference
#     if git clone --recursive --shallow-submodules "$deepEPgit" "$workingDir/deepep"; then
#       pushd "$workingDir/deepep"
#       git checkout "$deepEpRef"
#
#       # COMPILE AND PIP INSTALL VIA UV
#       echo "Compiling DeepEP C++/CUDA extensions directly into venv..."
#       if uv pip install --no-build-isolation -vvv .; then
#         echo "DeepEP successfully compiled and installed from tmpfs."
#         echo "DEEPEP_SETUP_SUCCESS"
#         compiled_any_ep=1
#       else
#         echo "WARNING: DeepEP compilation failed (this is expected on systems without Mellanox InfiniBand such as HPE Slingshot)."
#       fi
#       popd
#     fi
#   else
#     echo "WARNING: no DeepEP git reference found to compile"
#   fi
#
#   # Fallback to UCCL-EP if DeepEP compilation is skipped or fails
#   if [[ $compiled_any_ep -eq 0 ]]; then


    echo "=== Building UCCL-EP with HPE Slingshot (CXI) transport support ==="


    # Ensure build requirements match Isambard's Grace Hopper nodes
    export TORCH_CUDA_ARCH_LIST="9.0a"
    export USE_LIBFABRIC_CXI=1
    export USE_DMABUF=1

    # repicates specific instructions from:
    # https://raw.githubusercontent.com/uccl-project/uccl/refs/heads/main/build_inner.sh

    # Install build dependency: nanobind
    echo "Installing UCCL-EP build dependency: nanobind..."


    # The doublewordAI fork has been built for isambard.
    ucclEPgit="https://github.com/doublewordai/uccl.git"
    mkdir -p "$workingDir/uccl"

    if git clone --recursive --shallow-submodules -b main "$ucclEPgit" "$workingDir/uccl"; then

#         pushd "$workingDir/uccl/ep"
#         export USE_LIBFABRIC_CXI=1
#         export USE_DMABUF=1
#
#         echo "Executing native UCCL wheel compilation loop..."
#         # Format: bash build.sh [cuda_version] [module_target] [python_version] --install
#         if python3 setup.py build_ext --inplace; then
#             echo "UCCL-EP compiled successfully. Manually mapping binary to site-packages..."
#
#             # Identify your virtual environment layout
#             VENV_SITE_PACKAGES="$vllmVersionDir/lib/python3.12/site-packages"
#             mkdir -p "$VENV_SITE_PACKAGES/uccl"
#
#             # 5. Place the true compiled .so binary and initialize the module namespace
#             cp ep.abi3.so "$VENV_SITE_PACKAGES/uccl/"
#             touch "$VENV_SITE_PACKAGES/uccl/__init__.py"
#
#             # 6. Generate and register package metadata to satisfy pip dependency tracking
#             python3 setup.py egg_info
#             cp -r *.egg-info "$VENV_SITE_PACKAGES/uccl-0.0.1-py3.12.egg-info"
#             echo "UCCL-EP successfully manually integrated."
#
#             # 7. Install doublewordai's deep_ep drop-in wrapper helper
#             echo "Installing deep_ep drop-in wrapper..."
#             if uv pip install --no-build-isolation -vvv ./deep_ep_wrapper; then
#                 echo "deep_ep_wrapper successfully compiled and integrated."
#             else
#                 echo "WARNING: UCCL-EP compiled but deep_ep_wrapper setup encountered an error."
#             fi
#         else
#             echo "WARNING: Host-native UCCL-EP compilation failed."
#         fi

      pushd "$workingDir/uccl"
      echo "--> Compiling P2P extension components..."
          cd p2p && make clean && make -j$(nproc) && cd ..

          mkdir -p uccl/lib
          cp p2p/libuccl_p2p.so uccl/lib/
          cp p2p/p2p.*.so uccl/
          cp p2p/utils.py uccl/

          # Handle nanobind stable ABI naming convention if python >= 3.12
          py_stable_abi_ok=$(python3 -c "import sys; print(1 if sys.version_info >= (3, 12) else 0)")
          if [[ "$py_stable_abi_ok" == "1" ]]; then
              for f in uccl/*.cpython-*.so; do
                  if [[ -f "$f" ]]; then
                      #shellcheck disable 2001
                      newname=$(echo "$f" | sed 's/\.cpython-[^.]*-[^.]*-[^.]*\.so/.abi3.so/')
                      mv "$f" "$newname"
                  fi
              done
          fi

          # 3. Replicate 'build_ep' function from build_inner.sh
          echo "--> Compiling EP extension components..."
          cd ep && make clean && rm -rf build || true

          # Run the setup.py inline tracking hook inside ep/
          python3 setup.py build_ext --inplace
          cd ..

          # Mirror the metadata hooks into the target workspace
          cp -r ep/build/lib.linux-aarch64-*/* uccl/ 2>/dev/null || cp -r ep/*.so uccl/ 2>/dev/null || true

          # 4. Run the final 'python3 -m build' command from build_inner.sh
          echo "--> Executing top-level package compilation pass..."
          python3 -m build --wheel --no-isolation

          # 5. Extract and install the finished, native host wheel file via uv
          echo "--> Deploying unified UCCL wheel..."
          uv pip install --no-build-isolation dist/uccl-*.whl

          # 6. Install the deep_ep_wrapper so vLLM can use it as a drop-in replacement
          echo "--> Integrating deep_ep_wrapper..."
          cd ep
          uv pip install --no-build-isolation -vvv ./deep_ep_wrapper
        popd
    else
        echo "WARNING: Failed to clone UCCL repository."
    fi

    echo "installing humming from doublewordAI"
    uv pip install git+https://github.com/doublewordai/humming.git

fi

if uv pip show nixl &>/dev/null; then
    echo "NIXL with slingshot already installed"
else
    echo "=== Compiling NIXL with HPE Slingshot (CXI) Support ==="

    # 2. Clone the core NIXL codebase
    rm -rf "$workingDir/nixl"
    git clone --recursive https://github.com/ai-dynamo/nixl.git "$workingDir/nixl"
    cd "$workingDir/nixl"

    # 3. Compile the base C++ engine and build the target architecture wheel
    # Passing --no-build-isolation forces the setup layout to acknowledge Slingshot structures
    python3 -m build --wheel --no-isolation

    # 4. Install the resulting wheel package into your Python environment
    uv pip install --no-build-isolation dist/nixl-*.whl

fi
