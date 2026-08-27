#!/bin/bash
# shellcheck disable=1091,2155
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

set -euo pipefail
umask 0002

vllmVersion=${1:?must supply a vllm version to setup.sh}

source "$(dirname "${BASH_SOURCE[0]}")/utils.sh"
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
    --extra-index-url "https://wheels.vllm.ai/$vllmVersion/cu129" \
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

# NCCL version pin — see design/prototype/nccl-probe.sh + design/active-issues.md.
# Torch's default bundled nvidia-nccl-cu12 (2.28.9 as of vllm 0.25.1) is
# exactly the class of build vLLM issue #46097 traces a multi-node TP
# CUDA-graph-replay collective desync to. 2.30.4 is reported clean under
# sustained multi-hour multi-node load by a third party on the same
# architecture family, and verified here to still correctly select the
# Cassini/libfabric (cxi) transport via the existing aws-ofi-nccl 1.8.1-aws
# plugin, with a correct cross-node all_reduce (nccl-probe.sh section [7]).
echo "=== Pinning nvidia-nccl-cu12 to 2.30.4 (see vLLM issue #46097) ==="
uv pip install --upgrade nvidia-nccl-cu12==2.30.4

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
  deepGEMMRef=$(curl -fsSL "https://raw.githubusercontent.com/vllm-project/vllm/refs/heads/releases/v$vllmVersion"/tools/install_deepgemm.sh | grep "DEEPGEMM_GIT_REF=" | head -n 1 | sed 's/.*="\(.*\)".*/\1/')

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


# Ensure build requirements match the host environment
uv pip install tomlkit nanobind wheel setuptools build pybind11 meson-python

if uv pip show deep-ep &>/dev/null && uv pip show uccl &>/dev/null && uv pip show nixl-cu12 &>/dev/null; then
    echo "UCCL already installed with DeepEP wrapper & NIXL support"
else

    echo "=== Building UCCL-EP with HPE Slingshot (CXI) transport support ==="

    # Ensure build requirements match Isambard's Grace Hopper nodes
    export TORCH_CUDA_ARCH_LIST="9.0a"
    export USE_LIBFABRIC_CXI=1
    export USE_DMABUF=1

    # replicates specific instructions from:
    # https://raw.githubusercontent.com/uccl-project/uccl/refs/heads/main/build_inner.sh

    # Install build dependency: nanobind
    echo "Installing UCCL-EP build dependency: nanobind..."


    # The doublewordAI fork has been built for isambard.
    # ucclEPgit="https://github.com/doublewordai/uccl.git"
    # most of the pull requests have been merged
    ucclEPgit="https://github.com/uccl-project/uccl.git"
    mkdir -p "$workingDir/uccl"

    if git clone --recursive --shallow-submodules -b main "$ucclEPgit" "$workingDir/uccl"; then

      pushd "$workingDir/uccl"
      echo "--> Compiling P2P extension components..."
          cd p2p && make clean && make "-j$(nproc)" && cd ..

          mkdir -p uccl/lib
          cp p2p/libuccl_p2p.so uccl/lib/
          cp p2p/p2p.*.so uccl/
          cp p2p/utils.py uccl/

          # Handle nanobind stable ABI naming convention if python >= 3.12
          py_stable_abi_ok=$(python3 -c "import sys; print(1 if sys.version_info >= (3, 12) else 0)")
          if [[ "$py_stable_abi_ok" == "1" ]]; then
              for f in uccl/*.cpython-*.so; do
                  if [[ -f "$f" ]]; then
                      #shellcheck disable=2001
                      newname=$(echo "$f" | sed 's/\.cpython-[^.]*-[^.]*-[^.]*\.so/.abi3.so/')
                      mv "$f" "$newname"
                  fi
              done
          fi

          # 3. Replicate 'build_ep' function from build_inner.sh
          echo "--> Compiling EP extension components..."
          pushd ep
          make clean
          rm -rf build || true

          # Run the setup.py inline tracking hook inside ep/
          python3 setup.py build_ext --inplace
          popd

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

    # NIXL SUPPORT
    # Needs to run alongside the UCCL compilation to pick up UCCL headers
    # See an alternative strategy here:
    # https://raw.githubusercontent.com/uccl-project/uccl/61ee42402819cabba3ac2a56dd4addec3363976c/p2p/NIXL_plugin_readme.md
    uv pip install meson ninja pybind11 tomlkit pytest patchelf types-PyYAML setuptools wheel

    echo "=== Compiling NIXL with HPE Slingshot (CXI) Support ==="

    # 2. Clone the core NIXL codebase
    mkdir -p "$workingDir/nixl"

    git clone --recursive https://github.com/ai-dynamo/nixl.git "$workingDir/nixl"
    pushd "$workingDir/nixl"

    LOCAL_FABRIC="/opt/cray/libfabric/1.22.0"

    TARGET_PLUGINS="LIBFABRIC,UCCL,POSIX"

    # 1. Point to your custom UCCL source and compiled library spaces
    export UCCL_STAGING_DIR="$workingDir/uccl"

    # We explicitly add the nested 'p2p/util' subdirectory to find common.h
    export CPATH="$UCCL_STAGING_DIR/include:$UCCL_STAGING_DIR/p2p:$UCCL_STAGING_DIR/p2p/util:$CPATH"
    export CPLUS_INCLUDE_PATH="$UCCL_STAGING_DIR/include:$UCCL_STAGING_DIR/p2p:$UCCL_STAGING_DIR/p2p/util:${CPLUS_INCLUDE_PATH:-}"

    # ── Extended Linker Path Matrix (Build-Time Binaries) ──
    # Points the compiler to find libuccl_p2p.so inside your staging workspace
    export LIBRARY_PATH="$UCCL_STAGING_DIR/uccl/lib:$UCCL_STAGING_DIR/p2p:$LIBRARY_PATH"
    export LD_LIBRARY_PATH="$UCCL_STAGING_DIR/uccl/lib:$UCCL_STAGING_DIR/p2p:$LD_LIBRARY_PATH"

    # 3. Compile the base C++ engine and build the target architecture wheel
    # Passing --no-build-isolation forces the setup layout to acknowledge Slingshot structures
    # NIXL_EP requires UCX/UCP which is Infiniband specific no path exists to
    # install on Slingshot as far as I can see.
    python3 -m build --wheel --no-isolation --skip-dependency-check \
      -Csetup-args="-Dcudapath_inc=$CUDA_HOME/include" \
      -Csetup-args="-Dcudapath_lib=$CUDA_HOME/lib64" \
      -Csetup-args="-Ddisable_gds_backend=true" \
      -Csetup-args="-Ddisable_mooncake_backend=true" \
      -Csetup-args="-Ddisable_infinia_backend=true" \
      -Csetup-args="-Dlibfabric_path=$LOCAL_FABRIC" \
      -Csetup-args="-Dnixl_cuda_arch_list=90" \
      -Csetup-args="-Dbuild_nixl_ep=false" \
      -Csetup-args="-Dbuild_tests=false" \
      -Csetup-args="-Dbuild_examples=false" \
      -Csetup-args="-Denable_plugins=$TARGET_PLUGINS" \
      -Csetup-args="-Dcpp_args=-Wno-error=deprecated-enum-enum-conversion" \
      -Csetup-args="-Dwerror=false"

    # 4. Install the resulting wheel package into your Python environment
    uv pip install --no-build-isolation dist/nixl*.whl

    popd
fi

# UCCL-EP enables deep_ep moe kernels on GH200
# see:
# https://docs.vllm.ai/en/latest/design/moe_kernel_features/

# UCCL-P2P allows limited NIXL support:
# This allows disaggregated prefilling feature. It provides fully asynchronous send/receive operations using the NIXL library for efficient cross-process KV cache transfer (with "kv_connector_extra_config":{"backends":["LIBFABRIC"]}).
# see:
# https://docs.vllm.ai/en/stable/features/nixl_connector_usage/
# https://docs.vllm.ai/en/stable/features/nixl_connector_compatibility/#model-architecture-x-capability

# TODO: work out a better check:
# before the package is called "humming"
# after uv pip list shows:
# humming-kernels                          0.1.10.post19+ga07f4b350
if uv pip show humming-kernels &>/dev/null; then
    echo "humming kernels already installed (doubleword-AI)"
else
    echo "=== Installing humming from doublewordAI ==="
    uv pip install git+https://github.com/doublewordai/humming.git
fi

# Humming moe-backend and linear-backend
# See:
# https://blog.doubleword.ai/throughputmaxxing-v4-flash-single-node


if uv pip show hpc &>/dev/null; then
    echo "Tencent HPC-OPS already installed"
else
    echo "=== Tencent HPC-OPS ==="
    git clone https://github.com/Tencent/hpc-ops.git "$workingDir/hpc-ops"
    cd "$workingDir/hpc-ops"

    # build packages
    make wheel
    # 4. Install the resulting wheel package into your Python environment
    uv pip install --no-build-isolation dist/*.whl
fi

# HPC-OPS:
# https://vllm.ai/blog/2026-07-06-vllm-hpc-ops
# Attention and moe backends optimised for hopper
# Hy3 series of models use it.
#     --attention-backend HPC_ATTN \
#     --kv-cache-dtype fp8_e4m3 \
#     --moe-backend hpc



