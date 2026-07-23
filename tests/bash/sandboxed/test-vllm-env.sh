#!/bin/bash
# tests/bash/sandboxed/test-vllm-env.sh — Environment preamble tests.
#
# Exercises common-env.sh (NVHPC/CUDA/compiler setup) and vllm-env.sh
# (NCCL/Slingshot/vLLM tuning) together, the same way run_head_vllm.sh and
# run_worker_vllm.sh source them. Runs in the "compute" profile sandbox
# because common-env.sh calls `module` (shimmed no-op) and `which gcc`/
# `which g++` (shimmed present, see tests/bash/shims/gcc and g++).

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../lib" && pwd)/sandbox.sh"

# resolve_nvhpc_root() requires $IVLLM_PROJECTDIR/engine/nvhpc/Linux_aarch64/26.3
# to already exist (i.e. `ivllm setup` has already run) — set that up as a
# fixture before sourcing common-env.sh.
#
# NOTE: common-env.sh/vllm-env.sh are sourced here with `set +u`, matching
# their *actual* invocation context in run_head_vllm.sh/run_worker_vllm.sh
# (neither sets `-u`). Without this, sourcing common-env.sh crashes with
# "NVSHMEM_DIR: unbound variable" — a real latent bug, tracked as Issue 11
# in design/issues.md, not something these tests should silently paper over
# by weakening their own assertions.
_NVHPC_FIXTURE='
    mkdir -p "$IVLLM_PROJECTDIR/engine/nvhpc/Linux_aarch64/26.3"
    set +u
'

sandbox_run_test "common_env_sources" compute "
    $_NVHPC_FIXTURE
    source /work/project/engine/lib/common-env.sh
    echo sourced ok
"

sandbox_run_test "common_env_vars_set" compute "
    $_NVHPC_FIXTURE
    source /work/project/engine/lib/common-env.sh

    [ -n \"\$CUDA_HOME\" ] || { echo 'FAIL: CUDA_HOME not set'; exit 1; }
    [ -n \"\$CUDA_VERSION\" ] || { echo 'FAIL: CUDA_VERSION not set'; exit 1; }
    [[ \"\$CUDA_HOME\" == *\"/nvhpc/Linux_aarch64/26.3/cuda/12.9\" ]] || { echo \"FAIL: unexpected CUDA_HOME: \$CUDA_HOME\"; exit 1; }
    [ -n \"\$CC\" ] || { echo 'FAIL: CC not set'; exit 1; }
    [ -n \"\$CXX\" ] || { echo 'FAIL: CXX not set'; exit 1; }
    [[ \"\$LD_LIBRARY_PATH\" == *\"/cuda/12.9/compat\"* ]] || { echo 'FAIL: LD_LIBRARY_PATH missing compat dir'; exit 1; }
"

sandbox_run_test "common_env_missing_nvhpc_falls_through_empty" compute '
    set +u
    # RED (expected to fail until design/issues.md Issue 12 is fixed):
    # resolve_nvhpc_root() echoes its "not installed" diagnostic to STDOUT
    # instead of stderr, so common-env.sh'"'"'s
    # `export NVHPC_ROOT=$(resolve_nvhpc_root)` captures the error *message*
    # as the value of NVHPC_ROOT instead of leaving it empty. This test
    # documents the intended (correct) behaviour.
    source /work/project/engine/lib/common-env.sh 2>/work/err.log
    [ -z "$NVHPC_ROOT" ] || { echo "FAIL: expected empty NVHPC_ROOT, got: $NVHPC_ROOT"; exit 1; }
'

sandbox_run_test "vllm_env_sources_and_sets_vars" compute "
    $_NVHPC_FIXTURE
    source /work/project/engine/lib/common-env.sh
    source /work/project/engine/lib/vllm-env.sh

    [ -n \"\$NCCL_CROSS_NIC\" ] || { echo 'FAIL: NCCL_CROSS_NIC not set'; exit 1; }
    [ -n \"\$FI_PROVIDER\" ] || { echo 'FAIL: FI_PROVIDER not set'; exit 1; }
    [ -n \"\$VLLM_ENGINE_ITERATION_TIMEOUT_S\" ] || { echo 'FAIL: VLLM_ENGINE_ITERATION_TIMEOUT_S not set'; exit 1; }
    [[ \"\$VLLM_LOGGING_CONFIG_PATH\" == */vllm_logs.json ]] || { echo \"FAIL: VLLM_LOGGING_CONFIG_PATH wrong: \$VLLM_LOGGING_CONFIG_PATH\"; exit 1; }
    assert_file_exists \"\$VLLM_LOGGING_CONFIG_PATH\" || exit 1

    # set_jit_caches (called at the end of vllm-env.sh) should point cache
    # dirs at the per-job node-local scratch dir.
    [[ \"\$VLLM_CACHE_ROOT\" == /local/* ]] || { echo \"FAIL: VLLM_CACHE_ROOT not under /local: \$VLLM_CACHE_ROOT\"; exit 1; }
"

exit "$FAIL"
