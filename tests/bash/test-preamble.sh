#!/bin/bash
# tests/bash/test-preamble.sh — Preamble environment validation tests.

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/test-utils.sh"

FAIL=0

setup() {
    export NVHPC_ROOT="/projects/p/ivllm/nvhpc/Linux_aarch64/26.3"
    export VLLM_ENGINE_DIR="/projects/p/engine"
}

test_preamble_sources() {
    setup

    if source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../src/templates/lib" && pwd)/preamble.sh" 2>&1; then
        echo "✓ test_preamble_sources"
    else
        echo "FAIL: preamble.sh failed to source"
        FAIL=1
    fi
}

test_env_vars_set() {
    setup
    source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../src/templates/lib" && pwd)/preamble.sh"

    # Check key env vars
    [ -n "$CUDA_HOME" ] || { echo "FAIL: CUDA_HOME not set"; FAIL=1; return; }
    [ -n "$CUDA_VERSION" ] || { echo "FAIL: CUDA_VERSION not set"; FAIL=1; return; }
    [ -n "$NCCL_CROSS_NIC" ] || { echo "FAIL: NCCL_CROSS_NIC not set"; FAIL=1; return; }
    [ -n "$FI_PROVIDER" ] || { echo "FAIL: FI_PROVIDER not set"; FAIL=1; return; }
    [ -n "$VLLM_ENGINE_ITERATION_TIMEOUT_S" ] || { echo "FAIL: TIMEOUT not set"; FAIL=1; return; }
    [ -n "$CC" ] && [ "$CC" = "gcc" ] || { echo "FAIL: CC not gcc"; FAIL=1; return; }

    # Check LD_LIBRARY_PATH contains compat dir first
    [[ "$LD_LIBRARY_PATH" == "/projects/p/ivllm/nvhpc/Linux_aarch64/26.3/cuda/12.9/compat:"* ]] || {
        echo "FAIL: LD_LIBRARY_PATH should start with compat dir"
        FAIL=1
        return
    }

    echo "✓ test_env_vars_set"
}

# ── Run all tests ───────────────────────────────────────────────────────────

test_preamble_sources
test_env_vars_set

exit "$FAIL"
