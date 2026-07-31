#!/bin/bash
# tests/bash/shims/_shim-common.sh
#
# Sourced by every PATH shim (not a shim itself — has no shebang-executed
# purpose of its own). Provides call logging so tests can assert on what
# external commands were invoked and with what arguments.
#
# IVLLM_TEST_CALL_LOG is set by tests/bash/lib/sandbox.sh to /work/calls.log
# for every sandboxed run.

shim_log() {
    local tool="$1"
    shift
    local log="${IVLLM_TEST_CALL_LOG:-/dev/null}"
    printf '[%s] %s\n' "$tool" "$*" >> "$log" 2>/dev/null || true
}
