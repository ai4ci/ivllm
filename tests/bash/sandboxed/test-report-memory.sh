#!/bin/bash
# shellcheck disable=SC2016
# tests/bash/sandboxed/test-report-memory.sh — report_memory()/report_gpu()/
# report_processes() output shape.
#
# Updated 2026-09-03: these three used to be one monolithic report_memory()
# that read $IVLLM_DEBUG_LEVEL internally and gated GPU/py-spy detail on it.
# The "refactor debugging" commit split that apart — report_memory() now
# always prints just the base Cache/RAM/Top line unconditionally (no
# internal gating at all); report_gpu() and report_processes() are separate
# functions, invoked independently by monitor_node(), driven by the job's
# *config* setting `.ivllm-debug-level` rather than the $IVLLM_DEBUG_LEVEL
# env var. These tests now cover each function's own, narrower contract.
#
# Deliberately loose on exact formatting (see the original comment this test
# file kept from before the split): assertions pin to shape and gating
# behaviour, not exact content — e.g. "the Top field is non-empty" rather
# than "the Top field contains ray::RayWorkerP=...". That's what would have
# caught the awk line-split bug (a copy/paste mishap that left the Top field
# silently blank) without needing to know what a real run prints.

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../lib" && pwd)/sandbox.sh"

sandbox_run_test "report_memory_always_produces_a_single_populated_line" compute '
    create_status_pending "rm-job" "model" 30 > /dev/null 2>&1
    update_status_initialise "rm-job"

    out=$(report_memory "rm-job" "0")

    assert_contains "$out" "Cache:" "report_memory output" || exit 1
    assert_contains "$out" "RAM:" "report_memory output" || exit 1
    assert_contains "$out" "Top:" "report_memory output" || exit 1

    # The bug this guards against: an awk line split left everything after
    # "Top:" silently empty. Assert there is at least one non-whitespace
    # character following it, without caring what it is.
    top_field=$(echo "$out" | sed -n "s/.*Top: //p")
    [[ -n "${top_field// }" ]] || { echo "FAIL: Top: field is empty: $out"; exit 1; }

    # report_memory() no longer gates on any debug level at all — always
    # exactly one line, unconditionally.
    line_count=$(echo "$out" | wc -l)
    assert_eq "$line_count" "1" "report_memory line count" || exit 1
'

sandbox_run_test "report_gpu_produces_no_output_without_nvidia_smi" compute '
    create_status_pending "rm-job" "model" 30 > /dev/null 2>&1
    update_status_initialise "rm-job"

    # No nvidia-smi in this sandbox. Unlike report_processes()/py-spy below,
    # report_gpu() has no explicit "not installed" fallback message — it
    # just produces nothing when the command is absent. Documenting that
    # asymmetry here so it is a deliberate, tested fact, not a surprise.
    out=$(report_gpu "rm-job" "0")

    assert_eq "$out" "" "report_gpu output without nvidia-smi" || exit 1
'

sandbox_run_test "report_processes_reports_missing_py_spy" compute '
    create_status_pending "rm-job" "model" 30 > /dev/null 2>&1
    update_status_initialise "rm-job"

    # No py-spy in this sandbox — report_processes() should say so
    # explicitly rather than silently skip it.
    out=$(report_processes "rm-job" "0")

    assert_contains "$out" "py-spy" "report_processes output" || exit 1
    assert_contains "$out" "not installed" "report_processes output" || exit 1
'

exit "$FAIL"
