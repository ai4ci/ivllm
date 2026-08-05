#!/bin/bash
# shellcheck disable=SC2016
# tests/bash/sandboxed/test-report-memory.sh — report_memory() output shape
# and IVLLM_DEBUG_LEVEL gating.
#
# Deliberately loose: the exact Cache/RAM/Top formatting (and the GPU/py-spy
# detail levels layered on top of it) are expected to keep evolving as the
# multinode-hang debugging work continues, so these assertions pin to shape
# and gating behaviour, not exact content — e.g. "the Top field is non-empty"
# rather than "the Top field contains ray::RayWorkerP=...". That's what would
# have caught the awk line-split bug (a copy/paste mishap that left the Top
# field silently blank) without needing to know what a real run prints.

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../lib" && pwd)/sandbox.sh"

sandbox_run_test "report_memory_default_level_produces_populated_line" compute '
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

    # Debug levels default off — no GPU/py-spy lines at IVLLM_DEBUG_LEVEL=0.
    line_count=$(echo "$out" | wc -l)
    assert_eq "$line_count" "1" "line count at debug level 0" || exit 1
'

sandbox_run_test "report_memory_debug_level_1_stays_single_line_without_gpu" compute '
    create_status_pending "rm-job" "model" 30 > /dev/null 2>&1
    update_status_initialise "rm-job"

    # No nvidia-smi in this sandbox — level 1 should degrade to the same
    # single base line, not error out.
    export IVLLM_DEBUG_LEVEL=1
    out=$(report_memory "rm-job" "0")

    assert_contains "$out" "Cache:" "report_memory output" || exit 1
    line_count=$(echo "$out" | wc -l)
    assert_eq "$line_count" "1" "line count at debug level 1 without nvidia-smi" || exit 1
'

sandbox_run_test "report_memory_debug_level_2_reports_missing_py_spy" compute '
    create_status_pending "rm-job" "model" 30 > /dev/null 2>&1
    update_status_initialise "rm-job"

    # No py-spy in this sandbox either — level 2 should say so explicitly
    # rather than silently skip it.
    export IVLLM_DEBUG_LEVEL=2
    out=$(report_memory "rm-job" "0")

    assert_contains "$out" "py-spy" "report_memory output" || exit 1
    assert_contains "$out" "not installed" "report_memory output" || exit 1
'

exit "$FAIL"
