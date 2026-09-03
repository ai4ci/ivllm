#!/bin/bash
# shellcheck disable=SC2016
# tests/bash/sandboxed/test-diagnostics.sh — capture_job_diagnostics() and
# its routing through tidy_up()'s exit-code branches.
#
# Covers the two halves of "diagnostics aren't captured on crash" (see
# design/active-issues.md, resolved wait_report entry, for the full
# root-cause writeup):
#   1. capture_job_diagnostics() itself — does it actually copy the job's
#      artifacts into a timestamped diagnostics directory. There is no
#      group-writability test here: $job_dir is only ever populated with
#      files ivllm itself generates on the HPC side (unlike an older
#      architecture where slurm scripts were copied in from the local
#      client), so under this project's umask 0002 + setgid dirs, the
#      general permissions invariant already covers it — no explicit
#      chmod after the copy is needed (see design/active-issues.md).
#   2. tidy_up()'s routing — a genuine crash (nonzero exit) captures
#      diagnostics; a clean idle/cancel/SLURM-timeout shutdown (200/201/202)
#      does not, by design. An exit code of 0 is now ALSO treated as a
#      failure (see design decision: vLLM should never exit on its own
#      outside of an explicit monitor/SLURM signal) and DOES capture
#      diagnostics. This matters because monitor_head's idle-timeout path
#      always signals via a specific code (201/202) — before the wait_report
#      fix, a crash could get silently misrouted through there and lose its
#      diagnostics. Now that wait_report correctly detects and propagates a
#      real crash exit code, tidy_up receives it directly via the fast
#      exit-trap path instead.

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../lib" && pwd)/sandbox.sh"

# ── capture_job_diagnostics() directly ─────────────────────────────────

sandbox_run_test "capture_job_diagnostics_copies_artifacts" compute '
    create_status_pending "diag-job" "model" 30 > /dev/null 2>&1
    update_status_initialise "diag-job"
    job_dir=$(resolve_job_dir "diag-job")

    echo "log line one" > "$job_dir/vllm.0.log"
    echo "model: test" > "$job_dir/vllm.yaml"
    echo "model: test" > "$job_dir/vllm.yaml.clean.yaml"

    capture_job_diagnostics "diag-job"

    diag_root="$IVLLM_PROJECTDIR/engine/diagnostics/diag-job"
    assert_dir_exists "$diag_root" || exit 1

    # Exactly one timestamped subdirectory should have been created.
    diag_dir=$(find "$diag_root" -mindepth 1 -maxdepth 1 -type d | head -n1)
    [[ -n "$diag_dir" ]] || { echo "FAIL: no timestamped diagnostics dir found"; exit 1; }

    assert_file_exists "$diag_dir/vllm.0.log" || exit 1
    assert_file_exists "$diag_dir/vllm.yaml" || exit 1
    assert_file_exists "$diag_dir/vllm.yaml.clean.yaml" || exit 1
    assert_file_exists "$diag_dir/status.json" || exit 1

    content=$(cat "$diag_dir/vllm.0.log")
    [[ "$content" == "log line one" ]] || { echo "FAIL: log content mismatch: $content"; exit 1; }
'

# ── Routing through tidy_up() ───────────────────────────────────────────

_setup_running_job='
    create_status_pending "tidy-diag-job" "model" 30 > /dev/null 2>&1
    update_status_initialise "tidy-diag-job"
    update_status_running "tidy-diag-job"
    job_dir=$(resolve_job_dir "tidy-diag-job")
    echo "log line" > "$job_dir/vllm.0.log"
    sleep 9999 &
    _VLLM_PID=$!
'

sandbox_run_test "tidy_up_crash_captures_diagnostics" compute '
    '"$_setup_running_job"'
    tidy_up "tidy-diag-job" 42 "$_VLLM_PID"

    diag_root="$IVLLM_PROJECTDIR/engine/diagnostics/tidy-diag-job"
    assert_dir_exists "$diag_root" || { echo "FAIL: expected diagnostics to be captured on crash"; exit 1; }
'

sandbox_run_test "tidy_up_idle_timeout_does_not_capture_diagnostics" compute '
    '"$_setup_running_job"'
    tidy_up "tidy-diag-job" 201 "$_VLLM_PID"

    diag_root="$IVLLM_PROJECTDIR/engine/diagnostics/tidy-diag-job"
    assert_dir_exists "$diag_root" 2>/dev/null && { echo "FAIL: idle/cancel shutdown (201) should not capture diagnostics"; exit 1; }
    exit 0
'

sandbox_run_test "tidy_up_unsolicited_exit_captures_diagnostics" compute '
    '"$_setup_running_job"'
    tidy_up "tidy-diag-job" 0 "$_VLLM_PID"

    diag_root="$IVLLM_PROJECTDIR/engine/diagnostics/tidy-diag-job"
    assert_dir_exists "$diag_root" || { echo "FAIL: expected diagnostics to be captured for an unsolicited exit (0)"; exit 1; }
'

exit "$FAIL"
