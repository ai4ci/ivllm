#!/bin/bash
# shellcheck disable=SC2016
# tests/bash/sandboxed/test-wait-report.sh — wait_report() exit-code
# propagation (or lack of it).
#
# wait_report() is the LAST command in run_head_vllm.sh; its own exit status
# becomes the script's exit status, which srun mirrors back to the SLURM
# step host, which is what setup_traps' EXIT trap (tidy_up) receives as the
# real crash/exit code (see slurm-vllm-serve.sh's `wait_all` + `setup_traps`
# and utils.sh's `tidy_up`).
#
# wait_report's own loop only ever does `kill -0 "$pid"` — never `wait
# "$pid"`. `$pid` here is a DIRECT CHILD of this same shell (backgrounded
# with `&`), so once it exits it becomes a zombie: `kill -0` still succeeds
# on a zombie (the PID stays allocated until the parent reaps it), and
# nothing in wait_report (or run_head_vllm.sh) ever calls `wait` to reap it
# or capture its real exit code. In practice this means wait_report can
# never reliably notice — let alone propagate — the monitored process's
# real exit status.
#
# EXPECTED TO CURRENTLY FAIL (both tests below) — see design/active-issues.md
# for the full root-cause writeup: this is the reason a genuine vLLM crash
# ends up mis-routed through monitor_head's idle-timeout path (exit code
# 201) instead of tidy_up's crash branch, which is why diagnostics don't get
# captured. Not fixed yet — pre-existing bug, left for the user to apply the
# proposed fix (documented in active-issues.md) when ready.

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../lib" && pwd)/sandbox.sh"

# A short timeout — wait_report's kill -0 loop is expected to spin forever
# on a zombie direct child, so both tests below are expected to time out
# (rather than hang for the default 30s sandbox ceiling).
SANDBOX_TIMEOUT_SECS=6 sandbox_run_test "wait_report_propagates_crash_exit_code" compute '
    create_status_pending "wr-job" "model" 30 > /dev/null 2>&1
    update_status_initialise "wr-job"

    # Stand-in for a crashing "vllm serve" — a direct child backgrounded
    # exactly like run_head_vllm.sh does (`vllm serve ... & IVLLM_PID=$!`).
    ( exit 17 ) &
    pid=$!

    wait_report "wr-job" "$pid"
    rc=$?

    assert_exit_code "$rc" 17 || exit 1
'

SANDBOX_TIMEOUT_SECS=6 sandbox_run_test "wait_report_propagates_clean_exit_code" compute '
    create_status_pending "wr-job" "model" 30 > /dev/null 2>&1
    update_status_initialise "wr-job"

    ( exit 0 ) &
    pid=$!

    wait_report "wr-job" "$pid"
    rc=$?

    assert_exit_code "$rc" 0 || exit 1
'

exit "$FAIL"
