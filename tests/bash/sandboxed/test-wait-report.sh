#!/bin/bash
# shellcheck disable=SC2016
# tests/bash/sandboxed/test-wait-report.sh — wait_report() exit-code
# propagation.
#
# wait_report() is the LAST command in run_head_vllm.sh; its own exit status
# becomes the script's exit status, which srun mirrors back to the SLURM
# step host, which is what setup_traps' EXIT trap (tidy_up) receives as the
# real crash/exit code (see slurm-vllm-serve.sh's `wait_all` + `setup_traps`
# and utils.sh's `tidy_up`).
#
# wait_report detects the monitored process's death via process_died()
# (a /proc/$pid/stat state check, immune to the classic "kill -0 succeeds on
# a zombie" bug), then does a real `wait "$pid"` to reap it and capture its
# actual exit code.
#
# There is no legitimate reason for the monitored vLLM process to exit on
# its own — user cancel/idle-timeout/SLURM-timeout are all delivered as
# signals from monitor_head/SLURM, handled by run_head_vllm.sh's own signal
# trap, not by the process quietly returning. So wait_report treats ANY
# process exit as an error: a genuine crash (nonzero) propagates its real
# exit code; an unsolicited clean exit (0) is itself treated as anomalous
# and reported as a generic failure (1), never as 0.

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../lib" && pwd)/sandbox.sh"

sandbox_run_test "wait_report_propagates_crash_exit_code" compute '
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

sandbox_run_test "wait_report_treats_unsolicited_clean_exit_as_failure" compute '
    create_status_pending "wr-job" "model" 30 > /dev/null 2>&1
    update_status_initialise "wr-job"

    # vLLM exiting 0 on its own (no signal, no monitor request) is always
    # unexpected — wait_report reports it as a generic failure (1), not 0.
    ( exit 0 ) &
    pid=$!

    wait_report "wr-job" "$pid"
    rc=$?

    assert_exit_code "$rc" 1 || exit 1
'

exit "$FAIL"
