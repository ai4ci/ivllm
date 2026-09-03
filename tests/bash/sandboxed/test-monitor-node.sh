#!/bin/bash
# shellcheck disable=SC2016
# tests/bash/sandboxed/test-monitor-node.sh — monitor_node() exit-code
# propagation.
#
# monitor_node() (formerly wait_report(), renamed in the "refactor debugging"
# commit — it also gained trigger-file watching, py-spy/cuda-gdb capture,
# and a background node_hang_detector along the way, but the exit-code
# contract this test checks is unchanged, confirmed 2026-09-03) is the LAST
# command in run_head_vllm.sh/run_worker_vllm.sh; its own exit status
# becomes the script's exit status, which srun mirrors back to the SLURM
# step host, which is what setup_traps' EXIT trap (tidy_up) receives as the
# real crash/exit code (see slurm-vllm-serve.sh's `wait_all` + `setup_traps`
# and utils.sh's `tidy_up`).
#
# monitor_node detects the monitored process's death via process_died()
# (a /proc/$pid/stat state check, immune to the classic "kill -0 succeeds on
# a zombie" bug), then does a real `wait "$pid"` to reap it and capture its
# actual exit code.
#
# There is no legitimate reason for the monitored vLLM (or Ray head)
# process to exit on its own — user cancel/idle-timeout/SLURM-timeout are
# all delivered as signals from monitor_head/SLURM, handled by the launch
# script's own signal trap, not by the process quietly returning. So
# monitor_node treats ANY process exit as an error: a genuine crash
# (nonzero) propagates its real exit code; an unsolicited clean exit (0) is
# itself treated as anomalous and reported as a generic failure (1), never
# as 0.

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../lib" && pwd)/sandbox.sh"

sandbox_run_test "monitor_node_propagates_crash_exit_code" compute '
    create_status_pending "wr-job" "model" 30 > /dev/null 2>&1
    update_status_initialise "wr-job"

    # Stand-in for a crashing "vllm serve" (or ray head) — a direct child
    # backgrounded exactly like run_head_vllm.sh does
    # (`vllm serve ... & IVLLM_HEAD_NODE_PID=$!`).
    ( exit 17 ) &
    pid=$!

    monitor_node "wr-job" "$pid"
    rc=$?

    assert_exit_code "$rc" 17 || exit 1
'

sandbox_run_test "monitor_node_treats_unsolicited_clean_exit_as_failure" compute '
    create_status_pending "wr-job" "model" 30 > /dev/null 2>&1
    update_status_initialise "wr-job"

    # vLLM (or ray) exiting 0 on its own (no signal, no monitor request) is
    # always unexpected — monitor_node reports it as a generic failure (1),
    # not 0.
    ( exit 0 ) &
    pid=$!

    monitor_node "wr-job" "$pid"
    rc=$?

    assert_exit_code "$rc" 1 || exit 1
'

exit "$FAIL"
