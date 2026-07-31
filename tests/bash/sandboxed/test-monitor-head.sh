#!/bin/bash
# shellcheck disable=SC2016
# tests/bash/sandboxed/test-monitor-head.sh — monitor_head() tests.
#
# This is the clearest demonstration of *why* the bash tests run inside a
# bwrap sandbox rather than plain subshells: these tests background real
# processes (a stand-in "vllm parent", a stand-in "vllm pid"), send real
# signals (SIGUSR2, SIGTERM), and rely on real `kill -0` liveness checks —
# exactly what monitor_head() itself does. Because each test runs inside its
# own bwrap --unshare-pid sandbox, any process a test backgrounds (including
# ones left behind by a failing/crashing test) is automatically reaped the
# moment the sandboxed script exits — there is no risk of leaking a stray
# `sleep` or monitor loop onto the host, and no dependency on trap-based
# cleanup inside every test.
#
# IVLLM_CHECK_INTERVAL_SECS defaults to 1 inside the sandbox (see
# tests/bash/lib/sandbox.sh), so monitor_head's polling loop reacts quickly
# enough for these tests to run in a few seconds each.

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../lib" && pwd)/sandbox.sh"

# Common setup: a "running" job, with a real background process standing in
# for the vLLM *parent* (the orchestrator subshell on the SLURM step host, in
# real usage) — monitor_head's ONLY process-liveness check is `kill -0` on
# this single pid; there is no separate per-vLLM-process pid tracked via the
# lockfile any more (that was removed when process orchestration moved to a
# single parent-subshell-per-job model — see slurm-vllm-serve.sh). It's a
# bare `sleep`; `kill -0` works on it and SIGUSR2/SIGTERM kill it (default
# disposition).
_SETUP_RUNNING_JOB='
    create_status_pending "mjob" "model" "${IDLE_TIMEOUT:-30}" > /dev/null 2>&1
    sleep 9999 &
    parent_pid=$!
    # The monitor requires the log file to exist (sanity check). We also prime
    # it with a RECENT API activity line so monitor_head'"'"'s idle-timeout
    # check doesn'"'"'t fire immediately — tests that specifically target idle
    # timeout override by writing their own log content below.
    log=$(resolve_job_log "mjob")
    echo "[$(date "+%Y-%m-%d %H:%M:%S")] INFO: 127.0.0.1:0 - \"POST /v1/chat/completions HTTP/1.1\" 200 OK" > "$log"
'

sandbox_run_test "monitor_head_detects_cancel" compute '
    '"$_SETUP_RUNNING_JOB"'
    update_status_initialise "mjob"
    update_status_running "mjob"

    monitor_head "mjob" "$parent_pid" &
    monitor_pid=$!

    sleep 2
    request_cancel "mjob"

    wait "$monitor_pid"
    rc=$?

    lockfile=$(resolve_job_status "mjob")
    assert_json_eq "$lockfile" ".reason" "user cancel" || exit 1
    assert_exit_code "$rc" 0 || exit 1
    kill -0 "$parent_pid" 2>/dev/null && { echo "FAIL: parent stand-in should have exited"; exit 1; }
    exit 0
'

# monitor_head'"'"'s main loop checks `kill -0 "$vllm_parent"` first, before any
# lockfile/idle-timeout logic — this guards that check directly by killing
# the parent stand-in out from under it, unprompted by cancel/idle-timeout.
# Unlike the cancel/lockfile-deletion paths, this branch does not itself
# write a .reason (in real usage tidy_up, running as the parent'"'"'s own exit
# trap, has already updated the lockfile by the time monitor_head notices) —
# so this test only asserts that monitor_head notices promptly and returns,
# rather than looping forever.
sandbox_run_test "monitor_head_detects_parent_death" compute '
    '"$_SETUP_RUNNING_JOB"'
    update_status_initialise "mjob"
    update_status_running "mjob"

    monitor_head "mjob" "$parent_pid" &
    monitor_pid=$!

    sleep 2
    kill -9 "$parent_pid" 2>/dev/null

    wait "$monitor_pid"
    rc=$?
    assert_exit_code "$rc" 0 || exit 1
    exit 0
'

sandbox_run_test "monitor_head_detects_lockfile_deletion" compute '
    '"$_SETUP_RUNNING_JOB"'
    update_status_initialise "mjob"
    update_status_running "mjob"
    lockfile=$(resolve_job_status "mjob")

    monitor_head "mjob" "$parent_pid" &
    monitor_pid=$!

    sleep 2
    rm -f "$lockfile"

    wait "$monitor_pid"
    rc=$?
    assert_exit_code "$rc" 1 || exit 1

    kill -0 "$parent_pid" 2>/dev/null && { echo "FAIL: parent stand-in should have been SIGTERMed"; exit 1; }
    exit 0
'

sandbox_run_test "monitor_head_idle_timeout_shuts_down" compute '
    IDLE_TIMEOUT=1
    '"$_SETUP_RUNNING_JOB"'
    update_status_initialise "mjob"
    update_status_running "mjob"
    # Log file exists (just created in _SETUP_RUNNING_JOB) but has no API
    # activity at all — should trigger an idle-timeout shutdown on the very
    # first check.
    log=$(resolve_job_log "mjob")
    echo "server started, no requests yet" > "$log"

    monitor_head "mjob" "$parent_pid" &
    monitor_pid=$!

    wait "$monitor_pid"

    lockfile=$(resolve_job_status "mjob")
    assert_json_eq "$lockfile" ".reason" "idle timeout" || exit 1
'

sandbox_run_test "monitor_head_active_traffic_prevents_idle_timeout" compute '
    IDLE_TIMEOUT=1
    '"$_SETUP_RUNNING_JOB"'
    update_status_initialise "mjob"
    update_status_running "mjob"
    log=$(resolve_job_log "mjob")
    # A log line timestamped *right now*, containing a real endpoint —
    # matches IVLLM_TARGET_ENDPOINTS and the current-minute time pattern, so
    # monitor_head should NOT decide to shut down.
    echo "[$(date "+%Y-%m-%d %H:%M:%S")] INFO: 127.0.0.1:0 - \"POST /v1/chat/completions HTTP/1.1\" 200 OK" > "$log"

    monitor_head "mjob" "$parent_pid" &
    monitor_pid=$!

    sleep 3
    is_status "mjob" "running" || { echo "FAIL: job should still be running (no idle timeout expected)"; exit 1; }
    kill -0 "$parent_pid" 2>/dev/null || { echo "FAIL: parent stand-in should not have been signalled"; exit 1; }

    # Clean up: request cancel so the backgrounded monitor exits before the
    # test script itself exits (not strictly required — bwrap would reap it
    # regardless — but keeps the test tidy and its exit code meaningful).
    request_cancel "mjob"
    wait "$monitor_pid"
'

exit "$FAIL"
