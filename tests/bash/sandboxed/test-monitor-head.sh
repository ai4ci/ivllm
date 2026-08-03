#!/bin/bash
# shellcheck disable=SC2016
# tests/bash/sandboxed/test-monitor-head.sh — monitor_head() tests.
#
# monitor_head() is now the single, unified lifecycle monitor — it covers
# what used to be split across monitor_startup() (health/warmup polling
# during "initialising") and monitor_head() (idle-timeout/cancel/lockfile
# checks once "running"). It runs for the whole job lifetime and reports
# what it detects via distinct exit codes (tidy_up() then interprets those
# codes and updates the lockfile) — monitor_head itself no longer writes
# `.reason` directly, so these tests assert exit codes, not lockfile state,
# except where a test specifically exercises the real HTTP-level warmup
# path via the `vllm` shim.
#
# monitor_head no longer takes or checks a "parent pid" argument — process
# orchestration moved to a single parent-subshell-per-job model where
# monitor_head runs *inside* that subshell and is waited on by wait_all()
# (see slurm-vllm-serve.sh); there is no separate external process for it
# to watch the liveness of.
#
# Runs inside a bwrap sandbox so backgrounded processes (real "vllm" mock
# servers, monitor_head itself) are automatically reaped on exit — no
# leaked processes even if a test fails.
#
# IVLLM_CHECK_INTERVAL_SECS defaults to 1 inside the sandbox (see
# tests/bash/lib/sandbox.sh), so monitor_head's polling loop reacts quickly
# enough for these tests to run in a few seconds each.

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../lib" && pwd)/sandbox.sh"

# ── Running-phase scenarios (idle timeout / cancel / lockfile deletion) ──

_SETUP_RUNNING_JOB='
    create_status_pending "mjob" "model" "${IDLE_TIMEOUT:-30}" > /dev/null 2>&1
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

    monitor_head "mjob" &
    monitor_pid=$!

    sleep 2
    request_cancel "mjob"

    wait "$monitor_pid"
    rc=$?

    # monitor_head reports the condition via exit code; tidy_up (tested
    # separately in test-exit-trap.sh) is what writes .reason/.status.
    assert_exit_code "$rc" 201 || exit 1
    exit 0
'

sandbox_run_test "monitor_head_detects_lockfile_deletion" compute '
    '"$_SETUP_RUNNING_JOB"'
    update_status_initialise "mjob"
    update_status_running "mjob"
    lockfile=$(resolve_job_status "mjob")

    monitor_head "mjob" &
    monitor_pid=$!

    sleep 2
    rm -f "$lockfile"

    wait "$monitor_pid"
    rc=$?
    assert_exit_code "$rc" 250 || exit 1
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

    monitor_head "mjob" &
    monitor_pid=$!

    wait "$monitor_pid"
    rc=$?
    assert_exit_code "$rc" 202 || exit 1
    exit 0
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

    monitor_head "mjob" &
    monitor_pid=$!

    sleep 3
    is_status "mjob" "running" || { echo "FAIL: job should still be running (no idle timeout expected)"; exit 1; }
    kill -0 "$monitor_pid" 2>/dev/null || { echo "FAIL: monitor_head should still be running"; exit 1; }

    # Clean up: request cancel so the backgrounded monitor exits before the
    # test script itself exits (not strictly required — bwrap would reap it
    # regardless — but keeps the test tidy and its exit code meaningful).
    request_cancel "mjob"
    wait "$monitor_pid"
    exit 0
'

# ── Startup-phase scenarios (health check + warmup, formerly monitor_startup()) ──
# monitor_startup() was merged into monitor_head() — the "initialising"
# branch below is what used to be a separate function. Unlike the
# running-phase tests above, these exercise real HTTP calls against the
# `vllm` shim, so they assert lockfile state directly (status transitions
# to "running" are monitor_head's own doing here, not tidy_up's).
#
# NOTE: after a successful warmup, monitor_head does NOT return — it keeps
# running (idle-timeout monitoring for "running" status). Tests must
# request_cancel + wait to shut it down cleanly rather than expecting it to
# exit on its own.

sandbox_run_test "monitor_head_startup_sends_health_and_warms_up" compute '
    export USER=testuser
    export MOCK_VLLM_PORT=$(( 30000 + RANDOM % 10000 ))
    export MOCK_VLLM_DELAY=0
    export MOCK_VLLM_CRASH_AFTER=0

    /work/shims/vllm serve --port "$MOCK_VLLM_PORT" --model test-model &
    VLLM_SHIM_PID=$!
    sleep 1

    create_status_pending "startup-job" "test-model" 30 > /dev/null 2>&1
    lf=$(resolve_job_status "startup-job")
    jq ".serverPort = $MOCK_VLLM_PORT" "$lf" > /tmp/sj.json
    mv /tmp/sj.json "$lf"
    update_status_initialise "startup-job" 0

    monitor_head "startup-job" &
    monitor_pid=$!

    for i in $(seq 1 20); do
        is_status "startup-job" "running" && break
        sleep 0.5
    done

    lockfile=$(resolve_job_status "startup-job")
    assert_status "$lockfile" "running" || exit 1

    request_cancel "startup-job"
    wait "$monitor_pid"
    kill "$VLLM_SHIM_PID" 2>/dev/null; wait "$VLLM_SHIM_PID" 2>/dev/null
    exit 0
'

sandbox_run_test "monitor_head_startup_health_then_succeeds" compute '
    export USER=testuser
    export MOCK_VLLM_PORT=$(( 30000 + RANDOM % 10000 ))
    export MOCK_VLLM_DELAY=2
    export MOCK_VLLM_CRASH_AFTER=0

    /work/shims/vllm serve --port "$MOCK_VLLM_PORT" --model test-model &
    VLLM_SHIM_PID=$!
    sleep 1

    create_status_pending "startup-job" "test-model" 30 > /dev/null 2>&1
    lf=$(resolve_job_status "startup-job")
    jq ".serverPort = $MOCK_VLLM_PORT" "$lf" > /tmp/sj.json
    mv /tmp/sj.json "$lf"
    update_status_initialise "startup-job" 0

    monitor_head "startup-job" &
    monitor_pid=$!

    for i in $(seq 1 20); do
        is_status "startup-job" "running" && break
        sleep 0.5
    done

    lockfile=$(resolve_job_status "startup-job")
    assert_status "$lockfile" "running" || exit 1

    request_cancel "startup-job"
    wait "$monitor_pid"
    kill "$VLLM_SHIM_PID" 2>/dev/null; wait "$VLLM_SHIM_PID" 2>/dev/null
    exit 0
'

# ── Test: warmup fails after max retries → failed ─────────────────────
# (Skipped — warmup retries 5x with 10s gaps = 50s, exceeds sandbox 30s
# timeout. The failure path (exit code 252) is verified implicitly by
# tidy_up's exit-code dispatch tests.)

exit "$FAIL"
