#!/bin/bash
# shellcheck disable=SC2016
# tests/bash/sandboxed/test-monitor-startup.sh — monitor_startup() tests.
#
# Tests the `monitor_startup()` function — the foreground monitor that runs
# on the head node during initialisation. Blocks until vLLM responds to
# `/health`, sends a warmup request, saves the JIT cache, and transitions
# status to `running`.
#
# Uses a real mock vLLM HTTP server (the `vllm` shim) so that `curl`
# /health and /v1/chat/completions calls exercise real HTTP semantics.
#
# NOTE: report_memory() at utils.sh:864 uses `$USER` which is unset in the
# bwrap sandbox (clearenv). We export USER in every test body before
# calling monitor_startup so report_memory doesn't crash under set -u.

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../lib" && pwd)/sandbox.sh"

# ── Test: happy path — health succeeds, warmup succeeds → running ─────
sandbox_run_test "startup_sends_health_and_warms_up" compute '
    export USER=testuser
    export MOCK_VLLM_PORT=$(( 30000 + RANDOM % 10000 ))
    export MOCK_VLLM_DELAY=0
    export MOCK_VLLM_CRASH_AFTER=0

    # Start mock vLLM in background
    /work/shims/vllm serve --port "$MOCK_VLLM_PORT" --model test-model &
    VLLM_SHIM_PID=$!
    sleep 1

    # Create job, set serverPort to mock vLLM
    create_status_pending "startup-job" "test-model" 30 > /dev/null 2>&1
    lf=$(resolve_job_status "startup-job")
    jq ".serverPort = $MOCK_VLLM_PORT" "$lf" > /tmp/sj.json
    mv /tmp/sj.json "$lf"
    update_status_initialise "startup-job" 0

    monitor_startup "startup-job" $$
    rc=$?

    lockfile=$(resolve_job_status "startup-job")
    assert_status "$lockfile" "running" || exit 1
    kill "$VLLM_SHIM_PID" 2>/dev/null; wait "$VLLM_SHIM_PID" 2>/dev/null
    exit 0
'

# ── Test: health check retries on first failure (delay=2s) ─────────────
sandbox_run_test "startup_health_then_succeeds" compute '
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

    monitor_startup "startup-job" $$
    rc=$?

    lockfile=$(resolve_job_status "startup-job")
    assert_status "$lockfile" "running" || exit 1
    kill "$VLLM_SHIM_PID" 2>/dev/null; wait "$VLLM_SHIM_PID" 2>/dev/null
    exit 0
'

# ── Test: warmup fails after max retries → failed ─────────────────────
# (Skipped — warmup retries 5× with 10s gaps = 50s, exceeds sandbox 30s
# timeout. The failure path is verified implicitly by exit-trap tests.)

# ── Test: non-head node (SLURM_NODEID=1) returns immediately ──────────
sandbox_run_test "startup_non_head_node" compute '
    # Create lockfile on head node first
    create_status_pending "nonhead-job" "model" 30 > /dev/null 2>&1

    # Now switch to worker node and call monitor_startup
    export SLURM_NODEID=1
    monitor_startup "nonhead-job" $$
    rc=$?
    export SLURM_NODEID=0

    lockfile=$(resolve_job_status "nonhead-job")
    # Should stay in pending — monitor_startup returns 0 without checking
    assert_status "$lockfile" "pending" || exit 1
    [ "$rc" -eq 0 ] || { echo "FAIL: expected exit 0, got $rc"; exit 1; }
    exit 0
'

# ── Test: wrong status on lockfile → returns 1 ────────────────────────
sandbox_run_test "startup_wrong_status" compute '
    export USER=testuser
    create_status_pending "wrongstatus-job" "model" 30 > /dev/null 2>&1
    # Skip initialise — go straight to running so monitor_startup sees
    # the "else" branch (unexpected state)
    update_status_running "wrongstatus-job"

    monitor_startup "wrongstatus-job" $$
    rc=$?

    [ "$rc" -eq 1 ] || { echo "FAIL: expected exit 1 for wrong status, got $rc"; exit 1; }
    exit 0
'

exit "$FAIL"
