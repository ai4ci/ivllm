#!/bin/bash
# tests/bash/sandboxed/test-monitor-worker.sh — monitor_worker() tests.
#
# Tests the `monitor_worker()` function — the background monitor that runs
# on worker nodes (node ID > 0) in multi-node jobs. Watches the lockfile
# and shuts down the local vLLM process if the job is no longer running.
#
# IMPORTANT: bwrap sets SLURM_NODEID=0 for the compute profile by default.
# The tests must simulate multi-node by:
#   1. Using SLURM_NODEID=0 (default) to create the lockfile as head node
#   2. Exporting SLURM_NODEID=1+ inside the test body to switch to worker
#   3. Calling monitor_worker WITH SLURM_NODEID still set to 1+

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../lib" && pwd)/sandbox.sh"

# --- Test: rejects head node (SLURM_NODEID=0) ----------------------------
sandbox_run_test "worker_monitor_rejects_head_node" compute '
    # This IS the head node -- calling monitor_worker should fail
    sleep 9999 &
    _WORKER_VLLM_PID=$!
    monitor_worker "reject-job" "$_WORKER_VLLM_PID"
    rc=$?

    [ "$rc" -eq 1 ] || { echo "FAIL: expected exit 1 for head node, got $rc"; exit 1; }
    sleep 0.5
    kill -0 "$_WORKER_VLLM_PID" 2>/dev/null && { echo "FAIL: worker pid should have been killed"; exit 1; }
    exit 0
'

# --- Test: missing lockfile on startup -----------------------------------
sandbox_run_test "worker_monitor_missing_lockfile" compute '
    # Create lockfile on head, then delete it to simulate missing
    create_status_pending "ghost-job" "model" 30 > /dev/null 2>&1
    rm -f "$(resolve_job_status "ghost-job")"
    sleep 9999 &
    _WORKER_VLLM_PID=$!
    export SLURM_NODEID=1
    monitor_worker "ghost-job" "$_WORKER_VLLM_PID"
    rc=$?

    [ "$rc" -eq 1 ] || { echo "FAIL: expected exit 1 for missing lockfile, got $rc"; exit 1; }
    sleep 0.5
    kill -0 "$_WORKER_VLLM_PID" 2>/dev/null && { echo "FAIL: worker pid should have been killed"; exit 1; }
    exit 0
'

# --- Test: stays alive while status=running --------------------------------
sandbox_run_test "worker_monitor_stays_alive_running" compute '
    # Head node: create lockfile
    create_status_pending "worker-job" "model" 30 > /dev/null 2>&1
    update_status_initialise "worker-job" 0
    update_status_running "worker-job"

    # Worker: background vLLM + monitor
    sleep 9999 &
    _WORKER_VLLM_PID=$!
    export SLURM_NODEID=1
    monitor_worker "worker-job" "$_WORKER_VLLM_PID" &
    MON_PID=$!

    sleep 2

    kill -0 "$MON_PID" 2>/dev/null || { echo "FAIL: worker monitor should still be running"; exit 1; }
    kill -0 "$_WORKER_VLLM_PID" 2>/dev/null || { echo "FAIL: worker vllm should still be running"; exit 1; }

    kill "$MON_PID" 2>/dev/null
    kill "$_WORKER_VLLM_PID" 2>/dev/null
    exit 0
'

# --- Test: shuts down when status changes to cancel ----------------------
sandbox_run_test "worker_monitor_shuts_down_on_cancel" compute '
    create_status_pending "worker-job" "model" 30 > /dev/null 2>&1
    update_status_initialise "worker-job" 0
    update_status_running "worker-job"

    sleep 9999 &
    _WORKER_VLLM_PID=$!
    export SLURM_NODEID=1
    monitor_worker "worker-job" "$_WORKER_VLLM_PID" &
    MON_PID=$!

    sleep 2
    request_cancel "worker-job"
    wait "$MON_PID"

    sleep 0.5
    kill -0 "$_WORKER_VLLM_PID" 2>/dev/null && { echo "FAIL: worker pid should have been killed"; exit 1; }
    exit 0
'

# --- Test: shuts down when status changes to failed ----------------------
sandbox_run_test "worker_monitor_shuts_down_on_failed" compute '
    create_status_pending "worker-job" "model" 30 > /dev/null 2>&1
    update_status_initialise "worker-job" 0
    update_status_running "worker-job"

    sleep 9999 &
    _WORKER_VLLM_PID=$!
    export SLURM_NODEID=1
    monitor_worker "worker-job" "$_WORKER_VLLM_PID" &
    MON_PID=$!

    sleep 2
    # update_status_failed is head-node-only (SLURM_NODEID==0). Use
    # request_cancel which works from any node to change status.
    request_cancel "worker-job"
    wait "$MON_PID"

    sleep 0.5
    kill -0 "$_WORKER_VLLM_PID" 2>/dev/null && { echo "FAIL: worker pid should have been killed"; exit 1; }
    exit 0
'

exit "$FAIL"
