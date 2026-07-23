#!/bin/bash
# tests/bash/sandboxed/test-exit-trap.sh — Exit-trap / tidy_up() tests.
#
# Tests the `tidy_up()` function, which is called as an exit trap on ALL
# nodes when a job shuts down. It kills the vLLM process (if alive),
# updates the lockfile based on exit code, and cancels the SLURM job.
#
# Uses real background processes for the fake vLLM pid so that `kill -0`
# liveness checks exercise real subprocess semantics. Uses the `compute`
# profile sandbox (SLURM_* env set) because tidy_up reads slurmJobId from
# the lockfile and uses SLURM_NODEID gating.

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../lib" && pwd)/sandbox.sh"

# ── Helper: create a running job with a real background "vllm" process ──
_setup_running_job='
    create_status_pending "tidy-job" "model" 30 > /dev/null 2>&1
    update_status_initialise "tidy-job" 0
    update_status_running "tidy-job"
    sleep 9999 &
    _VLLM_PID=$!
    lf=$(resolve_job_status "tidy-job")
    jq ".vllmPid = $_VLLM_PID" "$lf" > /tmp/sj.json
    mv /tmp/sj.json "$lf"
'

# ── Test: exit_code=200 (SIGUSR1 — SLURM timeout) ──────────────────────
sandbox_run_test "tidy_up_200_triggers_slurm_timeout" compute '
    '"$_setup_running_job"'
    tidy_up "tidy-job" 200
    lockfile=$(resolve_job_status "tidy-job")
    assert_status "$lockfile" "stopped" || exit 1
    assert_json_eq "$lockfile" ".reason" "SLURM timeout" || exit 1
    exit 0
'

# ── Test: exit_code=201 (SIGUSR2 — user cancel or idle timeout) ────────
sandbox_run_test "tidy_up_201_triggers_user_cancel" compute '
    '"$_setup_running_job"'
    tidy_up "tidy-job" 201
    lockfile=$(resolve_job_status "tidy-job")
    assert_status "$lockfile" "stopped" || exit 1
    # 201 calls update_status_stopped without update_reason — no .reason set
    exit 0
'

# ── Test: exit_code=0 (normal shutdown — status unchanged) ─────────────
sandbox_run_test "tidy_up_0_normal_shutdown" compute '
    '"$_setup_running_job"'
    tidy_up "tidy-job" 0
    lockfile=$(resolve_job_status "tidy-job")
    assert_status "$lockfile" "running" || exit 1
    kill -0 "$_VLLM_PID" 2>/dev/null && { echo "FAIL: vLLM killed"; exit 1; }
    exit 0
'

# ── Test: exit_code=42 while running → failed ("crashed during inference") ──
sandbox_run_test "tidy_up_nonzero_crash_runtime" compute '
    '"$_setup_running_job"'
    tidy_up "tidy-job" 42
    lockfile=$(resolve_job_status "tidy-job")
    assert_status "$lockfile" "failed" || exit 1
    assert_json_eq "$lockfile" ".reason" "crashed during inference" || exit 1
    assert_json_eq "$lockfile" ".exitCode" "42" || exit 1
    kill -0 "$_VLLM_PID" 2>/dev/null && { echo "FAIL: vLLM killed"; exit 1; }
    exit 0
'

# ── Test: exit_code=1 while initialising → failed ("failed to start") ──
sandbox_run_test "tidy_up_nonzero_crash_startup" compute '
    create_status_pending "startup-crash" "model" 30 > /dev/null 2>&1
    update_status_initialise "startup-crash" 99999
    sleep 9999 &
    _VLLM_PID=$!
    lf=$(resolve_job_status "startup-crash")
    jq ".vllmPid = $_VLLM_PID" "$lf" > /tmp/sc.json
    mv /tmp/sc.json "$lf"
    tidy_up "startup-crash" 1
    lockfile=$(resolve_job_status "startup-crash")
    assert_status "$lockfile" "failed" || exit 1
    assert_json_eq "$lockfile" ".reason" "failed to start" || exit 1
    assert_json_eq "$lockfile" ".exitCode" "1" || exit 1
    exit 0
'

# ── Test: tidy_up kills vLLM process (SIGTERM then SIGKILL) ────────────
sandbox_run_test "tidy_up_kills_vllm_process" compute '
    '"$_setup_running_job"'
    kill -0 "$_VLLM_PID" 2>/dev/null || { echo "FAIL: vLLM should be alive"; exit 1; }
    tidy_up "tidy-job" 0
    kill -0 "$_VLLM_PID" 2>/dev/null && { echo "FAIL: vLLM should have been killed"; exit 1; }
    exit 0
'

# ── Test: tidy_up calls scancel with slurmJobId ────────────────────────
sandbox_run_test "tidy_up_cancels_slurm_job" compute '
    '"$_setup_running_job"'
    tidy_up "tidy-job" 0
    assert_shim_called "scancel" "99999" || exit 1
    exit 0
'

exit "$FAIL"
