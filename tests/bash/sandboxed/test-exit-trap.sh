#!/bin/bash
# shellcheck disable=SC2016
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
    update_status_initialise "tidy-job"
    update_status_running "tidy-job"
    sleep 9999 &
    _VLLM_PID=$!
'

# ── Test: exit_code=200 (SIGUSR1 — SLURM timeout) ──────────────────────
sandbox_run_test "tidy_up_200_triggers_slurm_timeout" compute '
    '"$_setup_running_job"'
    tidy_up "tidy-job" 200 "$_VLLM_PID"
    lockfile=$(resolve_job_status "tidy-job")
    assert_status "$lockfile" "stopped" || exit 1
    assert_json_eq "$lockfile" ".reason" "SLURM job cancelled" || exit 1
    kill -0 "$_VLLM_PID" 2>/dev/null && { echo "FAIL: vLLM should have been killed"; exit 1; }
    exit 0
'

# ── Test: exit_code=201 (SIGUSR2 — user cancel or idle timeout) ────────
sandbox_run_test "tidy_up_201_triggers_user_cancel" compute '
    '"$_setup_running_job"'
    tidy_up "tidy-job" 201 "$_VLLM_PID"
    lockfile=$(resolve_job_status "tidy-job")
    assert_status "$lockfile" "stopped" || exit 1
    # 201 calls update_status_stopped without update_reason — no .reason set
    kill -0 "$_VLLM_PID" 2>/dev/null && { echo "FAIL: vLLM should have been killed"; exit 1; }
    exit 0
'

# ── Test: exit_code=0 (vllm exited on its own, unsolicited) → failure ──
# There is no legitimate reason for vLLM to exit on its own: user cancel,
# idle timeout, and SLURM timeout all route through explicit codes
# (201/202/200). A raw 0 reaching tidy_up via the bottom-up exit-code path
# means the process died without anyone asking it to — always a failure.
sandbox_run_test "tidy_up_0_is_treated_as_failure" compute '
    '"$_setup_running_job"'
    tidy_up "tidy-job" 0 "$_VLLM_PID"
    lockfile=$(resolve_job_status "tidy-job")
    assert_status "$lockfile" "failed" || exit 1
    assert_json_eq "$lockfile" ".reason" "unexpected termination" || exit 1
    kill -0 "$_VLLM_PID" 2>/dev/null && { echo "FAIL: vLLM should have been killed"; exit 1; }
    exit 0
'

# ── Test: exit_code=42 while running → failed ("crashed after startup") ──
sandbox_run_test "tidy_up_nonzero_crash_runtime" compute '
    '"$_setup_running_job"'
    tidy_up "tidy-job" 42 "$_VLLM_PID"
    lockfile=$(resolve_job_status "tidy-job")
    assert_status "$lockfile" "failed" || exit 1
    assert_json_eq "$lockfile" ".reason" "crashed after startup" || exit 1
    assert_json_eq "$lockfile" ".exitCode" "42" || exit 1
    kill -0 "$_VLLM_PID" 2>/dev/null && { echo "FAIL: vLLM should have been killed"; exit 1; }
    exit 0
'

# ── Test: exit_code=1 while initialising → failed ("didn't start") ──
sandbox_run_test "tidy_up_nonzero_crash_startup" compute '
    create_status_pending "startup-crash" "model" 30 > /dev/null 2>&1
    update_status_initialise "startup-crash"
    sleep 9999 &
    _VLLM_PID=$!
    tidy_up "startup-crash" 1 "$_VLLM_PID"
    lockfile=$(resolve_job_status "startup-crash")
    assert_status "$lockfile" "failed" || exit 1
    assert_json_eq "$lockfile" ".reason" "didn'"'"'t start" || exit 1
    assert_json_eq "$lockfile" ".exitCode" "1" || exit 1
    kill -0 "$_VLLM_PID" 2>/dev/null && { echo "FAIL: vLLM should have been killed"; exit 1; }
    exit 0
'

# ── Test: tidy_up kills vLLM process (SIGTERM then SIGKILL) ────────────
sandbox_run_test "tidy_up_kills_vllm_process" compute '
    '"$_setup_running_job"'
    kill -0 "$_VLLM_PID" 2>/dev/null || { echo "FAIL: vLLM should be alive"; exit 1; }
    tidy_up "tidy-job" 0 "$_VLLM_PID"
    kill -0 "$_VLLM_PID" 2>/dev/null && { echo "FAIL: vLLM should have been killed"; exit 1; }
    exit 0
'

# ── Test: tidy_up calls scancel with slurmJobId ────────────────────────
# is_cancellable() gates scancel on the job still being visible in squeue —
# MOCK_SQUEUE_ACTIVE_JOBS makes the squeue shim report it as active, matching
# the real-world case (a job tearing itself down is still in the queue at
# the moment tidy_up runs).
sandbox_run_test "tidy_up_cancels_slurm_job" compute '
    '"$_setup_running_job"'
    tidy_up "tidy-job" 0 "$_VLLM_PID"
    assert_shim_called "scancel" "99999" || exit 1
    exit 0
' "MOCK_SQUEUE_ACTIVE_JOBS=99999"

exit "$FAIL"
