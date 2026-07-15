#!/bin/bash
# tests/bash/test-lockfile.sh — Lockfile state machine tests.
#
# Tests all lockfile operations: create, update state transitions,
# cancel requests, status checks, and error handling.

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/test-utils.sh"
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../src/templates/lib" && pwd)/utils.sh"

FAIL=0

# ── Setup / teardown ────────────────────────────────────────────────────────

setup() {
    TEST_DIR=$(setup_test_env)
    export ENGINE_DIR="$TEST_DIR/engine"
    export SLURM_NODEID=0
    export SLURM_JOB_ID=99999
    export SLURM_JOB_START_TIME=$(date +%s)
    export SLURM_JOB_END_TIME=$(date +%s)
}

teardown() {
    cleanup_test_env "$TEST_DIR"
}

# ── Test: create_status_pending ─────────────────────────────────────────────

test_create_pending_basic() {
    setup

    local port
    port=$(create_status_pending "test-job" "test-model" 30)
    local lockfile="$ENGINE_DIR/jobs/test-job/status.json"

    assert_file_exists "$lockfile" || { echo "FAIL: lockfile not created"; FAIL=1; teardown; return; }
    assert_status "$lockfile" "pending" || { FAIL=1; teardown; return; }
    assert_json_eq "$lockfile" ".jobName" "test-job" || { FAIL=1; teardown; return; }
    assert_json_eq "$lockfile" ".model" "test-model" || { FAIL=1; teardown; return; }
    assert_json_eq "$lockfile" ".idleTimeout" "30" || { FAIL=1; teardown; return; }

    # Verify port is a valid high port
    local lockfile_port
    lockfile_port=$(resolve_setting "test-job" ".serverPort")
    if [ "$lockfile_port" -ge 49152 ] 2>/dev/null && [ "$lockfile_port" -le 65535 ] 2>/dev/null; then
        [ "$port" = "$lockfile_port" ] || { echo "FAIL: port mismatch $port vs $lockfile_port"; FAIL=1; teardown; return; }
    else
        echo "FAIL: invalid port $lockfile_port (expected 49152-65535)"
        FAIL=1
        teardown
        return
    fi

    # Verify requestedTime is an ISO timestamp
    local req_time
    req_time=$(resolve_setting "test-job" ".requestedTime")
    [[ "$req_time" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}T ]] || { echo "FAIL: invalid timestamp $req_time"; FAIL=1; teardown; return; }

    echo "✓ test_create_pending_basic"
    teardown
}

test_create_pending_duplicate() {
    setup

    create_status_pending "dup-job" "model" 30 > /dev/null 2>&1
    if create_status_pending "dup-job" "model" 30 > /dev/null 2>&1; then
        echo "FAIL: duplicate create should have failed"
        FAIL=1
        teardown
        return
    fi

    echo "✓ test_create_pending_duplicate"
    teardown
}

test_create_pending_default_timeout() {
    setup

    local port
    port=$(create_status_pending "timeout-job" "model" 2>/dev/null)
    local lockfile="$ENGINE_DIR/jobs/timeout-job/status.json"

    assert_json_eq "$lockfile" ".idleTimeout" "30" || { FAIL=1; teardown; return; }

    echo "✓ test_create_pending_default_timeout"
    teardown
}

# ── Test: update_status_initialise ──────────────────────────────────────────

test_update_initialise_basic() {
    setup

    create_status_pending "init-job" "model" 30 > /dev/null 2>&1
    update_status_initialise "init-job" 12345
    local lockfile="$ENGINE_DIR/jobs/init-job/status.json"

    assert_status "$lockfile" "initialising" || { FAIL=1; teardown; return; }
    assert_json_eq "$lockfile" ".slurmJobId" "99999" || { FAIL=1; teardown; return; }
    assert_json_eq "$lockfile" ".vllmPid" "12345" || { FAIL=1; teardown; return; }

    # computeHostname should be set
    local hostname
    hostname=$(resolve_setting "init-job" ".computeHostname")
    [ -n "$hostname" ] && [ "$hostname" != "null" ] || { echo "FAIL: computeHostname not set"; FAIL=1; teardown; return; }

    echo "✓ test_update_initialise_basic"
    teardown
}

test_update_initialise_creates_log() {
    setup

    create_status_pending "log-job" "model" 30 > /dev/null 2>&1
    update_status_initialise "log-job" 12345

    local logfile="$ENGINE_DIR/jobs/log-job/vllm.0.log"
    assert_file_exists "$logfile" || { echo "FAIL: log file not created"; FAIL=1; teardown; return; }

    echo "✓ test_update_initialise_creates_log"
    teardown
}

test_update_initialise_worker_only() {
    setup

    create_status_pending "worker-job" "model" 30 > /dev/null 2>&1
    export SLURM_NODEID=1
    update_status_initialise "worker-job" 12345
    export SLURM_NODEID=0

    # Status should still be pending — only head node (node 0) updates
    local lockfile="$ENGINE_DIR/jobs/worker-job/status.json"
    assert_status "$lockfile" "pending" || { echo "FAIL: worker should not update lockfile"; FAIL=1; teardown; return; }

    echo "✓ test_update_initialise_worker_only"
    teardown
}

# ── Test: update_status_running ─────────────────────────────────────────────

test_update_running() {
    setup

    create_status_pending "run-job" "model" 30 > /dev/null 2>&1
    update_status_initialise "run-job" 12345
    update_status_running "run-job"

    local lockfile="$ENGINE_DIR/jobs/run-job/status.json"
    assert_status "$lockfile" "running" || { FAIL=1; teardown; return; }

    echo "✓ test_update_running"
    teardown
}

test_update_running_worker_only() {
    setup

    create_status_pending "run-worker" "model" 30 > /dev/null 2>&1
    export SLURM_NODEID=1
    update_status_running "run-worker"
    export SLURM_NODEID=0

    local lockfile="$ENGINE_DIR/jobs/run-worker/status.json"
    assert_status "$lockfile" "pending" || { echo "FAIL: worker should not set running"; FAIL=1; teardown; return; }

    echo "✓ test_update_running_worker_only"
    teardown
}

# ── Test: update_status_clean_shutdown ───────────────────────────────────────

test_clean_shutdown() {
    setup

    create_status_pending "clean-job" "model" 30 > /dev/null 2>&1
    update_status_initialise "clean-job" 12345
    update_status_running "clean-job"
    update_status_clean_shutdown "clean-job"

    local lockfile="$ENGINE_DIR/jobs/clean-job/status.json"
    assert_status "$lockfile" "stopped" || { FAIL=1; teardown; return; }
    assert_json_eq "$lockfile" ".exitCode" "0" || { FAIL=1; teardown; return; }

    # stopTime should be set
    local stop_time
    stop_time=$(resolve_setting "clean-job" ".stopTime")
    [[ "$stop_time" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}T ]] || { echo "FAIL: invalid stopTime $stop_time"; FAIL=1; teardown; return; }

    echo "✓ test_clean_shutdown"
    teardown
}

# ── Test: update_status_unclean_shutdown ─────────────────────────────────────

test_unclean_shutdown() {
    setup

    create_status_pending "fail-job" "model" 30 > /dev/null 2>&1
    update_status_initialise "fail-job" 12345
    update_status_unclean_shutdown "fail-job" "GPU error" 42

    local lockfile="$ENGINE_DIR/jobs/fail-job/status.json"
    assert_status "$lockfile" "failed" || { FAIL=1; teardown; return; }
    assert_json_eq "$lockfile" ".exitCode" "42" || { FAIL=1; teardown; return; }
    assert_json_eq "$lockfile" ".reason" "GPU error" || { FAIL=1; teardown; return; }

    echo "✓ test_unclean_shutdown"
    teardown
}

# ── Test: request_cancel ────────────────────────────────────────────────────

test_request_cancel() {
    setup

    create_status_pending "cancel-job" "model" 30 > /dev/null 2>&1
    update_status_initialise "cancel-job" 12345
    update_status_running "cancel-job"
    request_cancel "cancel-job"

    local lockfile="$ENGINE_DIR/jobs/cancel-job/status.json"
    assert_status "$lockfile" "cancel" || { FAIL=1; teardown; return; }

    echo "✓ test_request_cancel"
    teardown
}

test_request_cancel_no_lockfile() {
    setup

    if request_cancel "nonexistent-job" 2>/dev/null; then
        echo "FAIL: cancel on missing lockfile should fail"
        FAIL=1
        teardown
        return
    fi

    echo "✓ test_request_cancel_no_lockfile"
    teardown
}

test_request_cancel_from_worker() {
    setup

    create_status_pending "worker-cancel" "model" 30 > /dev/null 2>&1
    export SLURM_NODEID=1
    request_cancel "worker-cancel"
    export SLURM_NODEID=0

    local lockfile="$ENGINE_DIR/jobs/worker-cancel/status.json"
    assert_status "$lockfile" "cancel" || { echo "FAIL: cancel should work from any node"; FAIL=1; teardown; return; }

    echo "✓ test_request_cancel_from_worker"
    teardown
}

# ── Test: is_status ─────────────────────────────────────────────────────────

test_is_status() {
    setup

    create_status_pending "status-job" "model" 30 > /dev/null 2>&1
    is_status "status-job" "pending" || { echo "FAIL: should be pending"; FAIL=1; teardown; return; }

    update_status_initialise "status-job" 12345
    is_status "status-job" "initialising" || { echo "FAIL: should be initialising"; FAIL=1; teardown; return; }

    update_status_running "status-job"
    is_status "status-job" "running" || { echo "FAIL: should be running"; FAIL=1; teardown; return; }

    request_cancel "status-job"
    is_status "status-job" "cancel" || { echo "FAIL: should be cancel"; FAIL=1; teardown; return; }

    # Negative test
    is_status "status-job" "failed" && { echo "FAIL: should not be failed"; FAIL=1; teardown; return; }
    is_status "status-job" "stopped" && { echo "FAIL: should not be stopped"; FAIL=1; teardown; return; }

    echo "✓ test_is_status"
    teardown
}

test_is_status_missing_lockfile() {
    setup

    is_status "ghost-job" "pending" && { echo "FAIL: missing lockfile should return false"; FAIL=1; teardown; return; }
    is_status "ghost-job" "running" && { echo "FAIL: missing lockfile should return false"; FAIL=1; teardown; return; }

    echo "✓ test_is_status_missing_lockfile"
    teardown
}

# ── Test: update_reason ─────────────────────────────────────────────────────

test_update_reason() {
    setup

    create_status_pending "reason-job" "model" 30 > /dev/null 2>&1
    update_reason "reason-job" "user requested shutdown"

    local lockfile="$ENGINE_DIR/jobs/reason-job/status.json"
    assert_json_eq "$lockfile" ".reason" "user requested shutdown" || { FAIL=1; teardown; return; }
    # Status should remain unchanged
    assert_status "$lockfile" "pending" || { FAIL=1; teardown; return; }

    echo "✓ test_update_reason"
    teardown
}

# ── Test: resolve_setting ───────────────────────────────────────────────────

test_resolve_setting() {
    setup

    create_status_pending "resolve-job" "my-model" 15 > /dev/null 2>&1

    local name
    name=$(resolve_setting "resolve-job" ".jobName")
    [ "$name" = "resolve-job" ] || { echo "FAIL: jobName=$name"; FAIL=1; teardown; return; }

    local model
    model=$(resolve_setting "resolve-job" ".model")
    [ "$model" = "my-model" ] || { echo "FAIL: model=$model"; FAIL=1; teardown; return; }

    local timeout
    timeout=$(resolve_setting "resolve-job" ".idleTimeout")
    [ "$timeout" = "15" ] || { echo "FAIL: idleTimeout=$timeout"; FAIL=1; teardown; return; }

    # Missing field
    local missing
    missing=$(resolve_setting "resolve-job" ".nonexistent")
    [ "$missing" = "null" ] || { echo "FAIL: missing field should return null, got $missing"; FAIL=1; teardown; return; }

    echo "✓ test_resolve_setting"
    teardown
}

# ── Test: full lifecycle state transitions ───────────────────────────────────

test_full_lifecycle() {
    setup

    create_status_pending "lifecycle-job" "lifecycle-model" 60 > /dev/null 2>&1
    is_status "lifecycle-job" "pending" || { echo "FAIL: not pending"; FAIL=1; teardown; return; }

    update_status_initialise "lifecycle-job" 54321
    is_status "lifecycle-job" "initialising" || { echo "FAIL: not initialising"; FAIL=1; teardown; return; }

    update_status_running "lifecycle-job"
    is_status "lifecycle-job" "running" || { echo "FAIL: not running"; FAIL=1; teardown; return; }

    update_status_clean_shutdown "lifecycle-job"
    is_status "lifecycle-job" "stopped" || { echo "FAIL: not stopped"; FAIL=1; teardown; return; }

    # Verify all fields preserved
    local lockfile="$ENGINE_DIR/jobs/lifecycle-job/status.json"
    assert_json_eq "$lockfile" ".jobName" "lifecycle-job" || { FAIL=1; teardown; return; }
    assert_json_eq "$lockfile" ".model" "lifecycle-model" || { FAIL=1; teardown; return; }
    assert_json_eq "$lockfile" ".idleTimeout" "60" || { FAIL=1; teardown; return; }
    assert_json_eq "$lockfile" ".exitCode" "0" || { FAIL=1; teardown; return; }

    echo "✓ test_full_lifecycle"
    teardown
}

test_lifecycle_cancel() {
    setup

    create_status_pending "cancel-lifecycle" "model" 30 > /dev/null 2>&1
    update_status_initialise "cancel-lifecycle" 12345
    update_status_running "cancel-lifecycle"
    request_cancel "cancel-lifecycle"

    # Verify cancel is detected
    is_status "cancel-lifecycle" "cancel" || { echo "FAIL: not cancel"; FAIL=1; teardown; return; }

    # Simulate what tidy_up does on SIGUSR2
    update_reason "cancel-lifecycle" "user cancel"
    update_status_clean_shutdown "cancel-lifecycle"

    local lockfile="$ENGINE_DIR/jobs/cancel-lifecycle/status.json"
    assert_status "$lockfile" "stopped" || { FAIL=1; teardown; return; }
    assert_json_eq "$lockfile" ".reason" "user cancel" || { FAIL=1; teardown; return; }

    echo "✓ test_lifecycle_cancel"
    teardown
}

test_lifecycle_fail_during_startup() {
    setup

    create_status_pending "fail-startup" "model" 30 > /dev/null 2>&1
    update_status_initialise "fail-startup" 12345

    # Simulate vLLM crash with exit code 1 during initialising
    update_status_unclean_shutdown "fail-startup" "failed to start" 1

    local lockfile="$ENGINE_DIR/jobs/fail-startup/status.json"
    assert_status "$lockfile" "failed" || { FAIL=1; teardown; return; }
    assert_json_eq "$lockfile" ".reason" "failed to start" || { FAIL=1; teardown; return; }
    assert_json_eq "$lockfile" ".exitCode" "1" || { FAIL=1; teardown; return; }

    echo "✓ test_lifecycle_fail_during_startup"
    teardown
}

# ── Run all tests ───────────────────────────────────────────────────────────

test_create_pending_basic
test_create_pending_duplicate
test_create_pending_default_timeout
test_update_initialise_basic
test_update_initialise_creates_log
test_update_initialise_worker_only
test_update_running
test_update_running_worker_only
test_clean_shutdown
test_unclean_shutdown
test_request_cancel
test_request_cancel_no_lockfile
test_request_cancel_from_worker
test_is_status
test_is_status_missing_lockfile
test_update_reason
test_resolve_setting
test_full_lifecycle
test_lifecycle_cancel
test_lifecycle_fail_during_startup

exit "$FAIL"
