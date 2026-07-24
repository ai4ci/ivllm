#!/bin/bash
# shellcheck disable=SC2016
# tests/bash/sandboxed/test-lockfile.sh — Lockfile state machine tests.
#
# Runs inside the bwrap "compute" profile sandbox (see
# tests/bash/lib/sandbox.sh) so that jq operations run against a real,
# isolated filesystem, and SLURM_NODEID gating (create/init/running only act
# on node 0) can be tested by overriding MOCK_SLURM_NODEID per-test.

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../lib" && pwd)/sandbox.sh"

sandbox_run_test "create_pending_basic" compute '
    port=$(create_status_pending "test-job" "test-model" 30)
    lockfile=$(resolve_job_status "test-job")

    assert_file_exists "$lockfile" || exit 1
    assert_status "$lockfile" "pending" || exit 1
    assert_json_eq "$lockfile" ".jobName" "test-job" || exit 1
    assert_json_eq "$lockfile" ".model" "test-model" || exit 1
    assert_json_eq "$lockfile" ".idleTimeout" "30" || exit 1

    lockfile_port=$(get_job_status_setting "test-job" ".serverPort")
    if [ "$lockfile_port" -ge 49152 ] 2>/dev/null && [ "$lockfile_port" -le 65535 ] 2>/dev/null; then
        [ "$port" = "$lockfile_port" ] || { echo "FAIL: port mismatch $port vs $lockfile_port"; exit 1; }
    else
        echo "FAIL: invalid port $lockfile_port (expected 49152-65535)"
        exit 1
    fi

    req_time=$(get_job_status_setting "test-job" ".requestedTime")
    [[ "$req_time" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}T ]] || { echo "FAIL: invalid timestamp $req_time"; exit 1; }
'

sandbox_run_test "create_pending_duplicate" compute '
    create_status_pending "dup-job" "model" 30 > /dev/null 2>&1
    if create_status_pending "dup-job" "model" 30 > /dev/null 2>&1; then
        echo "FAIL: duplicate create should have failed"
        exit 1
    fi
'

sandbox_run_test "create_pending_default_timeout" compute '
    port=$(create_status_pending "timeout-job" "model" 2>/dev/null)
    lockfile=$(resolve_job_status "timeout-job")
    assert_json_eq "$lockfile" ".idleTimeout" "30" || exit 1
'

sandbox_run_test "update_initialise_basic" compute '
    create_status_pending "init-job" "model" 30 > /dev/null 2>&1
    update_status_initialise "init-job" 12345
    lockfile=$(resolve_job_status "init-job")

    assert_status "$lockfile" "initialising" || exit 1
    assert_json_eq "$lockfile" ".slurmJobId" "99999" || exit 1
    assert_json_eq "$lockfile" ".vllmPid" "12345" || exit 1

    hostname=$(get_job_status_setting "init-job" ".computeHostname")
    [ -n "$hostname" ] && [ "$hostname" != "null" ] || { echo "FAIL: computeHostname not set"; exit 1; }
'

sandbox_run_test "update_initialise_does_not_create_log" compute '
    # update_status_initialise only updates the lockfile — it does NOT
    # create the log file. In the real system, the log file already exists
    # by this point because SLURM created it via the sbatch --output/--error
    # redirection *before* the job script (which calls this function) even
    # starts running. That precondition, not this function, is what makes
    # resolve_job_log(...) resolvable later. (An earlier version of this
    # test incorrectly asserted the file was created by this call — see
    # design/issues.md for the convention of fixing such findings.)
    create_status_pending "log-job" "model" 30 > /dev/null 2>&1
    update_status_initialise "log-job" 12345
    logfile=$(resolve_job_log "log-job")
    assert_file_not_exists "$logfile" || exit 1
'

sandbox_run_test "update_initialise_worker_only" compute '
    create_status_pending "worker-job" "model" 30 > /dev/null 2>&1
    export SLURM_NODEID=1
    update_status_initialise "worker-job" 12345
    export SLURM_NODEID=0

    lockfile=$(resolve_job_status "worker-job")
    assert_status "$lockfile" "pending" || { echo "FAIL: worker should not update lockfile"; exit 1; }
'

sandbox_run_test "update_running" compute '
    create_status_pending "run-job" "model" 30 > /dev/null 2>&1
    update_status_initialise "run-job" 12345
    update_status_running "run-job"
    lockfile=$(resolve_job_status "run-job")
    assert_status "$lockfile" "running" || exit 1
'

sandbox_run_test "update_running_worker_only" compute '
    create_status_pending "run-worker" "model" 30 > /dev/null 2>&1
    export SLURM_NODEID=1
    update_status_running "run-worker"
    export SLURM_NODEID=0
    lockfile=$(resolve_job_status "run-worker")
    assert_status "$lockfile" "pending" || { echo "FAIL: worker should not set running"; exit 1; }
'

sandbox_run_test "clean_shutdown" compute '
    create_status_pending "clean-job" "model" 30 > /dev/null 2>&1
    update_status_initialise "clean-job" 12345
    update_status_running "clean-job"
    update_status_stopped "clean-job"

    lockfile=$(resolve_job_status "clean-job")
    assert_status "$lockfile" "stopped" || exit 1
    assert_json_eq "$lockfile" ".exitCode" "0" || exit 1

    stop_time=$(get_job_status_setting "clean-job" ".stopTime")
    [[ "$stop_time" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}T ]] || { echo "FAIL: invalid stopTime $stop_time"; exit 1; }
'

sandbox_run_test "unclean_shutdown" compute '
    create_status_pending "fail-job" "model" 30 > /dev/null 2>&1
    update_status_initialise "fail-job" 12345
    update_status_failed "fail-job" "GPU error" 42

    lockfile=$(resolve_job_status "fail-job")
    assert_status "$lockfile" "failed" || exit 1
    assert_json_eq "$lockfile" ".exitCode" "42" || exit 1
    assert_json_eq "$lockfile" ".reason" "GPU error" || exit 1
'

sandbox_run_test "request_cancel" compute '
    create_status_pending "cancel-job" "model" 30 > /dev/null 2>&1
    update_status_initialise "cancel-job" 12345
    update_status_running "cancel-job"
    request_cancel "cancel-job"

    lockfile=$(resolve_job_status "cancel-job")
    assert_status "$lockfile" "cancel" || exit 1
'

sandbox_run_test "request_cancel_no_lockfile" compute '
    if request_cancel "nonexistent-job" 2>/dev/null; then
        echo "FAIL: cancel on missing lockfile should fail"
        exit 1
    fi
'

sandbox_run_test "request_cancel_from_worker" compute '
    create_status_pending "worker-cancel" "model" 30 > /dev/null 2>&1
    export SLURM_NODEID=1
    request_cancel "worker-cancel"
    export SLURM_NODEID=0

    lockfile=$(resolve_job_status "worker-cancel")
    assert_status "$lockfile" "cancel" || { echo "FAIL: cancel should work from any node"; exit 1; }
'

sandbox_run_test "is_status" compute '
    create_status_pending "status-job" "model" 30 > /dev/null 2>&1
    is_status "status-job" "pending" || { echo "FAIL: should be pending"; exit 1; }

    update_status_initialise "status-job" 12345
    is_status "status-job" "initialising" || { echo "FAIL: should be initialising"; exit 1; }

    update_status_running "status-job"
    is_status "status-job" "running" || { echo "FAIL: should be running"; exit 1; }

    request_cancel "status-job"
    is_status "status-job" "cancel" || { echo "FAIL: should be cancel"; exit 1; }

    is_status "status-job" "failed" && { echo "FAIL: should not be failed"; exit 1; }
    is_status "status-job" "stopped" && { echo "FAIL: should not be stopped"; exit 1; }
    exit 0
'

sandbox_run_test "is_status_missing_lockfile" compute '
    is_status "ghost-job" "pending" && { echo "FAIL: missing lockfile should return false"; exit 1; }
    is_status "ghost-job" "running" && { echo "FAIL: missing lockfile should return false"; exit 1; }
    exit 0
'

sandbox_run_test "update_reason" compute '
    create_status_pending "reason-job" "model" 30 > /dev/null 2>&1
    update_reason "reason-job" "user requested shutdown"

    lockfile=$(resolve_job_status "reason-job")
    assert_json_eq "$lockfile" ".reason" "user requested shutdown" || exit 1
    assert_status "$lockfile" "pending" || exit 1
'

sandbox_run_test "get_job_status_setting" compute '
    create_status_pending "resolve-job" "my-model" 15 > /dev/null 2>&1

    name=$(get_job_status_setting "resolve-job" ".jobName")
    [ "$name" = "resolve-job" ] || { echo "FAIL: jobName=$name"; exit 1; }

    model=$(get_job_status_setting "resolve-job" ".model")
    [ "$model" = "my-model" ] || { echo "FAIL: model=$model"; exit 1; }

    timeout=$(get_job_status_setting "resolve-job" ".idleTimeout")
    [ "$timeout" = "15" ] || { echo "FAIL: idleTimeout=$timeout"; exit 1; }

    missing=$(get_job_status_setting "resolve-job" ".nonexistent")
    [ "$missing" = "null" ] || { echo "FAIL: missing field should return null, got $missing"; exit 1; }
'

sandbox_run_test "full_lifecycle" compute '
    create_status_pending "lifecycle-job" "lifecycle-model" 60 > /dev/null 2>&1
    is_status "lifecycle-job" "pending" || { echo "FAIL: not pending"; exit 1; }

    update_status_initialise "lifecycle-job" 54321
    is_status "lifecycle-job" "initialising" || { echo "FAIL: not initialising"; exit 1; }

    update_status_running "lifecycle-job"
    is_status "lifecycle-job" "running" || { echo "FAIL: not running"; exit 1; }

    update_status_stopped "lifecycle-job"
    is_status "lifecycle-job" "stopped" || { echo "FAIL: not stopped"; exit 1; }

    lockfile=$(resolve_job_status "lifecycle-job")
    assert_json_eq "$lockfile" ".jobName" "lifecycle-job" || exit 1
    assert_json_eq "$lockfile" ".model" "lifecycle-model" || exit 1
    assert_json_eq "$lockfile" ".idleTimeout" "60" || exit 1
    assert_json_eq "$lockfile" ".exitCode" "0" || exit 1
'

sandbox_run_test "lifecycle_cancel" compute '
    create_status_pending "cancel-lifecycle" "model" 30 > /dev/null 2>&1
    update_status_initialise "cancel-lifecycle" 12345
    update_status_running "cancel-lifecycle"
    request_cancel "cancel-lifecycle"

    is_status "cancel-lifecycle" "cancel" || { echo "FAIL: not cancel"; exit 1; }

    update_reason "cancel-lifecycle" "user cancel"
    update_status_stopped "cancel-lifecycle"

    lockfile=$(resolve_job_status "cancel-lifecycle")
    assert_status "$lockfile" "stopped" || exit 1
    assert_json_eq "$lockfile" ".reason" "user cancel" || exit 1
'

sandbox_run_test "lifecycle_fail_during_startup" compute '
    create_status_pending "fail-startup" "model" 30 > /dev/null 2>&1
    update_status_initialise "fail-startup" 12345
    update_status_failed "fail-startup" "failed to start" 1

    lockfile=$(resolve_job_status "fail-startup")
    assert_status "$lockfile" "failed" || exit 1
    assert_json_eq "$lockfile" ".reason" "failed to start" || exit 1
    assert_json_eq "$lockfile" ".exitCode" "1" || exit 1
'

exit "$FAIL"
