#!/bin/bash
# shellcheck disable=SC2016
# tests/bash/sandboxed/test-login-handoff.sh — Login-node wrapper script tests.
#
# Tests the login-node wrapper scripts (ivllm-cancel.sh, ivllm-status.sh,
# ivllm-setup.sh) in the `login` profile sandbox. These scripts call
# scancel/squeue/srun which are shimmed to record calls in /work/calls.log.
#
#
# QUOTING: sandbox_run_test bodies are single-quoted strings. Single quotes
# inside jq would terminate the body. Instead, we write JSON with printf
# (using only double quotes inside single-quoted printf format strings).

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../lib" && pwd)/sandbox.sh"

# --- Test: serve with minimal config -----------------------------------
#
# ivllm-serve.sh now sources ivllm-get-model.sh inline (synchronously, on
# the login node) before submitting the sbatch job — see ADR / roadmap
# "model download moved out of a separate sbatch job". MOCK_HF_CACHED_MODELS
# tells the `hf` shim that the fixture model is already cached, so
# ivllm-get-model.sh exits 0 immediately without attempting a download and
# ivllm-serve.sh proceeds to the sbatch call.
sandbox_run_test "login_serves_with_minimal_config" login '
    mkdir -p "$IVLLM_PROJECTDIR/engine/jobs/serve-job"
    cp /work/fixtures/minimal.yaml "$IVLLM_PROJECTDIR/engine/jobs/serve-job/vllm.yaml"
    # Create a stopped lockfile so create_status_pending sees "stopped" and
    # restarts it (the only way to pre-exist the lockfile without failing).
    create_status_pending "serve-job" "test-org/test-model-7b" 30 > /dev/null 2>&1
    lf=$(resolve_job_status "serve-job")
    printf '"'"'{"jobName":"serve-job","model":"test-model","status":"stopped","serverPort":49152}'"'"' > "$lf"
    unset IVLLM_UTILS
    # Model is fixture'"'"'s "test-org/test-model-7b" — mark it pre-cached so the
    # inline ivllm-get-model.sh step short-circuits without a real download.
    export MOCK_HF_CACHED_MODELS="test-org/test-model-7b"
    bash "$IVLLM_PROJECTDIR/engine/ivllm-serve.sh" -j serve-job
    assert_shim_called "hf" "cache" || exit 1
    # assert_shim_not_called only takes a tool name (checks it was never
    # called at all), so check the specific subcommand directly against
    # the call log instead.
    if grep -F "[hf]" "$IVLLM_TEST_CALL_LOG" | grep -qF "download"; then
        echo "FAIL: hf download should not have been called (model pre-cached)"
        exit 1
    fi
    assert_shim_called "sbatch" "--job-name serve-job" || exit 1
    assert_shim_called "sbatch" "--partition=interactive" || exit 1
    assert_shim_called "sbatch" "slurm-vllm-serve.sh" || exit 1
    exit 0
'

# --- Test: serve downloads model when not cached ------------------------
sandbox_run_test "login_serves_downloads_uncached_model" login '
    mkdir -p "$IVLLM_PROJECTDIR/engine/jobs/serve-dl-job"
    cp /work/fixtures/minimal.yaml "$IVLLM_PROJECTDIR/engine/jobs/serve-dl-job/vllm.yaml"
    create_status_pending "serve-dl-job" "test-org/test-model-7b" 30 > /dev/null 2>&1
    lf=$(resolve_job_status "serve-dl-job")
    printf '"'"'{"jobName":"serve-dl-job","model":"test-model","status":"stopped","serverPort":49153}'"'"' > "$lf"
    unset IVLLM_UTILS
    # Model not in MOCK_HF_CACHED_MODELS — ivllm-get-model.sh must fall back
    # to `srun ... hf download`. A token is required for the download branch.
    export HF_TOKEN="mock-token"
    bash "$IVLLM_PROJECTDIR/engine/ivllm-serve.sh" -j serve-dl-job
    assert_shim_called "hf" "cache" || exit 1
    assert_shim_called "srun" "hf download test-org/test-model-7b" || exit 1
    assert_shim_called "sbatch" "--job-name serve-dl-job" || exit 1
    exit 0
'

# --- Test: cancel existing job -----------------------------------------
sandbox_run_test "login_cancels_existing" login '
    mkdir -p "$IVLLM_PROJECTDIR/engine/jobs/cancel-job"
    cp /work/fixtures/minimal.yaml "$IVLLM_PROJECTDIR/engine/jobs/cancel-job/vllm.yaml"
    create_status_pending "cancel-job" "model" 30 > /dev/null 2>&1
    lf=$(resolve_job_status "cancel-job")
    printf '"'"'{"jobName":"cancel-job","model":"model","status":"pending","slurmJobId":"SLURM-12345","serverPort":50000}'"'"' > "$lf"

    # Unset IVLLM_UTILS guard so ivllm-cancel.sh can source utils.sh
    unset IVLLM_UTILS
    bash "$IVLLM_PROJECTDIR/engine/ivllm-cancel.sh" -j cancel-job
    assert_shim_not_called "scancel" || exit 1
    exit 0
'

# --- Test: cancel non-existent job fails -------------------------------
sandbox_run_test "login_cancels_missing_job" login '
    unset IVLLM_UTILS
    if bash "$IVLLM_PROJECTDIR/engine/ivllm-cancel.sh" -j nonexistent 2>/dev/null; then
        echo "FAIL: cancel on missing job should have failed"
        exit 1
    fi
    assert_shim_not_called "scancel" || exit 1
    exit 0
'

# --- Test: show status for single job ----------------------------------
sandbox_run_test "login_shows_status" login '
    mkdir -p "$IVLLM_PROJECTDIR/engine/jobs/status-job"
    cp /work/fixtures/minimal.yaml "$IVLLM_PROJECTDIR/engine/jobs/status-job/vllm.yaml"
    create_status_pending "status-job" "model" 30 > /dev/null 2>&1
    # Unset IVLLM_UTILS guard so ivllm-status.sh can source utils.sh
    unset IVLLM_UTILS
    output=$(bash "$IVLLM_PROJECTDIR/engine/ivllm-status.sh" -j status-job)
    rc=$?
    echo "$output" | jq -r ".jobName" | grep -q "status-job" || exit 1
    [ "$rc" -eq 0 ] || { echo "FAIL: ivllm-status returned exit code $rc"; exit 1; }
    exit 0
'

# --- Test: setup runs with version flag --------------------------------
sandbox_run_test "login_setup_runs" login '
    bash "$IVLLM_PROJECTDIR/engine/ivllm-setup.sh" -v 0.8.0
    assert_shim_called "srun" "slurm-vllm-setup.sh" || exit 1
    assert_shim_called "srun" "0.8.0" || exit 1
    exit 0
'

# --- Test: force cancel calls scancel ----------------------------------
sandbox_run_test "login_force_cancel" login '
    mkdir -p "$IVLLM_PROJECTDIR/engine/jobs/fcancel-job"
    cp /work/fixtures/minimal.yaml "$IVLLM_PROJECTDIR/engine/jobs/fcancel-job/vllm.yaml"
    create_status_pending "fcancel-job" "model" 30 > /dev/null 2>&1
    lf=$(resolve_job_status "fcancel-job")
    # Write JSON directly — include user field (set to root in sandbox)
    printf '"'"'{"jobName":"fcancel-job","model":"model","status":"running","slurmJobId":"SLURM-99999","serverPort":50001,"user":"root"}'"'"' > "$lf"
    # Unset IVLLM_UTILS guard so ivllm-cancel.sh can source utils.sh
    unset IVLLM_UTILS
    bash "$IVLLM_PROJECTDIR/engine/ivllm-cancel.sh" -j fcancel-job -f
    assert_shim_called "scancel" "SLURM-99999" || exit 1
    exit 0
'

exit "$FAIL"
