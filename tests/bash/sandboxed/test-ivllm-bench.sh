#!/bin/bash
# shellcheck disable=SC2016
# tests/bash/sandboxed/test-ivllm-bench.sh — unit tests for the pure/testable
# logic inside design/prototype/ivllm-bench.sh (still a prototype, not yet
# promoted to src/engine/ — see design/bench-implementation-plan.md).
#
# ivllm-bench.sh is structured so its function DEFINITIONS (process_job,
# write_status_summary, status_writer_loop, prefetch_unique_models) have no
# side effects when sourced — only main(), gated behind a source-guard at
# the bottom of the file, actually does anything. That's what lets these
# tests source it directly and call individual functions, the same way
# every other sandboxed test sources utils.sh.
#
# Not covered here (needs a real Isambard run, not a unit test):
# srun --overlap --jobid=<id> actually attaching to a job from outside it,
# real network reachability, and the full process_job() happy path end to
# end (it shells out to ivllm-serve.sh/ivllm-cancel.sh and polls real SLURM
# timing) — see design/bench-implementation-plan.md's testing strategy
# section for the fuller picture of what does and doesn't fit this harness.

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../lib" && pwd)/sandbox.sh"

_SOURCE_PROTOTYPE='source /work/prototype/ivllm-bench.sh'

# ── write_status_summary() ─────────────────────────────────────────────────
# Uses "compute" profile purely so SLURM_NODEID is defined for the
# create_status_pending/update_status_* fixture setup below (those check
# `(( SLURM_NODEID == 0 ))`) — write_status_summary() itself doesn't care
# about SLURM env at all, it just reads status.json files back out.

sandbox_run_test "write_status_summary_counts_and_incomplete" compute "
    $_SOURCE_PROTOTYPE

    create_status_pending job-pending model-a 30 >/dev/null
    create_status_pending job-running model-a 30 >/dev/null
    update_status_running job-running

    jobNames=(job-pending job-running)
    STATUS_FILE=/work/status-summary.json
    ORCHESTRATOR_PID=4242

    write_status_summary

    assert_file_exists \"\$STATUS_FILE\" || exit 1
    assert_eq \"\$(jq -r '.pid' \"\$STATUS_FILE\")\" '4242' 'pid' || exit 1
    assert_eq \"\$(jq -r '.complete' \"\$STATUS_FILE\")\" 'false' 'complete' || exit 1
    assert_eq \"\$(jq -r '.counts.pending' \"\$STATUS_FILE\")\" '1' 'counts.pending' || exit 1
    assert_eq \"\$(jq -r '.counts.running' \"\$STATUS_FILE\")\" '1' 'counts.running' || exit 1
    assert_eq \"\$(jq -r '.counts.stopped' \"\$STATUS_FILE\")\" '0' 'counts.stopped' || exit 1
    assert_eq \"\$(jq -r '.jobs[\"job-running\"].status' \"\$STATUS_FILE\")\" 'running' 'jobs.job-running.status' || exit 1
"

sandbox_run_test "write_status_summary_all_terminal_is_complete" compute "
    $_SOURCE_PROTOTYPE

    create_status_pending job-ok model-a 30 >/dev/null
    update_status_running job-ok
    update_status_stopped job-ok

    create_status_pending job-bad model-a 30 >/dev/null
    update_status_failed job-bad 'crashed after startup' 1

    jobNames=(job-ok job-bad)
    STATUS_FILE=/work/status-summary.json
    ORCHESTRATOR_PID=4242

    write_status_summary

    assert_eq \"\$(jq -r '.complete' \"\$STATUS_FILE\")\" 'true' 'complete' || exit 1
    assert_eq \"\$(jq -r '.counts.stopped' \"\$STATUS_FILE\")\" '1' 'counts.stopped' || exit 1
    assert_eq \"\$(jq -r '.counts.failed' \"\$STATUS_FILE\")\" '1' 'counts.failed' || exit 1
    assert_eq \"\$(jq -r '.jobs[\"job-bad\"].reason' \"\$STATUS_FILE\")\" 'crashed after startup' 'jobs.job-bad.reason' || exit 1
    assert_eq \"\$(jq -r '.jobs[\"job-ok\"].reason' \"\$STATUS_FILE\")\" 'null' 'jobs.job-ok.reason (no failure reason)' || exit 1
"

sandbox_run_test "write_status_summary_missing_lockfile_is_unknown_not_complete" compute "
    $_SOURCE_PROTOTYPE

    # 'job-ghost' never had create_status_pending called — no status.json
    # exists for it at all (e.g. a transient read race right after submit).
    jobNames=(job-ghost)
    STATUS_FILE=/work/status-summary.json
    ORCHESTRATOR_PID=4242

    write_status_summary

    assert_eq \"\$(jq -r '.jobs[\"job-ghost\"].status' \"\$STATUS_FILE\")\" 'unknown' 'jobs.job-ghost.status' || exit 1
    assert_eq \"\$(jq -r '.complete' \"\$STATUS_FILE\")\" 'false' 'complete (unknown must not count as terminal)' || exit 1
"

# ── prefetch_unique_models() ────────────────────────────────────────────────
# IVLLM_BIN points at the sandbox's own engine bind (where ivllm-get-model.sh
# actually lives in this harness) rather than the real ~/.local/bin default.

sandbox_run_test "prefetch_unique_models_skips_already_cached" login "
    $_SOURCE_PROTOTYPE

    mkdir -p \"\$(resolve_job_dir job1)\" \"\$(resolve_job_dir job2)\"
    cp /work/fixtures/minimal.yaml \"\$(resolve_job_config job1)\"
    cp /work/fixtures/minimal.yaml \"\$(resolve_job_config job2)\"

    # Both configs share minimal.yaml's model (test-org/test-model-7b) —
    # pre-create its cache dir so both are seen as already-cached.
    mkdir -p \"\$(resolve_model_dir test-org/test-model-7b)\"

    out=\$(prefetch_unique_models job1 job2)
    rc=\$?

    assert_exit_code \"\$rc\" 0 || exit 1
    cachedCount=\$(echo \"\$out\" | grep -c 'model already cached: test-org/test-model-7b')
    assert_eq \"\$cachedCount\" '1' 'dedup: cached-check should run once for two jobs sharing a model, not twice' || exit 1
" IVLLM_BIN=/work/project/engine

sandbox_run_test "prefetch_unique_models_checks_each_distinct_model" login "
    $_SOURCE_PROTOTYPE

    mkdir -p \"\$(resolve_job_dir job1)\" \"\$(resolve_job_dir job2)\"
    cp /work/fixtures/minimal.yaml \"\$(resolve_job_config job1)\"
    yq w /work/fixtures/minimal.yaml model other-org/other-model-3b > \"\$(resolve_job_config job2)\"

    mkdir -p \"\$(resolve_model_dir test-org/test-model-7b)\"
    mkdir -p \"\$(resolve_model_dir other-org/other-model-3b)\"

    out=\$(prefetch_unique_models job1 job2)
    rc=\$?

    assert_exit_code \"\$rc\" 0 || exit 1
    assert_contains \"\$out\" 'model already cached: test-org/test-model-7b' 'output' || exit 1
    assert_contains \"\$out\" 'model already cached: other-org/other-model-3b' 'output' || exit 1
" IVLLM_BIN=/work/project/engine

exit "$FAIL"
