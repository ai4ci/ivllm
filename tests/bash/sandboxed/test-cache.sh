#!/bin/bash
# tests/bash/sandboxed/test-cache.sh — JIT cache save/restore tests.
#
# Runs inside the bwrap "compute" profile sandbox. resolve_job_jit_cache()
# writes under $HOME/.cache/ivllm/<job>/ — HOME is a fresh, host-backed
# writable dir per test (see tests/bash/lib/sandbox.sh), so cache tarballs
# never touch the real invoking user's home directory.

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../lib" && pwd)/sandbox.sh"

sandbox_run_test "cache_save_restore" compute '
    create_status_pending "cache-job" "cache-model" 30 > /dev/null 2>&1
    localdir=$(resolve_localdir "cache-job")
    echo "test data" > "$localdir/test-file.txt"

    save_cache "cache-job"
    cachetar=$(resolve_job_jit_cache "cache-job")
    assert_file_exists "$cachetar" || exit 1

    rm -f "$localdir/test-file.txt"
    restore_cache "cache-job"
    assert_file_exists "$localdir/test-file.txt" || exit 1

    content=$(cat "$localdir/test-file.txt")
    [ "$content" = "test data" ] || { echo "FAIL: content mismatch: $content"; exit 1; }
'

sandbox_run_test "cache_restore_missing" compute '
    create_status_pending "cache-job" "cache-model" 30 > /dev/null 2>&1
    localdir=$(resolve_localdir "cache-job")
    touch "$localdir/.marker"

    # No tar has been saved yet — restore should be a graceful no-op.
    restore_cache "cache-job"
    assert_file_exists "$localdir/.marker" || exit 1
'

sandbox_run_test "cache_save_empty" compute '
    create_status_pending "cache-job" "cache-model" 30 > /dev/null 2>&1
    resolve_localdir "cache-job" > /dev/null

    save_cache "cache-job"
    cachetar=$(resolve_job_jit_cache "cache-job")
    assert_file_exists "$cachetar" || exit 1
'

sandbox_run_test "cache_permissions" compute '
    create_status_pending "cache-job" "cache-model" 30 > /dev/null 2>&1
    localdir=$(resolve_localdir "cache-job")
    echo "data" > "$localdir/perms-test.txt"

    save_cache "cache-job"
    cachetar=$(resolve_job_jit_cache "cache-job")
    assert_file_exists "$cachetar" || exit 1

    perms=$(stat -c "%a" "$cachetar")
    [ "$perms" = "664" ] || { echo "FAIL: expected mode 664, got $perms"; exit 1; }
'

sandbox_run_test "cache_worker_node_does_not_save" compute '
    create_status_pending "cache-job" "cache-model" 30 > /dev/null 2>&1
    localdir=$(resolve_localdir "cache-job")
    echo "data" > "$localdir/test-file.txt"

    export SLURM_NODEID=1
    save_cache "cache-job"
    export SLURM_NODEID=0

    cachetar=$(resolve_job_jit_cache "cache-job")
    assert_file_not_exists "$cachetar" || { echo "FAIL: worker node should not save cache"; exit 1; }
'

exit "$FAIL"
