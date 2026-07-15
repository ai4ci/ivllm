#!/bin/bash
# tests/bash/test-cache.sh — JIT cache save/restore tests.
#
# Each test runs in a completely isolated subshell to prevent state leakage.
# shellcheck disable=SC1091,SC2016

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/lib/test-utils.sh"

FAIL=0

TEST_LIB="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../src/templates/lib" && pwd)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

run_test() {
    local test_name="$1"
    local test_func="$2"

    bash -c "
source '$TEST_LIB/utils.sh'
source '$SCRIPT_DIR/lib/test-utils.sh'

test_dir=\$(setup_test_env)
export ENGINE_DIR=\"\$test_dir/engine\"
export SLURM_NODEID=0
export LOCALDIR=\"\$test_dir/local\"
mkdir -p \"\$LOCALDIR\"
$test_func
rc=\$?
cleanup_test_env \"\$test_dir\"
exit \$rc
" 2>&1

    local rc=$?
    if [ $rc -eq 0 ]; then
        echo "✓ $test_name"
    else
        FAIL=1
    fi
}

# ── Test functions ──────────────────────────────────────────────────────────

TEST_CACHE_SAVE_RESTORE='create_status_pending "cache-job" "cache-model" 30 > /dev/null 2>&1
localdir=$(resolve_localdir "cache-job")
mkdir -p "$localdir"
echo "test data" > "$localdir/test-file.txt"
save_cache "cache-job"
cachetar=$(resolve_cachetar "cache-job")
assert_file_exists "$cachetar" || exit 1
rm -f "$localdir/test-file.txt"
restore_cache "cache-job"
assert_file_exists "$localdir/test-file.txt" || exit 1
content=$(cat "$localdir/test-file.txt")
[ "$content" = "test data" ] || exit 1
exit 0'

TEST_CACHE_RESTORE_MISSING='create_status_pending "cache-job" "cache-model" 30 > /dev/null 2>&1
localdir=$(resolve_localdir "cache-job")
touch "$localdir/.marker"
restore_cache "cache-job"
# Marker file should still exist after restore (no data loss)
assert_file_exists "$localdir/.marker" || exit 1
exit 0'

TEST_CACHE_SAVE_EMPTY='create_status_pending "cache-job" "cache-model" 30 > /dev/null 2>&1
localdir=$(resolve_localdir "cache-job")
save_cache "cache-job"
cachetar=$(resolve_cachetar "cache-job")
assert_file_exists "$cachetar" || exit 1
exit 0'

TEST_CACHE_PERMISSIONS='create_status_pending "cache-job" "cache-model" 30 > /dev/null 2>&1
localdir=$(resolve_localdir "cache-job")
echo "data" > "$localdir"/perms-test.txt
save_cache "cache-job"
cachetar=$(resolve_cachetar "cache-job")
assert_file_exists "$cachetar" || exit 1
exit 0'

# ── Run tests ───────────────────────────────────────────────────────────────

run_test "cache_save_restore" "$TEST_CACHE_SAVE_RESTORE"
run_test "cache_restore_missing" "$TEST_CACHE_RESTORE_MISSING"
run_test "cache_save_empty" "$TEST_CACHE_SAVE_EMPTY"
run_test "cache_permissions" "$TEST_CACHE_PERMISSIONS"

exit "$FAIL"
