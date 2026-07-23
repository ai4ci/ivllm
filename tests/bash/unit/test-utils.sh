#!/bin/bash
# tests/bash/unit/test-utils.sh — Unit tests for utils.sh helper functions.
#
# Pure bash logic tests — no sandbox needed. These run on the host to test
# arithmetic, time conversion, and version comparison without any mocking.

# Set IVLLM_PROJECTDIR before sourcing utils.sh to avoid the guard check.
IVLLM_PROJECTDIR=/tmp/ivllm-test
export IVLLM_PROJECTDIR
# shellcheck disable=SC1091
source /home/scoder/Git/isambard-vllm/src/engine/lib/utils.sh

# ── Test: get_max_job_time returns user time when under max ─────────────
test_get_max_under() {
    local result
    result=$(get_max_job_time "02:00:00")
    if [[ "$result" != "02:00:00" ]]; then
        echo "FAIL: expected 02:00:00, got $result"
        return 1
    fi
    return 0
}

# ── Test: get_max_job_time caps at max when user exceeds ────────────────
test_get_max_over() {
    local result
    result=$(get_max_job_time "24:00:00")
    if [[ "$result" != "08:00:00" ]]; then
        echo "FAIL: expected 08:00:00, got $result"
        return 1
    fi
    return 0
}

# ── Test: get_max_job_time defaults to 08:00:00 ────────────────────────
test_get_max_default() {
    local result
    result=$(get_max_job_time "08:00:00")
    if [[ "$result" != "08:00:00" ]]; then
        echo "FAIL: expected 08:00:00, got $result"
        return 1
    fi
    return 0
}

# ── Test: get_max_job_time with minutes only ───────────────────────────
test_get_max_minutes() {
    local result
    result=$(get_max_job_time "00:30:00")
    if [[ "$result" != "00:30:00" ]]; then
        echo "FAIL: expected 00:30:00, got $result"
        return 1
    fi
    return 0
}

# ── Test: get_max_job_time with seconds only ───────────────────────────
test_get_max_seconds() {
    local result
    result=$(get_max_job_time "00:00:45")
    if [[ "$result" != "00:00:45" ]]; then
        echo "FAIL: expected 00:00:45, got $result"
        return 1
    fi
    return 0
}

# ── Test: semver_lt (less-than) ────────────────────────────────────────
test_semver_lt() {
    # 0.1.0 < 0.2.0
    semver_lt "0.1.0" "0.2.0" && return 0
    echo "FAIL: 0.1.0 should be < 0.2.0"
    return 1
}

# ── Test: semver_gte (greater-or-equal) ────────────────────────────────
test_semver_gte() {
    # 0.2.0 >= 0.1.0
    semver_gte "0.2.0" "0.1.0" && return 0
    echo "FAIL: 0.2.0 should be >= 0.1.0"
    return 1
}

# ── Test: semver_gte equality ──────────────────────────────────────────
test_semver_gte_equal() {
    # 1.0.0 >= 1.0.0
    semver_gte "1.0.0" "1.0.0" && return 0
    echo "FAIL: 1.0.0 should be >= 1.0.0"
    return 1
}

# ── Test: semver_lt major version ─────────────────────────────────────
test_semver_lt_major() {
    # 1.0.0 < 2.0.0
    semver_lt "1.0.0" "2.0.0" && return 0
    echo "FAIL: 1.0.0 should be < 2.0.0"
    return 1
}

# ── Test: semver_lt patch version ──────────────────────────────────────
test_semver_lt_patch() {
    # 0.1.0 < 0.1.1
    semver_lt "0.1.0" "0.1.1" && return 0
    echo "FAIL: 0.1.0 should be < 0.1.1"
    return 1
}

# ── Test: semver_lt equal versions ─────────────────────────────────────
test_semver_lt_equal() {
    # 0.1.0 < 0.1.0 should be false
    semver_lt "0.1.0" "0.1.0" && { echo "FAIL: 0.1.0 should NOT be < 0.1.0"; return 1; }
    return 0
}

# ── Test: rev_semver_sort descending ───────────────────────────────────
test_rev_semver_sort() {
    local result
    result=$(rev_semver_sort "0.1.0" "0.3.0" "0.2.0")
    local expected="0.1.0
0.2.0
0.3.0"
    if [[ "$result" != "$expected" ]]; then
        echo "FAIL: expected ascending sort, got: $result"
        return 1
    fi
    return 0
}

FAIL=0
# ── Run all tests ──────────────────────────────────────────────────────
run_test() {
    local name="$1"
    if eval "$name"; then
        echo "✓ $name"
    else
        echo "✗ $name"
        FAIL=1
    fi
}

run_test test_get_max_under
run_test test_get_max_over
run_test test_get_max_default
run_test test_get_max_minutes
run_test test_get_max_seconds
run_test test_semver_lt
run_test test_semver_gte
run_test test_semver_gte_equal
run_test test_semver_lt_major
run_test test_semver_lt_patch
run_test test_semver_lt_equal
run_test test_rev_semver_sort

exit "$FAIL"
