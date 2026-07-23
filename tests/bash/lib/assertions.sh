#!/bin/bash
# tests/bash/lib/assertions.sh
#
# Assertion helpers shared by all bash tests, both plain (unit/) and
# sandboxed (sandboxed/). Pure bash + jq — safe to source both on the host
# and inside a bwrap sandbox (jq is bound in read-only either way).

assert_file_exists() {
    if [[ ! -f "$1" ]]; then
        echo "FAIL: File not found: $1"
        return 1
    fi
}

assert_file_not_exists() {
    if [[ -f "$1" ]]; then
        echo "FAIL: File should not exist: $1"
        return 1
    fi
}

assert_dir_exists() {
    if [[ ! -d "$1" ]]; then
        echo "FAIL: Directory not found: $1"
        return 1
    fi
}

assert_json_eq() {
    local file="$1"
    local jq_expr="$2"
    local expected="$3"
    local actual
    actual=$(jq -r "$jq_expr" "$file" 2>/dev/null)
    if [[ "$actual" != "$expected" ]]; then
        echo "FAIL: $file $jq_expr expected '$expected' got '$actual'"
        return 1
    fi
}

assert_status() {
    local file="$1"
    local expected="$2"
    assert_json_eq "$file" ".status" "$expected"
}

assert_exit_code() {
    local actual=$1
    local expected=$2
    if [[ "$actual" -ne "$expected" ]]; then
        echo "FAIL: exit code expected $expected got $actual"
        return 1
    fi
}

assert_eq() {
    local actual="$1" expected="$2" label="${3:-value}"
    if [[ "$actual" != "$expected" ]]; then
        echo "FAIL: $label expected '$expected' got '$actual'"
        return 1
    fi
}

assert_contains() {
    local haystack="$1" needle="$2" label="${3:-output}"
    if [[ "$haystack" != *"$needle"* ]]; then
        echo "FAIL: $label expected to contain '$needle', got: $haystack"
        return 1
    fi
}

# Assert a PATH shim was invoked with a matching substring. Requires
# IVLLM_TEST_CALL_LOG to be set (sandbox_run sets it to /work/calls.log for
# every sandboxed test — see tests/bash/lib/sandbox.sh and
# tests/bash/shims/_shim-common.sh).
# Usage: assert_shim_called "sbatch" "--job-name qwen36"
assert_shim_called() {
    local tool="$1" substr="$2"
    local log="${IVLLM_TEST_CALL_LOG:-}"
    if [[ -z "$log" || ! -f "$log" ]]; then
        echo "FAIL: no call log found at '${log:-<unset>}'"
        return 1
    fi
    if ! grep -F "[$tool]" "$log" | grep -qF "$substr"; then
        echo "FAIL: expected $tool to be called with '$substr', call log:"
        sed 's/^/  /' "$log"
        return 1
    fi
}

assert_shim_not_called() {
    local tool="$1"
    local log="${IVLLM_TEST_CALL_LOG:-}"
    [[ -z "$log" || ! -f "$log" ]] && return 0
    if grep -qF "[$tool]" "$log"; then
        echo "FAIL: expected $tool NOT to be called, call log:"
        sed 's/^/  /' "$log"
        return 1
    fi
}
