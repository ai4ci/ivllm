#!/bin/bash
# tests/bash/run.sh — Run all bash framework tests.
#
# Usage:
#   bash tests/bash/run.sh              # Run all tests
#   bash tests/bash/test-lockfile.sh     # Run a single test file
#
# Exit code: 0 if all tests pass, 1 if any fail.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FAIL=0
PASS=0
TOTAL=0

echo "=== ivllm bash test suite ==="
echo ""

# Collect test files, excluding run.sh itself and lib/
shopt -s nullglob  # prevent literal glob when no matches
for test_file in "$SCRIPT_DIR"/test-*.sh; do
    # Skip the runner itself and lib directory files
    [[ "$(basename "$test_file")" == "run.sh" ]] && continue
    [[ "$test_file" == *"/lib/"* ]] && continue
    # Only accept regular files
    [[ ! -f "$test_file" ]] && continue

    TOTAL=$((TOTAL + 1))
    test_name=$(basename "$test_file")

    echo "--- $test_name ---"

    # Run the test file in a subshell so it can't affect other tests
    if bash "$test_file"; then
        echo "✓ $test_name"
        PASS=$((PASS + 1))
    else
        echo "✗ $test_name"
        FAIL=$((FAIL + 1))
    fi
    echo ""
done
shopt -u nullglob

if [[ "$TOTAL" -eq 0 ]]; then
    echo "(no test files found)"
fi

echo "=== Results: $PASS passed, $FAIL failed, $TOTAL total ==="
exit "$FAIL"
