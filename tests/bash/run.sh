#!/bin/bash
# tests/bash/run.sh — Run all bash framework tests.
#
# Usage:
#   bash tests/bash/run.sh                       # Run all tests
#   bash tests/bash/run.sh unit                  # Run only unit/ tests
#   bash tests/bash/run.sh sandboxed             # Run only sandboxed/ tests
#   bash tests/bash/unit/test-semver.sh          # Run a single test file
#
# Test layout:
#   unit/       — fast, non-sandboxed tests of pure bash logic (no external
#                 commands that need mocking: no srun/sbatch/yq/jq subprocess
#                 semantics). Run directly on the host.
#   sandboxed/  — tests that invoke external commands (real yq/jq, mocked
#                 srun/sbatch/scancel/vllm/...) or need process/signal
#                 isolation (monitor triad, exit traps). Run inside a
#                 bubblewrap sandbox — see tests/bash/lib/sandbox.sh and
#                 design/testing.md.
#
# Exit code: 0 if all tests pass, 1 if any fail.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FAIL=0
PASS=0
TOTAL=0

SCOPES=("$@")
if [[ ${#SCOPES[@]} -eq 0 ]]; then
    SCOPES=(unit sandboxed)
fi

echo "=== ivllm bash test suite ==="
echo ""

for scope in "${SCOPES[@]}"; do
    scope_dir="$SCRIPT_DIR/$scope"
    if [[ ! -d "$scope_dir" ]]; then
        echo "(unknown scope: $scope — skipping)"
        continue
    fi

    if [[ "$scope" == "sandboxed" ]] && ! command -v bwrap >/dev/null 2>&1; then
        echo "!!! bwrap (bubblewrap) not found — skipping sandboxed/ tests !!!"
        echo ""
        continue
    fi

    while IFS= read -r -d '' test_file; do
        TOTAL=$((TOTAL + 1))
        test_name="$scope/$(basename "$test_file")"

        echo "--- $test_name ---"

        if bash "$test_file"; then
            echo "✓ $test_name"
            PASS=$((PASS + 1))
        else
            echo "✗ $test_name"
            FAIL=$((FAIL + 1))
        fi
        echo ""
    done < <(find "$scope_dir" -maxdepth 1 -name 'test-*.sh' -print0 | sort -z)
done

if [[ "$TOTAL" -eq 0 ]]; then
    echo "(no test files found)"
fi

echo "=== Results: $PASS passed, $FAIL failed, $TOTAL total ==="
exit "$([[ $FAIL -eq 0 ]] && echo 0 || echo 1)"
