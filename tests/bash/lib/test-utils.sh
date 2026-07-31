#!/bin/bash
# tests/bash/lib/test-utils.sh
#
# Shared helpers for PLAIN (non-sandboxed) bash unit tests — i.e. tests that
# exercise pure utils.sh logic without invoking any external command that
# needs mocking (no srun/sbatch/scancel/vllm/yq/jq-adjacent process
# behaviour). These run directly on the host, no bwrap involved, so they
# stay fast for quick TDD iteration on things like path resolution and
# semver comparison.
#
# For anything that shells out to an external command (jq/yq included, since
# we test against the *real* installed yq — see design/issues.md #7-9) or
# needs process/signal isolation (monitor triad, exit traps), use
# tests/bash/lib/sandbox.sh instead.
#
# shellcheck disable=SC2155

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/assertions.sh"

# Creates a temporary ENGINE_DIR-equivalent (IVLLM_PROJECTDIR) and sets up
# the environment for a plain (non-sandboxed) test. Returns the path via
# stdout — caller should capture it and clean up with cleanup_test_env.
# Usage: local test_dir=$(setup_test_env)
setup_test_env() {
    local dir
    dir=$(mktemp -d)
    export IVLLM_PROJECTDIR="$dir/project"
    mkdir -p "$IVLLM_PROJECTDIR/engine/jobs"
    export HOME="$dir/home"
    mkdir -p "$HOME"
    echo "$dir"
}

# Cleans up a test environment created by setup_test_env.
# Usage: cleanup_test_env "$test_dir"
cleanup_test_env() {
    rm -rf "$1"
}
