#!/bin/bash
# tests/bash/lib/sandbox.sh
#
# Bubblewrap (bwrap)-based sandbox harness for bash framework tests.
#
# WHY: the old mock approach (tests/bash/lib/test-utils.sh, pre-rewrite)
# overrode srun/sbatch/scancel/vllm as bash FUNCTIONS sourced into the same
# shell as the code under test. That's fast, but has real gaps:
#   - It never exercises real subprocess/exec semantics (arg parsing across
#     process boundaries, PATH resolution, exit code propagation).
#   - Background processes started by monitor_head/monitor_startup/mock vLLM
#     servers are real host processes. If a test crashes before its trap
#     fires, they leak on the host and can pollute later tests.
#   - There is no isolation from the real HPC modules/tools that might be on
#     the test-runner's PATH (real srun, real scancel, ...).
#
# This harness instead runs the code under test as a *subprocess tree* inside
# a bubblewrap sandbox:
#   - External commands (sbatch, srun, scancel, squeue, scontrol, dig, hf,
#     uv, module, vllm, gcc/g++) are intercepted via PATH shims
#     (tests/bash/shims/*), each a standalone executable — not sourced bash
#     functions — so real argument-passing/exit-code semantics are tested.
#   - Real system tools we are NOT trying to mock (bash, jq, yq, tar, awk,
#     sed, grep, python3, curl, coreutils, ...) are bound in read-only from
#     the host, so tests run against the *actual* yq/jq behaviour rather
#     than an idealised stand-in. This is deliberate: it is how issues 7-9
#     in design/issues.md were discovered (real yq 3.4.1 vs the v4 jq-style
#     syntax used in utils.sh).
#   - `--unshare-net` blocks all real network access (loopback still works,
#     so a mock vLLM HTTP server on localhost is reachable) — this makes it
#     impossible for a test to accidentally hit a real HPC, HuggingFace, or
#     NVIDIA download URL.
#   - `--unshare-pid` + `--die-with-parent` gives each test its own PID
#     namespace with bwrap itself acting as the reaper (pid 1). When the
#     sandboxed command exits, EVERY process it spawned (monitors, mock vLLM
#     servers, backgrounded mock srun children) is killed automatically —
#     no leaked background processes, even on test failure/crash.
#
# Two profiles are provided, matching where a script actually runs:
#   login   — no SLURM_* env vars. For testing the top-level ivllm-*.sh
#             wrapper scripts that run on the LOGIN node and submit work
#             (sbatch/srun) to the scheduler.
#   compute — SLURM_JOB_ID/SLURM_NODEID/SLURM_NNODES/SLURM_JOB_NODELIST/
#             COMPUTE_HOSTNAME set. For testing code that runs *inside* a
#             SLURM allocation: lockfile state transitions, the monitor
#             triad, exit traps/signals, run_head_vllm.sh/run_worker_vllm.sh.
#
# Usage (low level):
#   sandbox_run compute "$work_dir" [KEY=VALUE ...] -- bash /work/project/engine/... args
#
# Usage (high level, mirrors the old run_test() pattern):
#   sandbox_run_test "my test name" compute '
#       create_status_pending "job1" "model1" 30 >/dev/null
#       assert_status "$(resolve_job_status job1)" "pending"
#   '
#
# design/prototype/*.sh scripts are also reachable inside the sandbox, at
# /work/prototype/<name>.sh — `source /work/prototype/ivllm-bench.sh` (say)
# to unit test a prototype's functions before it's promoted to src/engine/.
#
# See design/testing.md for the full architecture writeup.

SANDBOX_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SANDBOX_TESTS_DIR="$(cd "$SANDBOX_LIB_DIR/.." && pwd)"
SANDBOX_REPO_ROOT="$(cd "$SANDBOX_TESTS_DIR/../.." && pwd)"
SANDBOX_ENGINE_SRC="$SANDBOX_REPO_ROOT/src/engine"
SANDBOX_SHIMS_DIR="$SANDBOX_TESTS_DIR/shims"
# Prototype scripts under design/prototype/ aren't part of src/engine yet
# (see AGENTS.md: design/ scripts are instructional, not production) but
# still need real, non-mocked sandbox testing to validate their bash logic
# before promotion — bound separately at /work/prototype/ rather than
# folded into the src/engine bind loop below, so it's obvious from a test
# body which category a sourced script falls into. Missing file is not
# fatal here (only tests that actually source it will notice).
SANDBOX_PROTOTYPE_DIR="$SANDBOX_REPO_ROOT/design/prototype"

if ! command -v bwrap >/dev/null 2>&1; then
    echo "FATAL: bwrap (bubblewrap) not found on PATH — install bubblewrap to run sandboxed bash tests" >&2
    exit 1
fi

if [[ ! -d "$SANDBOX_ENGINE_SRC" ]]; then
    echo "FATAL: expected engine source at $SANDBOX_ENGINE_SRC" >&2
    exit 1
fi

# Real host tools that must be reachable inside the sandbox. /usr, /bin,
# /lib, /lib64, /etc are bound wholesale (see _sandbox_build_args), so this
# list only matters for tools installed outside those roots (e.g. a
# user-local `yq` install under ~/.local/bin).
_SANDBOX_PASSTHROUGH_TOOLS=(bash sh jq yq tar gzip awk sed grep curl python3 date)
# Note: 'git' is deliberately NOT in this list — it is always shimmed (see
# tests/bash/shims/git) since real git clones are out of scope for sandboxed
# tests. 'wget' is likewise always shimmed.

# Populates the global SANDBOX_ARGS array with the profile-independent
# bwrap arguments (system binds, isolation flags). Callers append
# work-dir/profile-specific binds and env after calling this.
_sandbox_build_args() {
    SANDBOX_ARGS=(
        --ro-bind /usr /usr
        --ro-bind /bin /bin
    )
    [[ -d /lib ]] && SANDBOX_ARGS+=(--ro-bind /lib /lib)
    [[ -d /lib64 ]] && SANDBOX_ARGS+=(--ro-bind /lib64 /lib64)
    [[ -d /etc ]] && SANDBOX_ARGS+=(--ro-bind /etc /etc)

    local tool path dir
    local -A seen=()
    SANDBOX_EXTRA_PATH_DIRS=()
    for tool in "${_SANDBOX_PASSTHROUGH_TOOLS[@]}"; do
        path=$(command -v "$tool" 2>/dev/null) || continue
        dir=$(dirname "$path")
        case "$dir" in
            /usr|/usr/*|/bin|/lib|/lib64) continue ;;
        esac
        [[ -n "${seen[$dir]:-}" ]] && continue
        seen[$dir]=1
        SANDBOX_ARGS+=(--ro-bind "$dir" "$dir")
        SANDBOX_EXTRA_PATH_DIRS+=("$dir")
    done

    SANDBOX_ARGS+=(
        --proc /proc
        --dev /dev
        --tmpfs /local
        --tmpfs /tmp
        --unshare-net
        --unshare-pid
        --die-with-parent
        --new-session
        --clearenv
    )
}

# Builds a ":dir1:dir2" suffix from any extra tool directories resolved by
# _sandbox_build_args (e.g. a user-local yq install), to append to PATH.
_sandbox_extra_path_suffix() {
    local d suffix=""
    for d in "${SANDBOX_EXTRA_PATH_DIRS[@]:-}"; do
        [[ -z "$d" ]] && continue
        suffix+=":$d"
    done
    echo "$suffix"
}

# Run a command inside the sandbox.
# Usage: sandbox_run <login|compute> <host_work_dir> [KEY=VALUE ...] -- <command...>
#
# host_work_dir is bound read-write at /work inside the sandbox, so results
# (status.json, logs, calls.log) remain inspectable on the host after the
# sandboxed command exits. It is created fresh (mktemp -d) by the caller;
# sandbox_run creates the writable sub-directories it needs under it.
sandbox_run() {
    local profile="$1"; shift
    local work_dir="$1"; shift

    local -a extra_env=()
    while [[ $# -gt 0 && "$1" != "--" ]]; do
        extra_env+=("$1")
        shift
    done
    if [[ "${1:-}" == "--" ]]; then shift; fi
    local -a cmd=("$@")

    if [[ ${#cmd[@]} -eq 0 ]]; then
        echo "sandbox_run: no command given" >&2
        return 2
    fi

    mkdir -p \
        "$work_dir/project/engine/jobs" \
        "$work_dir/project/engine/vllm" \
        "$work_dir/project/engine/nvhpc" \
        "$work_dir/home/.cache"

    _sandbox_build_args # populates SANDBOX_ARGS

    local -a args=("${SANDBOX_ARGS[@]}")
    args+=(
        --bind "$work_dir" /work
    )

    # Bind each top-level entry of src/engine individually (rather than the
    # whole tree in one ro-bind) so that jobs/, vllm/, and nvhpc/ — created
    # writable above, as real host directories under $work_dir — remain
    # writable. A single ro-bind of the whole tree would make its *entire*
    # subtree read-only, including those runtime-data directories, since
    # bwrap cannot mkdir a new mountpoint inside an already read-only mount.
    local entry name
    for entry in "$SANDBOX_ENGINE_SRC"/*; do
        name=$(basename "$entry")
        case "$name" in
            jobs | vllm | nvhpc) continue ;; # these come from the writable dirs above
        esac
        args+=(--ro-bind "$entry" "/work/project/engine/$name")
    done

    args+=(
        --ro-bind "$SANDBOX_TESTS_DIR/lib" /work/testlib
        --ro-bind "$SANDBOX_SHIMS_DIR" /work/shims
        --ro-bind "$SANDBOX_TESTS_DIR/fixtures" /work/fixtures
    )
    [[ -d "$SANDBOX_PROTOTYPE_DIR" ]] && args+=(--ro-bind "$SANDBOX_PROTOTYPE_DIR" /work/prototype)

    args+=(
        --setenv HOME /work/home
        --setenv PATH "/work/shims:/usr/local/bin:/usr/bin:/bin$(_sandbox_extra_path_suffix)"
        --setenv IVLLM_PROJECTDIR /work/project
        --setenv IVLLM_TEST_CALL_LOG /work/calls.log
        --setenv IVLLM_TIME_FMT "+%Y-%m-%d %H:%M"
        --setenv IVLLM_CHECK_INTERVAL_SECS "${MOCK_CHECK_INTERVAL_SECS:-1}"
        --setenv TERM dumb
        --setenv LANG C.UTF-8
        --setenv LC_ALL C.UTF-8
        --chdir /work/project/engine
    )

    case "$profile" in
        login)
            : # no SLURM_* env vars — this is the login-node context
            ;;
        compute)
            args+=(
                --setenv SLURM_JOB_ID "${MOCK_SLURM_JOB_ID:-99999}"
                --setenv SLURM_NODEID "${MOCK_SLURM_NODEID:-0}"
                --setenv SLURM_NNODES "${MOCK_SLURM_NNODES:-1}"
                --setenv SLURM_JOB_NODELIST "${MOCK_SLURM_JOB_NODELIST:-sandbox-node0}"
                --setenv SLURM_JOB_START_TIME "$(date +%s)"
                --setenv SLURM_JOB_END_TIME "$(($(date +%s) + 3600))"
                --setenv COMPUTE_HOSTNAME "${MOCK_COMPUTE_HOSTNAME:-sandbox-node0}"
            )
            ;;
        *)
            echo "sandbox_run: unknown profile '$profile' (expected login|compute)" >&2
            return 2
            ;;
    esac

    local kv
    for kv in "${extra_env[@]}"; do
        args+=(--setenv "${kv%%=*}" "${kv#*=}")
    done

    # Guard against a runaway test (e.g. a monitor loop that never sees the
    # condition it's waiting for) hanging the whole test suite. bwrap's
    # --die-with-parent means SIGTERM from `timeout` tears down the entire
    # sandboxed process tree, not just the top command.
    timeout "${SANDBOX_TIMEOUT_SECS:-30}" bwrap "${args[@]}" -- "${cmd[@]}"
}

# High-level test runner: builds a temp work dir, writes the given bash
# snippet to a script that sources utils.sh + assertions.sh, runs it inside
# the sandbox, and reports pass/fail. Mirrors the pre-rewrite run_test()
# helper in tests/bash/lib/test-utils.sh, but the body now runs inside a
# bwrap sandbox instead of a bare subshell.
#
# Usage: sandbox_run_test <name> <login|compute> <body> [KEY=VALUE ...]
#
# The body runs with `set -uo pipefail` (not -e — tests use assert_* helpers
# that return non-zero on failure and are expected to be checked explicitly,
# same convention as the rest of the bash framework).
FAIL="${FAIL:-0}"

sandbox_run_test() {
    local name="$1" profile="$2" body="$3"
    shift 3
    local -a extra_env=("$@")

    local work_dir
    work_dir=$(mktemp -d)

    cat > "$work_dir/_test_body.sh" <<SCRIPT
#!/bin/bash
set -uo pipefail
source /work/project/engine/lib/utils.sh
source /work/testlib/assertions.sh

$body
SCRIPT
    chmod +x "$work_dir/_test_body.sh"

    if sandbox_run "$profile" "$work_dir" "${extra_env[@]}" -- bash /work/_test_body.sh \
        > "$work_dir/output.log" 2>&1; then
        echo "✓ $name"
    else
        echo "✗ $name"
        echo "  --- test output ---"
        sed 's/^/  /' "$work_dir/output.log"
        if [[ -f "$work_dir/calls.log" ]]; then
            echo "  --- shim call log ---"
            sed 's/^/  /' "$work_dir/calls.log"
        fi
        FAIL=1
    fi

    if [[ "${KEEP_TEST_DIRS:-0}" == "1" ]]; then
        echo "  (kept: $work_dir)"
    else
        rm -rf "$work_dir"
    fi

    # Allow the kernel to release namespaces created by the previous bwrap
    # instance (--unshare-net, --unshare-pid, --new-session).  Without this
    # delay, running > ~7 sandboxed tests in quick succession can hit kernel
    # namespace allocation limits (ENOSPC / "setenv failed"), especially in
    # unprivileged sandbox environments.
    sleep 0.2
}

# Assert a shim was invoked with a matching substring, reading calls.log from
# a sandbox_run_test work dir. Mostly useful when writing tests with
# sandbox_run directly (not sandbox_run_test) so you keep the work_dir handle.
# Usage: sandbox_assert_called "$work_dir" "sbatch" "--job-name qwen36"
sandbox_assert_called() {
    local work_dir="$1" tool="$2" substr="$3"
    grep -F "$tool " "$work_dir/calls.log" 2>/dev/null | grep -qF "$substr"
}
