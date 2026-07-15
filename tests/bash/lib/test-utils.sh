#!/bin/bash
# shellcheck disable=SC2155
#
# test-utils.sh — Reusable mock infrastructure for bash framework tests.
#
# Source this file at the top of any test file to get mock implementations
# of srun, scancel, and vllm, plus assertion helpers.
#
# Environment variables for configuring mocks:
#   MOCK_VLLM_DELAY      Seconds to delay before /health responds (default: 0)
#   MOCK_VLLM_CRASH_AFTER Number of requests before mock vLLM crashes (default: 0 = never)
#   MOCK_SRUN_FAIL_EXIT   Exit code for mock_srun_fail (default: 1)
#   MOCK_SRUN_SLEEP       Seconds for mock srun to simulate startup time (default: 0)

# ── Mock scancel ────────────────────────────────────────────────────────────

# Overrides the real scancel. Logs the call and no-ops.
scancel() {
    local jobid=$1
    echo "[mock] scancel $jobid"
}

# ── Mock srun ───────────────────────────────────────────────────────────────

# Overrides the real srun. Finds the 'vllm' command in the argument list
# and spawns it in the background. Captures PID in MOCK_SRUN_PID.
srun() {
    # Find "vllm" in the arguments
    local args=("$@")
    local vllm_found=false
    local vllm_args=()
    local capturing=false

    for arg in "${args[@]}"; do
        if [[ "$arg" == "vllm" ]]; then
            vllm_found=true
            capturing=true
            vllm_args+=("$arg")
        elif $capturing; then
            vllm_args+=("$arg")
        fi
    done

    if $vllm_found; then
        "${vllm_args[@]}" &
        MOCK_SRUN_PID=$!
        echo "[mock] srun: started PID $MOCK_SRUN_PID: ${vllm_args[*]}"
        # Forward signals to the child
        trap 'kill -SIGTERM $MOCK_SRUN_PID 2>/dev/null' SIGTERM SIGINT
        trap 'kill -SIGUSR1 $MOCK_SRUN_PID 2>/dev/null' SIGUSR1
        trap 'kill -SIGUSR2 $MOCK_SRUN_PID 2>/dev/null' SIGUSR2
        trap 'kill -SIGKILL $MOCK_SRUN_PID 2>/dev/null' EXIT
        wait $MOCK_SRUN_PID
    else
        echo "[mock] srun: no vllm command found in arguments, skipping"
        return 0
    fi
}

# Mock srun that fails immediately with a configurable exit code.
srun_fail() {
    local exit_code="${MOCK_SRUN_FAIL_EXIT:-1}"
    echo "[mock] srun_fail: exiting with code $exit_code"
    return "$exit_code"
}

# ── Mock sbatch ─────────────────────────────────────────────────────────────

# Overrides the real sbatch. Returns a fake job ID for testing.
sbatch() {
    local script_path=""
    # Find the last positional argument (the script path)
    for arg in "$@"; do
        if [[ "$arg" != -* ]]; then
            script_path="$arg"
        fi
    done

    local fake_job_id="${MOCK_SBATCH_JOB_ID:-99999}"
    echo "[mock] sbatch: submitted batch job $fake_job_id ($script_path)"
    echo "Submitted batch job $fake_job_id"
}

# ── Mock vLLM HTTP server ───────────────────────────────────────────────────

# Starts a lightweight Python HTTP server that responds to /health,
# /v1/models, and /v1/chat/completions.
# Configure with MOCK_VLLM_DELAY and MOCK_VLLM_CRASH_AFTER env vars.
mock_vllm() {
    local delay="${MOCK_VLLM_DELAY:-0}"
    local crash_after="${MOCK_VLLM_CRASH_AFTER:-0}"

    # Extract --model and --port from arguments
    local model="test-model"
    local port="8000"
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --model) model="$2"; shift 2 ;;
            --port) port="$2"; shift 2 ;;
            --config) shift 2 ;;  # skip vllm config file
            --host) shift 2 ;;    # skip host
            --served-model-name) shift 2 ;;  # skip served model names
            --) shift; break ;;
            *) shift ;;
        esac
    done

    # Inject delay and crash config into the Python script
    exec python3 -c "
import http.server, json, os, sys, atexit, signal
from datetime import datetime

delay = $delay
crash_after = $crash_after
request_count = 0

# Handle signals from parent srun mock
signal.signal(signal.SIGUSR1, lambda *_: sys.exit(200))
signal.signal(signal.SIGUSR2, lambda *_: sys.exit(201))

# Simulate startup delay
if delay > 0:
    import time
    time.sleep(delay)

def handle_exit():
    sys.stderr.write('Mock vLLM: shutdown\n')
    sys.stderr.flush()
atexit.register(handle_exit)

class MockHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        global request_count
        timestamp = datetime.now().strftime('%m-%d %H:%M:%S')
        request_count += 1

        if crash_after > 0 and request_count > crash_after:
            sys.stderr.write('Mock vLLM: crashing after $crash_after requests\n')
            sys.stderr.flush()
            sys.exit(1)

        sys.stderr.write(f'[{timestamp}] api call {self.path}\\n')
        sys.stderr.flush()

        if self.path == '/health':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(b'{}')
        elif self.path == '/v1/models':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            body = json.dumps({'object': 'list', 'data': [{'id': '$model', 'object': 'model'}]})
            self.wfile.write(body.encode())
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        global request_count
        timestamp = datetime.now().strftime('%m-%d %H:%M:%S')
        request_count += 1

        if crash_after > 0 and request_count > crash_after:
            sys.stderr.write('Mock vLLM: crashing after $crash_after requests\\n')
            sys.stderr.flush()
            sys.exit(1)

        if self.path == '/v1/chat/completions':
            sys.stderr.write(f'[{timestamp}] api call /v1/chat/completions\\n')
            sys.stderr.flush()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            body = json.dumps({'object': 'list', 'data': [{'result': 'the response'}]})
            self.wfile.write(body.encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, fmt, *args):
        pass

print(f'Mock vLLM listening on port $port, model=$model, delay={delay}s, crash_after={crash_after}', flush=True)
try:
    http.server.HTTPServer(('0.0.0.0', $port), MockHandler).serve_forever()
except KeyboardInterrupt:
    sys.exit(130)
"
}

# ── Assertion helpers ───────────────────────────────────────────────────────

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

# ── Test environment setup helper ───────────────────────────────────────────

# Creates a temporary ENGINE_DIR and sets up the environment for testing.
# Returns the path via stdout. Caller should save to a local var.
# Usage: local engine_dir=$(setup_test_env)
setup_test_env() {
    local dir
    dir=$(mktemp -d)
    export ENGINE_DIR="$dir/engine"
    mkdir -p "$ENGINE_DIR/jobs"
    echo "$dir"
}

# Cleans up a test environment created by setup_test_env.
# Usage: cleanup_test_env "$engine_dir"
cleanup_test_env() {
    rm -rf "$1"
}
