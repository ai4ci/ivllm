#!/bin/bash
# shellcheck disable=SC2155

scancel() {
    local jobid=$1
    # mock scancel for testing
    echo "[test] mock cancelling slurm job $jobid"
}

export STARTUP_DELAY=10


srun() {

    # Loop until the first argument ($1) becomes exactly "vllm"
    while [[ $# -gt 0 && "$1" != "vllm" ]]; do
        shift
    done

    # If "vllm" was found in the arguments, execute it and everything after it
    if [[ $# -gt 0 ]]; then
        "$@" & SS_PID=$!
        echo "[test] srun: 'mock vllm' started with PID $SS_PID: with command: $*"
        trap 'echo "[test] srun: SIGKILL $SS_PID"; kill -s SIGKILL $SS_PID' SIGQUIT SIGTERM SIGHUP SIGINT
        trap 'echo "[test] srun: SIGUSR1 $SS_PID"; kill -s SIGUSR1 $SS_PID' SIGUSR1
        trap 'echo "[test] srun: SIGUSR2 $SS_PID"; kill -s SIGUSR2 $SS_PID' SIGUSR2
        trap 'echo "[test] srun: SIGKILL (EXIT) $SS_PID"; kill -s SIGKILL $SS_PID' ERR EXIT
        wait $SS_PID
    else
        echo "[test] mock srun error: 'vllm' command not found in arguments."
        return 1
    fi
}

vllm() {

    # 1. Initialize local variables to hold the extracted arguments
    local model=""
    local serverPort=""

    # 2. Parse arguments loop
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --model)
                model="$2"
                shift 2  # Consume both '--model' and its value
                ;;
            --port)
                serverPort="$2"
                shift 2  # Consume both '--port' and its value
                ;;
            *)
                shift    # Ignore and skip any other parameter
                ;;
        esac
    done

    echo "Mock vLLM: starting HTTP server on port $serverPort for model $model..."
    echo "Mock vLLM: Initialising..."
    export VLLM_CACHE_ROOT=$(resolve_localdir "test-job")
    dd if=/dev/urandom of="$VLLM_CACHE_ROOT/random_data.bin" bs=1024 count=1
    sleep $STARTUP_DELAY

    echo "Mock vLLM: ready"

    # Capture the true exit code first, log it, then exit with it

    exec python3 << PYEOF
import http.server, json, os, sys, atexit
from datetime import datetime

# Handle the exit/shutdown logging cleanly within Python
def handle_exit():
    sys.stderr.write("Mock vLLM: shutdown\n")
    sys.stderr.flush()

atexit.register(handle_exit)

class MockHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        timestamp = datetime.now().strftime("%m-%d %H:%M:%S")
        if self.path == "/health":
            sys.stderr.write(f"[{timestamp}] api call /health\n")
            sys.stderr.flush()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b"{}")
        elif self.path == "/v1/models":
            sys.stderr.write(f"[{timestamp}] api call /v1/models\n")
            sys.stderr.flush()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            body = json.dumps({"object": "list", "data": [{"id": "$model", "object": "model"}]})
            self.wfile.write(body.encode())
        else:
            self.send_response(404)
            self.end_headers()
    def do_POST(self):
        timestamp = datetime.now().strftime("%m-%d %H:%M:%S")
        if self.path == "/v1/chat/completions":
            # test log message
            sys.stderr.write(f"[{timestamp}] api call /v1/completions\n")
            sys.stderr.flush()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            body = json.dumps({"object": "list", "data": [{"result": "the response"}]})
            self.wfile.write(body.encode())
        else:
            self.send_response(404)
            self.end_headers()
    def log_message(self, fmt, *args):
        pass

print(f"Mock vLLM listening on port $serverPort, model=$model", flush=True)

try:
    http.server.HTTPServer(("0.0.0.0", $serverPort), MockHandler).serve_forever()
except KeyboardInterrupt:
    sys.exit(130)
PYEOF

}


# set -x
set -euo pipefail

# TODO: in the real thing this needs to resolve to the vllm_logs.json path
export VLLM_LOGGING_CONFIG_PATH="vllm_logs.json"

# main execution flow in a slurm sbatch script or via srun:
export PROJECTDIR="./test_proj"
export LOCALDIR="./test_cache"
export JOBNAME="test_job"
export SLURM_JOB_ID=123456
export SLURM_NODEID=0
export COMPUTE_HOSTNAME="nid1234"
export SLURM_JOB_START_TIME="$(date +%s)"
export SLURM_JOB_END_TIME="$(date +%s)"

rm -rf $PROJECTDIR
rm -rf $LOCALDIR

source "prototype.sh"

# prior to slurm allocation
create_status_pending "$JOBNAME" test_model 1

# waiting in the slurm queue
# slurm sbatch job:
if is_status "$JOBNAME" "pending"; then

    PORT=$(get_job_status_setting $JOBNAME "serverPort")
    MODEL=$(get_job_status_setting $JOBNAME "model")

    # launch and background long running process
    (

        LOG=$(resolve_job_log $JOBNAME)


        # The long running process
        srun vllm --model "$MODEL" --port "$PORT" > "$LOG" 2>&1 & VLLM_PID=$!
        sleep 1

        echo "[test] vllm pid: $VLLM_PID; jobid: $SLURM_JOB_ID"
        update_status_initialise "$JOBNAME" "$VLLM_PID"
        setup_traps "$JOBNAME"
        echo "[test] initialised job $JOBNAME"

        wait $VLLM_PID
    ) & VLLM_PARENT_PID=$!

    # Waits for user cancel via lockfile instruction, slurm timeout, or idle timeout:
    # This will idle timeout eventually
    echo "[test] initialised head monitor for job $JOBNAME"
    monitor_head "$JOBNAME" "$VLLM_PARENT_PID" & MONITOR_PID=$!

    echo "[test] initialised startup monitor for job $JOBNAME"
    # poll health until api responding
    monitor_startup "$JOBNAME"

    # Outside of slurm
    # simulate a request.
    curl "http://localhost:$PORT/v1/chat/completions" \
            -H "Content-Type: application/json" \
            -d "{
                \"model\": \"$MODEL\",
                \"messages\": [{\"role\": \"user\", \"content\": \"Hello.\"}]
            }"

    # simulate a user requested shutdown:
    request_cancel "$JOBNAME"

    wait $VLLM_PARENT_PID
    wait $MONITOR_PID

else
    echo "[test] unable to get lock for job $JOBNAME."
fi

# Second model:
# create_status_pending "lockfile2.json" test_job_2 test_model_2 8001 1 vllm2.log

