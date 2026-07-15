#!/bin/bash
# shellcheck disable=SC2155
#
# utils.sh — Shared bash library for vLLM job lifecycle on Isambard.
#
# This file is part of the isambard-vllm bash framework. It provides lockfile
# management, cache operations, the monitor triad, and graceful shutdown.
#
# Source this file from SLURM scripts and test files. It does NOT set -e;
# the caller controls error handling.
#
# Required environment variables:
#   ENGINE_DIR     — Root of the job engine directory (e.g. $PROJECTDIR/engine)
#   SLURM_NODEID   — Set by SLURM (0 = head node, >0 = worker)
#   SLURM_JOB_ID   — Set by SLURM
#
# Optional environment variables:
#   VLLM_TIME_FMT         — Date format for log timestamp matching (default: +%Y-%m-%d %H:%M)
#   CHECK_INTERVAL_SECS   — Monitor polling interval (default: 10)
#   COMPUTE_HOSTNAME      — Hostname of the compute node (default: $(hostname))

# ── Configurable defaults ───────────────────────────────────────────────────

export VLLM_TIME_FMT="${VLLM_TIME_FMT:-+%Y-%m-%d %H:%M}"
export CHECK_INTERVAL_SECS="${CHECK_INTERVAL_SECS:-10}"
export TARGET_ENDPOINTS=(
    "/v1/models"
    "/v1/chat"
    "/v1/chat/completions"
    "/v1/responses"
    "/v1/completions"
    "/v1/messages"
)

# ── Path helpers ───────────────────────────────────────────────────────────

# Resolve the per-node local working directory (RAM-backed tmpfs).
# Creates the directory if it doesn't exist.
# Usage: local localdir=$(resolve_localdir "$job")
resolve_localdir() {
    local job="$1"
    local node_local

    if [ -z "${LOCALDIR:-}" ]; then
        export LOCALDIR="/local/user/$UID"
        mkdir -p "$LOCALDIR" 2>/dev/null || true
        chmod 700 "$LOCALDIR" 2>/dev/null || true
    fi

    node_local="$LOCALDIR/$(hostname -s)/$job"
    mkdir -p "$node_local"
    echo "$node_local"
}

# Resolve the path to a file in a job's directory.
# Usage: local path=$(resolve_location "$job" "filename")
resolve_location() {
    local job="$1"
    local file="${2:-}"
    echo "$ENGINE_DIR/jobs/$job/$file"
}

# Resolve the path to a job's JIT cache tarball.
resolve_cachetar() {
    resolve_location "$1" "jit-cache.tar.gz"
}

# Resolve the path to a job's lockfile (status.json).
resolve_lockfile() {
    resolve_location "$1" "status.json"
}

# Resolve the path to a job's log file (per-node).
resolve_logfile() {
    local node="${SLURM_NODEID:-0}"
    resolve_location "$1" "vllm.$node.log"
}

# Read a field from a job's lockfile using jq.
# Usage: local value=$(resolve_setting "$job" ".fieldName")
resolve_setting() {
    local lockfile
    lockfile=$(resolve_lockfile "$1")
    jq -r "$2" "$lockfile" 2>/dev/null || echo "null"
}

# ── Lockfile state machine ─────────────────────────────────────────────────

# Create a lockfile with status "pending". Run on LOGIN node before sbatch.
# Uses set -C (noclobber) for atomic creation — fails if lockfile exists.
# Usage: create_status_pending "$job" "$model" "$idle_timeout"
create_status_pending() {
    local job="$1"
    local model="$2"
    local idle_timeout="${3:-30}"
    local lockfile
    local server_port

    lockfile=$(resolve_lockfile "$job")
    mkdir -p "$(resolve_location "$job")"

    # Generate random high port for the vLLM server
    server_port=$(shuf -i 49152-65535 -n 1)

    echo "[startup] creating lockfile for job $job (port=$server_port)" >&2

    # Atomic create with noclobber
    (
        set -C
        jq -n \
            --arg job_name "$job" \
            --arg model "$model" \
            --argjson server_port "$server_port" \
            --argjson idle_timeout "$idle_timeout" \
            --arg req_time "$(date -Iseconds)" \
            '{status: "pending", jobName: $job_name, model: $model, serverPort: $server_port, requestedTime: $req_time, idleTimeout: $idle_timeout}' \
            > "$lockfile"
    ) 2>/dev/null || {
        echo "[startup] ERROR: lockfile already exists for job $job" >&2
        return 1
    }

    echo "$server_port"
}

# Update lockfile with SLURM allocation details. Run on head compute node.
# Usage: update_status_initialise "$job" "$vllm_pid"
update_status_initialise() {
    local job="$1"
    local vllm_pid="$2"
    local lockfile
    local log
    local hostname="${COMPUTE_HOSTNAME:-$(hostname)}"

    lockfile=$(resolve_lockfile "$job")
    log=$(resolve_logfile "$job")
    touch "$log"

    if (( SLURM_NODEID == 0 )); then
        echo "[startup] slurm job allocated for job $job (SLURM_JOB_ID=$SLURM_JOB_ID)"

        jq \
            --argjson vllm_pid "$vllm_pid" \
            --argjson slurm_job_id "$SLURM_JOB_ID" \
            --arg compute_hostname "$hostname" \
            --arg start_time "$(date -d "@${SLURM_JOB_START_TIME:-$(date +%s)}" -Iseconds 2>/dev/null || date -Iseconds)" \
            --arg stop_time "$(date -d "@${SLURM_JOB_END_TIME:-$(date +%s)}" -Iseconds 2>/dev/null || date -Iseconds)" \
            '.status = "initialising" | .slurmJobId = $slurm_job_id | .computeHostname = $compute_hostname | .startTime = $start_time | .stopTime = $stop_time | .vllmPid = $vllm_pid' \
            "$lockfile" > "$lockfile.tmp" && mv "$lockfile.tmp" "$lockfile"
    fi
}

# Mark job as running. Run on head compute node when vLLM health check passes.
# Usage: update_status_running "$job"
update_status_running() {
    local job="$1"
    local lockfile

    lockfile=$(resolve_lockfile "$job")

    if (( SLURM_NODEID == 0 )); then
        echo "[startup] job $job is running."
        jq '.status = "running"' "$lockfile" > "$lockfile.tmp" && mv "$lockfile.tmp" "$lockfile"
    fi
}

# Mark job as cleanly stopped. Used by exit trap for user cancel, idle timeout.
# Usage: update_status_clean_shutdown "$job"
update_status_clean_shutdown() {
    local job="$1"
    local lockfile

    lockfile=$(resolve_lockfile "$job")

    if (( SLURM_NODEID == 0 )); then
        echo "[shutdown] clean shutdown for job $job."
        jq \
            --arg stop_time "$(date -Iseconds)" \
            '.status = "stopped" | .stopTime = $stop_time | .exitCode = "0"' \
            "$lockfile" > "$lockfile.tmp" && mv "$lockfile.tmp" "$lockfile"
    fi
}

# Mark job as failed. Used by exit trap for startup failures and crashes.
# Usage: update_status_unclean_shutdown "$job" "reason" exit_code
update_status_unclean_shutdown() {
    local job="$1"
    local reason="$2"
    local exit_code="$3"
    local lockfile

    lockfile=$(resolve_lockfile "$job")

    if (( SLURM_NODEID == 0 )); then
        echo "[shutdown] unclean shutdown for job $job due to $reason ($exit_code)."
        jq \
            --arg error "$reason" \
            --argjson exit_code "$exit_code" \
            --arg stop_time "$(date -Iseconds)" \
            '.status = "failed" | .reason = $error | .stopTime = $stop_time | .exitCode = $exit_code' \
            "$lockfile" > "$lockfile.tmp" && mv "$lockfile.tmp" "$lockfile"
    fi
}

# Request cancel (user-initiated). Writes "cancel" to lockfile for the monitor
# to detect. Can be run from LOGIN node or any client.
# Usage: request_cancel "$job"
request_cancel() {
    local job="$1"
    local lockfile

    lockfile=$(resolve_lockfile "$job")

    if [ ! -f "$lockfile" ]; then
        echo "[cancel] ERROR: lockfile not found for job $job"
        return 1
    fi

    echo "[cancel] requesting cancel for job $job."
    jq '.status = "cancel"' "$lockfile" > "$lockfile.tmp" && mv "$lockfile.tmp" "$lockfile"
}

# Write a reason string to the lockfile without changing status.
# Usage: update_reason "$job" "reason text"
update_reason() {
    local job="$1"
    local reason="$2"
    local lockfile

    lockfile=$(resolve_lockfile "$job")

    if (( SLURM_NODEID == 0 )); then
        echo "[shutdown] reason for job $job: $reason."
        jq --arg reason "$reason" '.reason = $reason' "$lockfile" > "$lockfile.tmp" && mv "$lockfile.tmp" "$lockfile"
    fi
}

# Check if a job has a specific status. Returns 0 if match, 1 otherwise.
# Usage: if is_status "$job" "running"; then ...
is_status() {
    local lockfile
    lockfile=$(resolve_lockfile "$1")
    [ ! -f "$lockfile" ] && return 1
    jq -e --arg test "$2" 'has("status") and .status == $test' "$lockfile" > /dev/null 2>&1
}

# ── Shutdown and cleanup ───────────────────────────────────────────────────

# Graceful shutdown exit trap. Handles all exit codes:
#   200 = SIGUSR1 (SLURM timeout)
#   201 = SIGUSR2 (user cancel or idle timeout)
#   0   = normal exit
#   other = crash (check status to distinguish startup vs runtime)
# Usage: trap 'tidy_up "$job" $?' EXIT
tidy_up() {
    local job="$1"
    local exit_code="$2"
    local pid
    local slurm_job_id

    pid=$(resolve_setting "$job" "vllmPid")
    slurm_job_id=$(resolve_setting "$job" "slurmJobId")

    echo "[shutdown] shutting down job $job (vllm: $pid, slurm: $slurm_job_id, exit: $exit_code)"

    # Kill vLLM process if still alive
    if [ "$pid" != "null" ] && [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
        echo "[shutdown] killing vLLM process $pid"
        kill -15 "$pid" 2>/dev/null
        sleep 2
        if kill -0 "$pid" 2>/dev/null; then
            kill -9 "$pid" 2>/dev/null
        fi
    fi

    # Update lockfile based on exit code
    case "$exit_code" in
        200)
            # SIGUSR1 — SLURM timeout
            echo "[shutdown] received SIGUSR1: SLURM timeout"
            update_reason "$job" "SLURM timeout"
            update_status_clean_shutdown "$job"
            ;;
        201)
            # SIGUSR2 — user cancel or idle timeout
            echo "[shutdown] received SIGUSR2: user cancel or idle timeout"
            update_status_clean_shutdown "$job"
            ;;
        0)
            echo "[shutdown] vLLM terminated normally"
            ;;
        *)
            if is_status "$job" "initialising"; then
                echo "[shutdown] exit code $exit_code: vLLM failed to start"
                update_status_unclean_shutdown "$job" "failed to start" "$exit_code"
            else
                echo "[shutdown] exit code $exit_code: vLLM crashed during inference"
                update_status_unclean_shutdown "$job" "crashed during inference" "$exit_code"
            fi
            ;;
    esac

    # Cancel SLURM job
    if [ "$slurm_job_id" != "null" ] && [ -n "$slurm_job_id" ]; then
        echo "[shutdown] cancelling SLURM job $slurm_job_id"
        scancel "$slurm_job_id" 2>/dev/null || true
    fi

    exit 0
}

# Set up exit traps for the monitor triad.
# Usage: setup_traps "$job"
setup_traps() {
    local job="$1"
    trap 'tidy_up "'"$job"'" 200' SIGUSR1   # SLURM timeout
    trap 'tidy_up "'"$job"'" 201' SIGUSR2   # user cancel or idle timeout
    trap 'tidy_up "'"$job"'" $?' ERR
    trap 'tidy_up "'"$job"'" $?' EXIT
}

# Clear the per-node local working directory.
# Usage: clear_localdir "$job"
clear_localdir() {
    local localdir
    localdir=$(resolve_localdir "$1")
    echo "[cache] cleaning working directory: $localdir"
    if [ -d "$localdir" ]; then
        rm -rf "${localdir:?no localdir defined}/*"
    else
        mkdir -p "$localdir"
    fi
}

# ── Monitor: startup (foreground, head node only) ─────────────────────────

# Monitor that blocks until vLLM responds to /health, runs a warmup request,
# saves the JIT cache, and transitions status to "running".
# Usage: monitor_startup "$job" "$vllm_parent_pid"
monitor_startup() {
    local job="$1"
    local vllm_parent="$2"
    local server_port
    local model

    server_port=$(resolve_setting "$job" "serverPort")
    model=$(resolve_setting "$job" "model")

    # Only runs on head node
    if (( SLURM_NODEID != 0 )); then
        return 0
    fi

    while true; do
        if is_status "$job" "pending"; then
            echo "[startup] waiting for job $job to initialise"
            sleep "$CHECK_INTERVAL_SECS"
            continue
        fi

        if is_status "$job" "initialising"; then
            if curl -sf "http://localhost:$server_port/health" > /dev/null 2>&1; then
                echo "[startup] vLLM /health active — saving JIT cache"
                save_cache "$job"

                # Warmup: send a test request to trigger JIT compilation
                local max_retries=5
                local attempt=1
                local warmup_ok=1

                echo "[startup] sending warmup request..."
                while (( attempt <= max_retries )); do
                    if curl -sf "http://localhost:$server_port/v1/chat/completions" \
                        -H "Content-Type: application/json" \
                        -d "{\"model\": \"$model\", \"messages\": [{\"role\": \"user\", \"content\": \"Hello.\"}]}" \
                        > /dev/null 2>&1; then
                        warmup_ok=0
                        break
                    fi
                    echo "[startup] WARNING: warmup attempt $attempt/$max_retries failed, retrying..."
                    (( attempt++ ))
                    sleep 10
                done

                if (( warmup_ok == 0 )); then
                    echo "[startup] warmup complete — saving JIT cache"
                    save_cache "$job"
                    echo "[startup] job $job startup complete."
                    update_status_running "$job"
                    break
                else
                    echo "[startup] ERROR: warmup failed after $max_retries attempts"
                    update_reason "$job" "vLLM warmup failed"
                    kill -s SIGUSR2 "$vllm_parent" 2>/dev/null || exit 1
                    exit 1
                fi
            else
                echo "[startup] job $job waiting for vLLM /health"
                report_memory "$job"
                sleep "$CHECK_INTERVAL_SECS"
                continue
            fi
        else
            local status
            status=$(resolve_setting "$job" "status")
            echo "[startup] ERROR: job $job is in unexpected state $status"
            exit 1
        fi
    done

    echo "[startup] job $job startup monitor detached."
}

# ── Monitor: head node (background) ───────────────────────────────────────

# Background monitor that runs on the head compute node for the entire job
# lifetime. Checks for: lockfile deletion, cancel requests, vLLM process death,
# and idle timeout.
# Usage: monitor_head "$job" "$vllm_parent_pid" &
monitor_head() {
    local job="$1"
    local vllm_parent="$2"
    local lockfile
    local log
    local idle_timeout

    lockfile=$(resolve_lockfile "$job")
    log=$(resolve_logfile "$job")
    idle_timeout=$(resolve_setting "$job" "idleTimeout")

    if [ ! -f "$lockfile" ]; then
        echo "[head] FATAL: lockfile $lockfile missing on startup"
        kill -s SIGTERM "$vllm_parent" 2>/dev/null
        exit 1
    fi

    echo "[head] starting monitor (idle_timeout=$idle_timeout)..."

    while true; do
        # Lockfile deleted
        if [ ! -f "$lockfile" ]; then
            echo "[head] lockfile $lockfile has been deleted — shutting down"
            kill -s SIGTERM "$vllm_parent" 2>/dev/null
            exit 1
        fi

        # Terminal states — exit loop
        if is_status "$job" "failed" || is_status "$job" "stopped"; then
            break
        fi

        # Still pending — wait for SLURM allocation
        if is_status "$job" "pending"; then
            sleep "$CHECK_INTERVAL_SECS"
            continue
        fi

        local vllm_pid
        vllm_pid=$(resolve_setting "$job" "vllmPid")

        # vLLM process died
        if [ "$vllm_pid" != "null" ] && ! kill -0 "$vllm_pid" > /dev/null 2>&1; then
            echo "[head] vLLM process ($vllm_pid) has gone away"
            update_reason "$job" "lost contact with vLLM process"
            break
        fi

        # User requested cancel
        if is_status "$job" "cancel"; then
            echo "[head] user cancel request detected"
            update_reason "$job" "user cancel"
            break
        fi

        # Still initialising — skip idle checks
        if is_status "$job" "initialising"; then
            sleep "$CHECK_INTERVAL_SECS"
            continue
        fi

        # Running — check idle timeout
        if is_status "$job" "running" && [ "$idle_timeout" != "null" ] && [ "$idle_timeout" -ge 0 ]; then
            if [ ! -f "$log" ]; then
                echo "[head] log file $log is missing — shutting down"
                update_reason "$job" "missing log file"
                break
            fi

            # Build time patterns for the idle window
            local time_patterns=()
            for i in $(seq 0 "$idle_timeout"); do
                time_patterns+=("-e" "$(date -d "$i minutes ago" "$VLLM_TIME_FMT")")
            done

            local endpoint_patterns=()
            for endpoint in "${TARGET_ENDPOINTS[@]}"; do
                endpoint_patterns+=("-e" "$endpoint")
            done

            # Check for recent API requests (not /health)
            if tail -n 5000 "$log" 2>/dev/null | grep -F "${time_patterns[@]}" | grep -q -F "${endpoint_patterns[@]}"; then
                sleep "$CHECK_INTERVAL_SECS"
                continue
            fi

            echo "[head] no API requests for $idle_timeout minutes — shutting down"
            update_reason "$job" "idle timeout"
            break
        fi

        sleep "$CHECK_INTERVAL_SECS"
    done

    # Signal shutdown to parent
    kill -s SIGUSR2 "$vllm_parent" 2>/dev/null

    # Wait for parent to exit
    while kill -0 "$vllm_parent" 2>/dev/null; do
        sleep 2
    done

    clear_localdir "$job"
    echo "[head] monitor shutting down for job $job."
    exit 0
}

# ── Monitor: worker node (background) ─────────────────────────────────────

# Background monitor for worker nodes in multi-node jobs. Watches the lockfile
# and shuts down the local vLLM process if the job is no longer in a valid
# state.
# Usage: monitor_worker "$job" "$vllm_parent_pid" &
monitor_worker() {
    local job="$1"
    local vllm_parent="$2"
    local lockfile
    local node

    lockfile=$(resolve_lockfile "$job")
    node=${SLURM_NODEID:-0}

    # This monitor is for worker nodes only
    if [ "$node" -eq 0 ]; then
        echo "[worker] ERROR: worker monitor started on head node"
        kill -s SIGTERM "$vllm_parent" 2>/dev/null
        exit 1
    fi

    if [ ! -f "$lockfile" ]; then
        echo "[worker $node] FATAL: lockfile $lockfile missing on startup"
        kill -s SIGTERM "$vllm_parent" 2>/dev/null
        exit 1
    fi

    echo "[worker $node] starting lockfile monitor..."

    while true; do
        if [ ! -f "$lockfile" ]; then
            echo "[worker $node] lockfile deleted — shutting down"
            kill -s SIGTERM "$vllm_parent" 2>/dev/null
            exit 1
        fi

        if is_status "$job" "pending"; then
            sleep "$CHECK_INTERVAL_SECS"
            continue
        fi

        if is_status "$job" "initialising"; then
            report_memory "$job"
            sleep "$CHECK_INTERVAL_SECS"
            continue
        fi

        if ! is_status "$job" "running"; then
            echo "[worker $node] job is not running (status=$(resolve_setting "$job" ".status")) — shutting down"
            break
        fi

        sleep "$CHECK_INTERVAL_SECS"
    done

    kill -s SIGTERM "$vllm_parent" 2>/dev/null

    while kill -0 "$vllm_parent" 2>/dev/null; do
        sleep 2
    done

    clear_localdir "$job"
    echo "[worker $node] monitor shutting down."
    exit 0
}

# ── Resource monitoring ───────────────────────────────────────────────────

# Report memory and JIT cache usage for the current node.
# Usage: report_memory "$job"
report_memory() {
    local localdir
    local node

    localdir=$(resolve_localdir "$1")
    node=${SLURM_NODEID:-0}

    local raw_ps
    raw_ps=$(ps -u "$USER" -o rss=,comm= 2>/dev/null || true)

    local total_ram
    total_ram=$(echo "$raw_ps" | awk '{sum+=$1} END{if(sum>1024) printf "%dM", sum/1024; else printf "%dK", sum}')

    local top_6
    top_6=$(echo "$raw_ps" | awk '{m[$2]+=$1} END{for(c in m) printf "%d %s\n", m[c], c}' | sort -rn | head -n 6 | awk '{if($1>1024) printf "%s=%dM ",$2,$1/1024; else printf "%s=%dK ",$2,$1}')

    printf "[%s-node %s] Cache: %sK | RAM: %s | Top: %s\n" \
        "$(date +%H:%M:%S)" \
        "$node" \
        "$(du -sk "$localdir" 2>/dev/null | cut -f1)" \
        "$total_ram" \
        "$top_6"
}

# ── JIT cache operations ──────────────────────────────────────────────────

# Restore the JIT compilation cache from shared storage to local tmpfs.
# Usage: restore_cache "$job"
restore_cache() {
    local job="$1"
    local cachetar
    local localdir

    cachetar=$(resolve_cachetar "$job")
    localdir=$(resolve_localdir "$job")

    if [ -f "$cachetar" ]; then
        echo "[cache] restoring JIT cache from shared storage..."
        tar xzf "$cachetar" --no-same-permissions -C "$localdir" 2>/dev/null && \
            echo "[cache] cache restored" || \
            echo "[cache] cache corrupt — recompiling"
    fi
}

# Save the JIT compilation cache from local tmpfs to shared storage.
# Only runs on the head node (SLURM_NODEID == 0).
# Usage: save_cache "$job"
save_cache() {
    local job="$1"
    local cachetar
    local localdir

    cachetar=$(resolve_cachetar "$job")
    localdir=$(resolve_localdir "$job")

    if (( SLURM_NODEID == 0 )); then
        echo "[cache] archiving JIT cache to shared storage..."

        chmod -R g+rwX "$localdir" 2>/dev/null || true

        tar czf "${cachetar}.tmp" \
            --owner=0 --group=0 \
            --mode='g+rwX,o-rwx' \
            -C "$localdir" . 2>/dev/null && \
            mv "$cachetar.tmp" "$cachetar" && \
            chmod 664 "$cachetar" && \
            echo "[cache] saved: $(du -sh "$cachetar" | cut -f1)" || \
            echo "[cache] failed to save JIT cache"
    fi
}
