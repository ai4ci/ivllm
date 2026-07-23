#!/bin/bash
# shellcheck disable=SC2155,SC2153
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
#   $IVLLM_PROJECTDIR    — Root of the shared project space
#
# Optional environment variables:
#   IVLLM_TIME_FMT         — Date format for log timestamp matching (default: +%Y-%m-%d %H:%M)
#   IVLLM_CHECK_INTERVAL_SECS   — Monitor polling interval (default: 10)
#   COMPUTE_HOSTNAME      — Hostname of the compute node (default: $(hostname))

# ── Configurable defaults ───────────────────────────────────────────────────

# Marker to prevent
# Include some form of the following 2 lines to prevent the remainder of the script from being executed again. You can include lines before this line if you still want to execute something every time this is called.
[[ -v IVLLM_UTILS ]] && return
export IVLLM_UTILS=

if [[ -z ${IVLLM_PROJECTDIR:-} ]]; then
    echo "CRITICAL ERROR: \$IVLLM_PROJECTDIR is undefined"
    exit 1
fi

export IVLLM_TIME_FMT="${IVLLM_TIME_FMT:-+%Y-%m-%d %H:%M}"
export IVLLM_CHECK_INTERVAL_SECS="${IVLLM_CHECK_INTERVAL_SECS:-10}"
export IVLLM_TARGET_ENDPOINTS=(
    "/v1/models"
    "/v1/chat"
    "/v1/chat/completions"
    "/v1/responses"
    "/v1/completions"
    "/v1/messages"
)

# ── Path helpers ───────────────────────────────────────────────────────────

# Resolve the per-node local working directory (RAM-backed tmpfs).
# Creates the directory if it doesn't exist. This is per node per user job.
# exports $LOCALDIR, returns node local dir
# Usage: local localdir=$(resolve_localdir "$job")
resolve_localdir() {
    # Resolve the per-node local working directory (RAM-backed tmpfs).
    # Creates the directory if it doesn't exist.
    # Usage: local dir=$(resolve_localdir "$job")
    local job="$1"
    local id=$(id -u)

    unset LOCALDIR
    export LOCALDIR="/local/user/$id"
    mkdir -p "$LOCALDIR"
    chmod 700 "$LOCALDIR"

    local node_local
    node_local="$LOCALDIR/$(hostname -s)/$job"
    mkdir -p "$node_local"
    chmod 700 "$node_local"
    echo "$node_local"
}

# Create shared model directories (HF cache + venv) and export HF_HOME.
# Creates $IVLLM_PROJECTDIR/model/hf and $IVLLM_PROJECTDIR/model/venv if they don't exist.
# Sets $HF_HOME to point at the HuggingFace cache directory.
# Returns: path to the model directory via stdout.
# Usage: local modeldir=$(resolve_model_dir)
resolve_model_dir() {
    # Create shared model directories (HF cache + venv) and export HF_HOME.
    # Sets $HF_HOME to point at the HuggingFace cache directory.
    # Returns: path to the model directory via stdout.
    mkdir -p "$IVLLM_PROJECTDIR/model/hf"
    export HF_HOME="$IVLLM_PROJECTDIR/model/hf"
    mkdir -p "$IVLLM_PROJECTDIR/model/venv"
    chmod -R g+rw "$IVLLM_PROJECTDIR/model"
    echo "$IVLLM_PROJECTDIR/model"
}

# Create the NVHPC SDK base directory with group-write permissions.
# Creates $IVLLM_PROJECTDIR/engine/nvhpc if it doesn't exist.
# Returns: path to the NVHPC directory via stdout.
# Usage: local dir=$(resolve_nvhpc_dir)
resolve_nvhpc_dir() {
    # Create the NVHPC SDK base directory with group-write permissions.
    # Returns: path to the NVHPC directory via stdout.
    mkdir -p "$IVLLM_PROJECTDIR/engine/nvhpc"
    chmod -R g+rw "$IVLLM_PROJECTDIR/engine/nvhpc"
    echo "$IVLLM_PROJECTDIR/engine/nvhpc"
}

# Resolve the NVHPC root directory with a version check (26.3).
# Exits with status 1 and prints an error if the expected NVHPC version is not found.
# Calls resolve_nvhpc_dir() to determine the base path.
# Returns: path to the NVHPC versioned directory via stdout, or exit 1 on failure.
# Usage: local root=$(resolve_nvhpc_root)
resolve_nvhpc_root() {
    # Resolve the NVHPC root directory with a version check (26.3).
    # Calls resolve_nvhpc_dir() to determine the base path.
    # Returns: path to the NVHPC versioned directory via stdout, or exit 1 on failure.
    local nvhpcDir=$(resolve_nvhpc_dir)
    if [[ ! -d "$nvhpcDir/Linux_aarch64/26.3" ]]; then
      echo "NVHPC SDK version 26.3 is not installed. please run ivllm setup." >&2
      return 1
    fi
    echo "$nvhpcDir/Linux_aarch64/26.3"
}

# Create the vLLM virtual environment base directory with group-write permissions.
# Creates $IVLLM_PROJECTDIR/engine/vllm if it doesn't exist.
# Returns: path to the vLLM directory via stdout.
# Usage: local dir=$(resolve_vllm_dir)
resolve_vllm_dir() {
    # Create the vLLM virtual environment base directory with group-write permissions.
    # Returns: path to the vLLM directory via stdout.
    mkdir -p "$IVLLM_PROJECTDIR/engine/vllm"
    chmod -R g+rw "$IVLLM_PROJECTDIR/engine/vllm"
    echo "$IVLLM_PROJECTDIR/engine/vllm"
}

# Create and return the versioned vLLM install directory.
# Calls resolve_vllm_dir() to get the base path, then creates $base/$version.
# Args: $1 — vLLM version string (e.g. "0.19.1"); default is empty string.
# Returns: path to the versioned vLLM directory via stdout.
# Usage: local dir=$(resolve_vllm_version_dir "0.19.1")
resolve_vllm_version_dir() {
    # Create and return the versioned vLLM install directory.
    # Calls resolve_vllm_dir() to get the base path, then creates $base/$version.
    local version="${1:-}"
    local vllm_dir=$(resolve_vllm_dir)
    mkdir -p "$vllm_dir/$version"
    chmod -R g+rw "$vllm_dir/$version"
    echo "$vllm_dir/$version"
}

# Create the shared job root directory with group-write permissions.
# Creates $IVLLM_PROJECTDIR/engine/jobs if it doesn't exist.
# Returns: path to the job root directory via stdout.
# Usage: local dir=$(resolve_job_root_dir)
resolve_job_root_dir() {
    # Create the shared job root directory with group-write permissions.
    # Returns: path to the job root directory via stdout.
    mkdir -p "$IVLLM_PROJECTDIR/engine/jobs"
    chmod -R g+rw "$IVLLM_PROJECTDIR/engine/jobs"
    echo "$IVLLM_PROJECTDIR/engine/jobs"
}

# Resolve a path within a job's directory, creating the directory if needed.
# Args: $1 — job name (required); $2 — optional file/dir name within job dir.
# Returns: path via stdout — $root/$job if no 2nd arg, or $root/$job/$2 otherwise.
# Does not check whether the returned path exists.
# Usage: local path=$(resolve_job_dir "$job" "filename")
resolve_job_dir() {
    # Resolve a path within a job's directory, creating the directory if needed.
    # Returns: path via stdout — $root/$job if no 2nd arg, or $root/$job/$2 otherwise.
    local job="$1"
    local root=$(resolve_job_root_dir)
    local out
    if [[ -z "${2:-}" ]]; then
        out="$root/$job"
    else
        out="$root/$job/$2"
    fi
    mkdir -p "$root/$job"
    chmod -R g+rw "$root/$job"
    echo "$out"
}

# Resolve the path to a per-job JIT cache tarball (~/.cache/ivllm/<job>/jit-cache.tar.gz).
# Caches are user-specific because cache files contain hard-coded paths that cause
# permission issues if shared between users.
# Args: $1 — job name.
# Creates the parent cache directory if it doesn't exist.
# Returns: path to the cache tarball via stdout. Does not check if the file exists.
# Usage: local cache=$(resolve_job_jit_cache "$job")
resolve_job_jit_cache() {
    # Resolve the path to a per-job JIT cache tarball (~/.cache/ivllm/<job>/jit-cache.tar.gz).
    # Caches are user-specific because cache files contain hard-coded paths.
    # Returns: path to the cache tarball via stdout.
    mkdir -p "$HOME/.cache/ivllm/$1/"
    echo "$HOME/.cache/ivllm/$1/jit-cache.tar.gz"
}

# Resolve the path to a job's lockfile (status.json) under the job directory.
# Args: $1 — job name (supports glob patterns like "*").
# Returns: path to status.json via stdout. Does not check if the file exists.
# Usage: local status=$(resolve_job_status "$job")
resolve_job_status() {
    # Resolve the path to a job's lockfile (status.json) under the job directory.
    # Returns: path to status.json via stdout.
    resolve_job_dir "$1" "status.json"
}

# Resolve the path to a per-node vLLM log file (vllm.<nodeid>.log).
# Args: $1 — job name.
# Uses $SLURM_NODEID (default 0) for the log filename; does not take a node parameter.
# Returns: path to the log file via stdout. Does not check if the file exists.
# Usage: local log=$(resolve_job_log "$job")
resolve_job_log() {
    # Resolve the path to a per-node vLLM log file (vllm.<nodeid>.log).
    # Returns: path to the log file via stdout.
    local node="${SLURM_NODEID:-0}"
    resolve_job_dir "$1" "vllm.$node.log"
}

# Resolve the path to a job's vllm.yaml config file.
# Args: $1 — job name.
# Returns: path to vllm.yaml via stdout. Does not check if the file exists.
# Usage: local config=$(resolve_job_config "$job")
resolve_job_config() {
    # Resolve the path to a job's vllm.yaml config file.
    # Returns: path to vllm.yaml via stdout.
    resolve_job_dir "$1" "vllm.yaml"
}

# Strip non-vllm keys from a job's vllm.yaml config and write a clean copy.
# Args: $1 — job name (or path to config file).
# Strips top-level keys: env, nnodes, min-vllm-version, ivllm, idle-timeout, metadata.
# Output: writes $config.clean (e.g. vllm.yaml.clean).
# Calls resolve_job_config() internally to find the config file.
# Returns: path to the cleaned config file via stdout.
# Usage: resolve_stripped_job_config "$job"
resolve_stripped_job_config() {
    # Strip non-vllm keys from a job's vllm.yaml config and write a clean copy.
    local file=$(resolve_job_config "$1")
    local output_file="$file.clean"
    # v3-compatible: chain single-path deletes (v3's `delete`/`d` subcommand
    # takes exactly one path per invocation, not a v4-style filter pipeline)
    yq d "$file" env \
        | yq d - nnodes \
        | yq d - min-vllm-version \
        | yq d - ivllm \
        | yq d - idle-timeout \
        | yq d - metadata > "$output_file"
    echo "$output_file"
}

# Read a field from a job's lockfile using jq.
# must include the leading .
# Usage: local value=$(get_job_status_setting "$job" ".fieldName")
# throws error if the lockfile is not there.
# returns empty value is the lockfile is there but the value is missing.
get_job_status_setting() {
    # Read a field from the lockfile (status.json) using jq.
    # Args: $1 — job name; $2 — jq filter (must include leading dot, e.g. ".status").
    # Exits with code 1 if the lockfile does not exist.
    # Returns empty string if the lockfile exists but the field is missing.
    # Usage: local val=$(get_job_status_setting "$job" ".status")
    local lockfile
    lockfile=$(resolve_job_status "$1")
    if [[ ! -f $lockfile ]]; then
        echo "ERROR: no status file found for job $1" >&2
        exit 1
    fi
    jq -r "$2" "$lockfile" 2>/dev/null || echo ""
}

# Return the lesser of a user-specified time string and 08:00:00 (8 hours).
# Args: $1 — time string in HH:MM:SS format.
# Useful for capping SLURM job time to a maximum of 8 hours.
# Returns: the capped time string via stdout.
# Usage: local max=$(get_max_job_time "$time_str")
get_max_job_time() {
    # Return the lesser of a user-specified time string and 08:00:00 (8 hours).
    local user_time="$1"
    local max_time="08:00:00"
    # Convert max_time to total seconds (HH*3600 + MM*60 + SS)
    local max_secs
    max_secs=$(echo "$max_time" | awk -F: '{print ($1 * 3600) + ($2 * 60) + $3}')

    # Convert user_time to total seconds
    local user_secs
    user_secs=$(echo "$user_time" | awk -F: '{print ($1 * 3600) + ($2 * 60) + $3}')

    # Compare seconds and echo back the correct time string
    if [ "$user_secs" -gt "$max_secs" ]; then
        echo "$max_time"
    else
        echo "$user_time"
    fi
}

# Read a field from a job's vllm.yaml config using yq (v3 syntax).
# Args: $1 — job name; $2 — key path with leading dot (e.g. ".model").
# Strips the leading dot before passing to yq (v3 syntax has no leading dot).
# Exits with code 1 if the config file does not exist.
# Returns empty string if the config exists but the key is missing.
# Usage: local val=$(get_job_config_setting "$job" ".model")
get_job_config_setting() {
    local file=$(resolve_job_config "$1")
    if [[ ! -f $file ]]; then
        echo "ERROR: no configuration file found for job $1" >&2
        exit 1
    fi
    # yq v3's path syntax does NOT use a leading '.' (unlike jq or yq v4);
    # all callers pass the path as `.field` (because jq *does* require it, and
    # the codebase originally mixed `get_job_config_setting` with
    # `get_job_status_setting` call-sites that both use the same dotted
    # convention). Strip the leading '.' before handing to yq.
    local expr="${2#.}"
    yq r "$file" "$expr" 2>/dev/null || echo ""
}

# Extract the top-level 'env:' block from vllm.yaml as bash export lines.
# Args: $1 — job name.
# Uses yq v3 to read key-value pairs and converts them to 'export KEY="VALUE"' lines.
# Returns: export lines via stdout. Returns nothing (exit 0) if config file is missing.
# Usage: eval "$(get_job_config_exports "$job")"
get_job_config_exports() {
    local file=$(resolve_job_config "$1")
    # For a config without an env: block, emit nothing.
    if [[ ! -f "$file" ]]; then
        return 0
    fi
    # v3-compatible: read path+value pairs and build export lines in bash
    # (v3 has no jq-style filter pipeline; v4's `( .env // {} ) | to_entries | ...`
    # is not supported by the installed yq 3.4.1)
    yq r -p pv "$file" 'env.*' 2>/dev/null | sed -E 's/^env\.([^:]+): (.*)$/export \1="\2"/'
}


# Set VLLM and Triton JIT cache environment variables under the node-local directory.
# Calls resolve_localdir() to determine the base path. Sets 7 cache dir variables:
#   VLLM_CACHE_ROOT, EP_JIT_CACHE_DIR, DG_JIT_CACHE_DIR, TRITON_CACHE_DIR,
#   FLASHINFER_JIT_CACHE_DIR, VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR, TORCHINDUCTOR_CACHE_DIR.
# Called by vllm-env.sh during job startup.
# No arguments — determines paths from resolve_localdir().
# Usage: set_jit_caches
set_jit_caches() {
    # Set VLLM and Triton JIT cache environment variables under the node-local directory.
    local localdir=$(resolve_localdir)
    export VLLM_CACHE_ROOT="$localdir/vllm"
    export EP_JIT_CACHE_DIR="$localdir/deep_ep_cache"
    export DG_JIT_CACHE_DIR="$localdir/deep_gemm_cache"
    export TRITON_CACHE_DIR="$localdir/triton"
    export FLASHINFER_JIT_CACHE_DIR="$localdir/flashinfer"
    export VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR="$localdir/flashinfer_auto"
    export TORCHINDUCTOR_CACHE_DIR="$localdir/torchinductor"
}

# ── Lockfile state machine ─────────────────────────────────────────────────

# Create a lockfile with pending status on the login node before sbatch.
# Args: $1 — job name; $2 — model identifier; $3 — idle timeout (default: 30).
# Generates a random high port (49152-65535) for the vLLM server internally.
# Removes existing lockfile if job is in failed/stopped state (restart logic).
# Exits with code 1 if the job is already active.
# Uses set -C (noclobber) for atomic file creation.
# Usage: create_status_pending "$job" "$model" "$idle_timeout"
create_status_pending() {
    local job="$1"
    local model="$2"
    local idle_timeout="${3:-30}"
    local lockfile
    local server_port

    lockfile=$(resolve_job_status "$job")
    mkdir -p "$(resolve_job_dir "$job")"

    # Generate random high port for the vLLM server
    server_port=$(shuf -i 49152-65535 -n 1)

    echo "[startup] creating lockfile for job $job (port=$server_port)" >&2

    if [[ -f $lockfile ]]; then
        if is_status "$job" "failed"; then
            echo "[startup] restarting failed job $job"
            rm -f "$lockfile"
        elif is_status "$job" "stopped"; then
            echo "[startup] restarting stopped job $job"
            rm -f "$lockfile"
        else
            local status=$(get_job_status_setting "$job" ".status")
            echo "[startup] WARNING: job $job is already active with status: $status" >&2
            return 1
        fi
    fi

    # Atomic create with noclobber
    (
        set -C
        jq -n \
            --arg job_name "$job" \
            --arg model "$model" \
            --argjson server_port "$server_port" \
            --argjson idle_timeout "$idle_timeout" \
            --arg req_time "$(date -Iseconds)" \
            --arg user "$(whoami)" \
            '{status: "pending", jobName: $job_name, model: $model, serverPort: $server_port, requestedTime: $req_time, idleTimeout: $idle_timeout, user: $user}' \
            > "$lockfile"
    ) 2>/dev/null || {
        echo "[startup] ERROR: lockfile already exists for job $job" >&2
        return 1
    }

    echo "$server_port"
}

# Update lockfile with SLURM job ID after sbatch submits.
# Args: $1 — job name; $2 — SLURM job ID string.
# Runs on the login node after job submission. Uses jq to write the slurmJobId field.
# BUG: current code uses `-z` check (only writes when ID is empty) — should be `-n`.
# Usage: update_status_slurm_id "$job" "$slurm_id"
update_status_slurm_id() {
    local job="$1"
    local slurm_job_id="${2:-}"
    local lockfile

    lockfile=$(resolve_job_status "$job")

    if [[ -z "$slurm_job_id" ]]; then
        jq --arg slurm_job_id "$slurm_job_id" '.slurmJobId = $slurm_job_id' "$lockfile" > "$lockfile.tmp" && mv "$lockfile.tmp" "$lockfile"
    fi

}

# Update lockfile with SLURM allocation details on the head compute node.
# Args: $1 — job name; $2 — vLLM process PID.
# Sets slurmJobId, computeHostname, vllmPid, start_time, stop_time.
# Only runs on SLURM_NODEID==0 (head node). Reads SLURM_JOB_ID, COMPUTE_HOSTNAME from env.
# Usage: update_status_initialise "$job" "$vllm_pid"
update_status_initialise() {
    local job="$1"
    local vllm_pid="$2"
    local lockfile
    local hostname="${COMPUTE_HOSTNAME:-$(hostname)}"

    lockfile=$(resolve_job_status "$job")

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



# Mark job as running when vLLM health check passes.
# Args: $1 — job name.
# Run on head compute node (SLURM_NODEID==0). Sets .status to "running" via jq.
# Usage: update_status_running "$job"
update_status_running() {
    local job="$1"
    local lockfile

    lockfile=$(resolve_job_status "$job")

    if (( SLURM_NODEID == 0 )); then
        echo "[startup] job $job is running."
        jq '.status = "running"' "$lockfile" > "$lockfile.tmp" && mv "$lockfile.tmp" "$lockfile"
    fi
}

# Mark job as cleanly stopped. Used by tidy_up() exit trap for user cancel or idle timeout.
# Args: $1 — job name.
# Run on head node (SLURM_NODEID==0). Sets .status="stopped", .stopTime, .exitCode="0".
# Usage: update_status_stopped "$job"
update_status_stopped() {
    # Transition lockfile: stopped → stopped.
    # Sets .status="stopped", .stopTime, .exitCode="0" via jq.
    # Run on head node (SLURM_NODEID==0).
    # Usage: update_status_stopped "$job"
    local job="$1"
    local lockfile

    lockfile=$(resolve_job_status "$job")

    if (( SLURM_NODEID == 0 )); then
        echo "[shutdown] clean shutdown for job $job."
        jq \
            --arg stop_time "$(date -Iseconds)" \
            '.status = "stopped" | .stopTime = $stop_time | .exitCode = "0"' \
            "$lockfile" > "$lockfile.tmp" && mv "$lockfile.tmp" "$lockfile"
    fi
}

# Mark job as failed. Used by exit trap for startup failures and crashes.
# Args: $1 — job name; $2 — failure reason string; $3 — numeric exit code.
# Run on head node (SLURM_NODEID==0). Sets .status="failed", .reason, .stopTime, .exitCode.
# Usage: update_status_failed "$job" "reason" "$exit_code"
update_status_failed() {
    # Transition lockfile: → failed.
    # Usage: update_status_failed "$job" "$reason" "$exit_code"
    local job="$1"
    local reason="$2"
    local exit_code="$3"
    local lockfile

    lockfile=$(resolve_job_status "$job")

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
    # Write "cancel" to the lockfile to request graceful shutdown.
    # Usage: request_cancel "$job"
    local job="$1"
    local lockfile

    lockfile=$(resolve_job_status "$job")

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
    # Set the reason field in the lockfile.
    # Usage: update_reason "$job" "$reason_text"
    local job="${1?must supply job name}"
    local reason="${2?must supply reason}"
    local lockfile

    lockfile=$(resolve_job_status "$job")

    if (( SLURM_NODEID == 0 )); then
        echo "[shutdown] reason for job $job: $reason."
        jq --arg reason "$reason" '.reason = $reason' "$lockfile" > "$lockfile.tmp" && mv "$lockfile.tmp" "$lockfile"
    fi
}

# Check if a job has a specific status. Returns 0 if match, 1 otherwise.
# Usage: if is_status "$job" "running"; then ...
is_status() {
    # Check if lockfile status matches expected value.
    # Usage: is_status "$job" "running" → returns 0 if true
    local lockfile
    lockfile=$(resolve_job_status "${1}")
    [ ! -f "$lockfile" ] && return 1
    jq -e --arg test "${2?must supply status}" 'has("status") and .status == $test' "$lockfile" > /dev/null 2>&1
}

is_cancellable() {
    # Check if job is in a cancellable state.
    # Usage: is_cancellable "$job" → returns 0 if cancellable
    squeue -j "${1?must supply slurm id}" -u "$(whoami)" -h -o "%i" | grep -q .
}

# ── Shutdown and cleanup ───────────────────────────────────────────────────

# Graceful shutdown exit trap. Handles all exit codes:
#   200 = SIGUSR1 (SLURM timeout)
#   201 = SIGUSR2 (user cancel or idle timeout)
#   0   = normal exit
#   other = crash (check status to distinguish startup vs runtime)
# Usage: trap 'tidy_up "$job" $?' EXIT
tidy_up() {
    # Exit trap handler: kill vLLM, update lockfile, scancel job.
    # Called on EXIT, SIGUSR1 (SLURM timeout), SIGUSR2 (cancel/idle).
    # Usage: called automatically by setup_traps EXIT handler
    local job="$1"
    local exit_code="$2"
    local pid
    local slurm_job_id

    pid=$(get_job_status_setting "$job" ".vllmPid")
    slurm_job_id=$(get_job_status_setting "$job" ".slurmJobId")

    echo "[shutdown] shutting down job $job (vllm: ${pid:-unknown}, slurm: ${slurm_job_id:-unknown}, exit: $exit_code)"

    # Kill vLLM process if still alive
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
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
            update_status_stopped "$job"
            ;;
        201)
            # SIGUSR2 — user cancel or idle timeout
            echo "[shutdown] received SIGUSR2: user cancel or idle timeout"
            update_status_stopped "$job"
            ;;
        0)
            echo "[shutdown] vLLM terminated normally"
            ;;
        *)
            if is_status "$job" "initialising"; then
                echo "[shutdown] exit code $exit_code: vLLM failed to start"
                update_status_failed "$job" "failed to start" "$exit_code"
            else
                echo "[shutdown] exit code $exit_code: vLLM crashed during inference"
                update_status_failed "$job" "crashed during inference" "$exit_code"
            fi
            ;;
    esac

    # Cancel SLURM job
    if [ "$slurm_job_id" != "null" ] && [ -n "$slurm_job_id" ]; then
        echo "[shutdown] cancelling SLURM job $slurm_job_id"
        scancel "$slurm_job_id" 2>/dev/null || true
    fi

    return 0
}

# Set up exit traps for the monitor triad.
# Usage: setup_traps "$job"
setup_traps() {
    # Register EXIT/SIGUSR1/SIGUSR2 traps for graceful shutdown.
    # Usage: setup_traps "$job" "$vllm_pid" "$slurm_id"
    local job="$1"
    trap 'tidy_up "'"$job"'" 200' SIGUSR1   # SLURM timeout
    trap 'tidy_up "'"$job"'" 201' SIGUSR2   # user cancel or idle timeout
    trap 'tidy_up "'"$job"'" $?' ERR
    trap 'tidy_up "'"$job"'" $?' EXIT
}

# Clear the per-node local working directory.
# Usage: clear_localdir "$job"
clear_localdir() {
    # Remove the local working directory and contents.
    # Usage: clear_localdir "$localdir"
    local localdir
    localdir=$(resolve_localdir "$1")
    [ ! -d "$localdir" ] && exit 1
    echo "[cache] cleaning working directory: $localdir"
    if [ -d "$localdir" ]; then
        rm -rf "$localdir"
    else
        mkdir -p "$localdir"
    fi
}

# ── Monitor: startup (foreground, head node only) ─────────────────────────

# Monitor that blocks until vLLM responds to /health, runs a warmup request,
# saves the JIT cache, and transitions status to "running".
# Usage: monitor_startup "$job" "$vllm_parent_pid"
monitor_startup() {
    # Poll /health endpoint until vLLM is running.
    # Sends warmup request, saves JIT cache, transitions to running.
    # Usage: monitor_startup "$job" "$port" "$model"
    local job="$1"
    local vllm_parent="$2"
    local server_port
    local model

    server_port=$(get_job_status_setting "$job" ".serverPort")
    model=$(get_job_status_setting "$job" ".model")

    # Only runs on head node
    if (( SLURM_NODEID != 0 )); then
        return 0
    fi

    while true; do
        if is_status "$job" "pending"; then
            echo "[startup] waiting for job $job to initialise"
            sleep "$IVLLM_CHECK_INTERVAL_SECS"
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
                    echo "[startup] Startup complete: vLLM is running."
                    break
                else
                    echo "[startup] ERROR: warmup failed after $max_retries attempts"
                    update_reason "$job" "vLLM warmup failed"
                    echo "[startup] Startup complete: vLLM failed warmup."
                    kill -s SIGUSR2 "$vllm_parent" 2>/dev/null

                    return 1
                fi
            else
                echo "[startup] job $job waiting for vLLM /health"
                report_memory "$job"
                sleep "$IVLLM_CHECK_INTERVAL_SECS"
                continue
            fi
        else
            local status
            status=$(get_job_status_setting "$job" ".status")
            echo "[startup] ERROR: job $job is in unexpected state $status"
            echo "[startup] Startup complete: vLLM failed to start."
            return 1
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
    # Background monitor loop on head node.
    # Checks cancel flag, vLLM liveness, idle timeout, lockfile presence.
    # Usage: monitor_head "$job" "$port" "$timeout"
    local job="$1"
    local vllm_parent="$2"
    local lockfile
    local log
    local idle_timeout

    lockfile=$(resolve_job_status "$job")
    log=$(resolve_job_log "$job")
    idle_timeout=$(get_job_status_setting "$job" ".idleTimeout")

    if [ ! -f "$lockfile" ]; then
        echo "[head] FATAL: lockfile $lockfile missing on startup"
        kill -s SIGTERM "$vllm_parent" 2>/dev/null
        return 1
    fi

    echo "[head] starting monitor (idle_timeout=$idle_timeout)..."

    while true; do
        # Lockfile deleted
        if [ ! -f "$lockfile" ]; then
            echo "[head] lockfile $lockfile has been deleted — shutting down"
            kill -s SIGTERM "$vllm_parent" 2>/dev/null
            return 1
        fi

        # Terminal states — exit loop
        if is_status "$job" "failed" || is_status "$job" "stopped"; then
            break
        fi

        # Still pending — wait for SLURM allocation
        if is_status "$job" "pending"; then
            sleep "$IVLLM_CHECK_INTERVAL_SECS"
            continue
        fi

        local vllm_pid
        vllm_pid=$(get_job_status_setting "$job" ".vllmPid")

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
            sleep "$IVLLM_CHECK_INTERVAL_SECS"
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
                time_patterns+=("-e" "$(date -d "$i minutes ago" "$IVLLM_TIME_FMT")")
            done

            local endpoint_patterns=()
            for endpoint in "${IVLLM_TARGET_ENDPOINTS[@]}"; do
                endpoint_patterns+=("-e" "$endpoint")
            done

            # Check for recent API requests (not /health)
            if tail -n 5000 "$log" 2>/dev/null | grep -F "${time_patterns[@]}" | grep -q -F "${endpoint_patterns[@]}"; then
                sleep "$IVLLM_CHECK_INTERVAL_SECS"
                continue
            fi

            echo "[head] no API requests for $idle_timeout minutes — shutting down"
            update_reason "$job" "idle timeout"
            break
        fi

        sleep "$IVLLM_CHECK_INTERVAL_SECS"
    done

    # Signal shutdown to parent
    kill -s SIGUSR2 "$vllm_parent" 2>/dev/null

    # Wait for parent to exit
    while kill -0 "$vllm_parent" 2>/dev/null; do
        sleep 2
    done

    clear_localdir "$job"
    echo "[head] monitor shutting down for job $job."
    return 0
}

# ── Monitor: worker node (background) ─────────────────────────────────────

# Background monitor for worker nodes in multi-node jobs. Watches the lockfile
# and shuts down the local vLLM process if the job is no longer in a valid
# state.
# Usage: monitor_worker "$job" "$vllm_worker_pid" &
monitor_worker() {
    # Background monitor on worker nodes.
    # Watches lockfile; shuts down if job is no longer running.
    # Usage: monitor_worker "$job" "$vllm_pid"
    local job="$1"
    local vllm_worker="$2"
    local lockfile
    local node

    lockfile=$(resolve_job_status "$job")
    node=${SLURM_NODEID:-0}

    # This monitor is for worker nodes only
    if [ "$node" -eq 0 ]; then
        echo "[worker] ERROR: worker monitor started on head node"
        kill -s SIGTERM "$vllm_worker" 2>/dev/null
        return 1
    fi

    if [ ! -f "$lockfile" ]; then
        echo "[worker $node] FATAL: lockfile $lockfile missing on startup"
        kill -s SIGTERM "$vllm_worker" 2>/dev/null
        return 1
    fi

    echo "[worker $node] starting lockfile monitor..."

    while true; do
        if [ ! -f "$lockfile" ]; then
            echo "[worker $node] lockfile deleted — shutting down"
            kill -s SIGTERM "$vllm_worker" 2>/dev/null
            exit 1
        fi

        if is_status "$job" "pending"; then
            sleep "$IVLLM_CHECK_INTERVAL_SECS"
            continue
        fi

        if is_status "$job" "initialising"; then
            report_memory "$job"
            sleep "$IVLLM_CHECK_INTERVAL_SECS"
            continue
        fi

        if ! is_status "$job" "running"; then
            echo "[worker $node] job is not running (status=$(get_job_status_setting "$job" ".status")) — shutting down"
            break
        fi

        sleep "$IVLLM_CHECK_INTERVAL_SECS"
    done

    kill -s SIGTERM "$vllm_worker" 2>/dev/null

    while kill -0 "$vllm_worker" 2>/dev/null; do
        sleep 2
    done

    clear_localdir "$job"
    echo "[worker $node] monitor shutting down."
    return 0
}

# ── Resource monitoring ───────────────────────────────────────────────────

report_setup() {
    # Report setup progress and exit code to lockfile.
    # Usage: report_setup "$job" "$exit_code"
echo "=== Python & Library Extension Environment ==="
python -c "
import os, sys, torch, deep_gemm, deep_ep

print(f'Python Interpreter: {sys.executable}')
print(f'PyTorch Source CUDA: {torch.version.cuda}')
print(f'Device 0 Target Name: {torch.cuda.get_device_name(0)}')
print(f'Device Compute Capability: {torch.cuda.get_device_capability(0)}')

print('\n--- Extension Library Status ---')
# Crash-proof DeepGEMM Check
try:
    import deep_gemm
    print(f'DeepGEMM Package Version: {deep_gemm.__version__}')
except ImportError as e:
    print(f'❌ DeepGEMM Status: NOT AVAILABLE ({e})')

# Crash-proof DeepEP Check
try:
    import deep_ep
    print(f'DeepEP Package Version:  {deep_ep.__version__}')
except ImportError as e:
    print(f'❌ DeepEP Status:  NOT AVAILABLE ({e})')
"
echo "=== Final Environment Variables for vLLM ==="
# Expanded search to capture your critical NVSHMEM, EP, DG, and GLOO runtime flags
env | grep -E "^(VLLM_|RAY_|NCCL_|FI_|NVHPC|CUDA_|LD_CONFIG|CPATH|PATH|SLURM_|TRITON|NVSHMEM_|EP_|DG_|GLOO_)" | sort
echo "============================================"
}


# Report memory and JIT cache usage for the current node.
# Usage: report_memory "$job"
report_memory() {
    # Report memory usage to the log.
    # Usage: report_memory
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
    # Restore JIT cache from shared storage to local tmpfs.
    # Usage: restore_cache "$cache_path" "$target_dir"
    local job="$1"
    local cachetar
    local localdir

    cachetar=$(resolve_job_jit_cache "$job")
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
    # Save JIT cache from local tmpfs to shared storage.
    # Usage: save_cache "$local_dir" "$cache_path"
    local job="$1"
    local cachetar
    local localdir

    cachetar=$(resolve_job_jit_cache "$job")
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

# ── VLLM versioning utils ──────────────────────────────────────────────────

# Helper: Parse version string into components, defaulting missing/invalid parts to 0
_parse_semver() {
    # Internal: parse semver string into MAJOR.MINOR.PATCH.
    # Usage: _parse_semver "0.19.1" → sets _MAJOR _MINOR _PATCH
    local IFS='.'
    local -a parts
    read -r -a parts <<< "$1"
    # Ensure non-integers or empty values become 0
    echo $((parts[0] + 0)) $((parts[1] + 0)) $((parts[2] + 0))
}

# Compare two semantic version strings: a < b
# Returns 0 (true) if a < b, otherwise returns 1 (false)
semver_lt() {
    # Internal: less-than comparison for semver strings.
    # Usage: semver_lt "0.19.0" "0.20.0" → returns 0 if a < b
    read -r a1 a2 a3 <<< "$(_parse_semver "$1")"
    read -r b1 b2 b3 <<< "$(_parse_semver "$2")"

    if (( a1 != b1 )); then return $(( a1 >= b1 )); fi
    if (( a2 != b2 )); then return $(( a2 >= b2 )); fi
    return $(( a3 >= b3 ))
}

# Compare two semantic version strings: a >= b
# Returns 0 (true) if a >= b, otherwise returns 1 (false)
semver_gte() {
    # Internal: greater-or-equal comparison for semver strings.
    # Usage: semver_gte "0.20.0" "0.19.0" → returns 0 if a >= b
    semver_lt "$1" "$2"
    # Invert the boolean return status (0 becomes 1, 1 becomes 0)
    return $(( ! $? ))
}

# Sort an array of semantic version strings in descending order
# Expects versions as separate arguments. Outputs sorted list to stdout.
semver_sort() {
    # Internal: sort semver strings in ascending order.
    # Usage: semver_sort "0.19.0" "0.21.0" "0.20.0"
    printf '%s\n' "$@" | sort -V -r
}

# Sort an array of semantic version strings in ascending order
# Expects versions as separate arguments. Outputs sorted list to stdout.
rev_semver_sort() {
    # Internal: sort semver strings in descending order.
    # Usage: rev_semver_sort "0.19.0" "0.21.0" "0.20.0"
    printf '%s\n' "$@" | sort -V
}

# Find the LOWEST installed version directory that satisfies a minimum version constraint.
# Expects minimum vllm version to match
select_closest_version() {
    # Find the best installed vLLM version >= minimum.
    # Usage: select_closest_version "0.19.0" → returns matching version
    local install_dir="$(resolve_vllm_dir)"
    local min_version="$1"
    local candidate
    local -a valid_candidates=()

    if [[ ! -d "$install_dir" ]]; then
        return 1
    fi

    # 1. Discover and filter subdirectories
    while IFS= read -r -d '' dir; do
        candidate=$(basename "$dir")

        # Filter: Must be >= min_version
        if semver_gte "$candidate" "$min_version"; then
            valid_candidates+=("$candidate")
        fi
    done < <(find "$install_dir" -mindepth 1 -maxdepth 1 -type d -print0 2>/dev/null)

    if (( ${#valid_candidates[@]} == 0 )); then
        return 0
    fi

    # 2. Sort ASCENDING and pick the first one (the lowest valid version)
    rev_semver_sort "${valid_candidates[@]}" | head -n 1
}
