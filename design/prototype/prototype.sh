#!/bin/bash
# shellcheck disable=SC2155

# export TIME_FMT="+%Y-%m-%d %H:%M:%S"
export VLLM_TIME_FMT="+%m-%d %H:%M"
export CHECK_INTERVAL_SECS=10
export TARGET_ENDPOINTS=(
    "/v1/models"
    "/v1/chat"
    "/v1/chat/completions"
    "/v1/responses"
    "/v1/completions"
    "/v1/messages"
)
export ENGINE_DIR="$PROJECTDIR/engine"

##################
### PATHS     ####
##################

# Resolve the location of the local working directory for a specific node.
# This is a per job per node working directory. We have to assume multiple
# differently named jobs are running on each node
# expects job only
resolve_localdir() {

    local job=$1

    # Detect and handle missing LOCALDIR in interactive srun allocations
    if [ -z "${LOCALDIR:-}" ]; then
        export LOCALDIR="/local/user/$UID"
        mkdir -p "$LOCALDIR"
        chmod 700 "$LOCALDIR"
    fi

    # Isolate node workspaces by true kernel hostname to prevent cross-node
    # collisions on shared interactive loopbacks
    local nodeLocal="$LOCALDIR/$(hostname -s)/$job"
    mkdir -p "$nodeLocal"
    echo "$nodeLocal"
}

# The location that the vllm process is executed in
# resolve_workdir() {
#
# }


# resolve the location of a file in the job directory
# expects job and optionally the file path.
resolve_location() {
    local job=$1
    local file=${2:-}
    local dir="$ENGINE_DIR/jobs/$job"
    echo "$dir/$file"
}

# resolve the location of the jit cache tar for a specific job
# expects job
resolve_cachetar() {
    resolve_location "$1" "jit-cache.tar.gz"
}

# resolve the location of the lockfile for a specific job
# expects job
resolve_lockfile() {
    resolve_location "$1" "status.json"
}

# resolve the location of the log for a specific job
# expects job
resolve_logfile() {
    local node=${SLURM_NODEID:-0}
    resolve_location "$1" "vllm.$node.log"
}

# resolve a setting from the lockfile
# expects job and the setting name
resolve_setting() {
    local lockfile=$(resolve_lockfile "$1")
    jq ".$2" "$lockfile"
}

# Runs on login node. Creates a lockfile.
# Expects job, model, server port, idle_timeout.
create_status_pending() {
    local lockfile=$(resolve_lockfile "$1")
    mkdir -p "$(resolve_location "$1")"
    echo "[startup] creating lockfile for job $1"
    if ((${SLURM_NODEID:-0} == 0)); then
        jq -n \
        --arg job_name "$1" \
        --arg model "$2" \
        --argjson server_port "$3" \
        --argjson idle_timeout "$4" \
        --arg req_time "$(date -Iseconds)" \
        '{status: "pending", jobName: $job_name, model: $model, serverPort: $server_port, requestedTime: $req_time, idleTimeout: $idle_timeout}' \
        > "$lockfile"
    fi
}

# Runs on compute node (head node of multinode).
# updates lockfile with SLURM parameters and PID of master vllm process.
# initialising status implies resources have been assigned and vllm is starting up.
# Expects job, and vllm PID.
update_status_initialise() {
    local lockfile=$(resolve_lockfile "$1")
    local log=$(resolve_logfile "$1")
    touch "$log"

    if ((${SLURM_NODEID:-0} == 0)); then

        echo "[startup] slurm job allocated for job $1: slurm details ($SLURM_JOB_ID)"
        jq \
        --argjson vllm_pid "$2" \
        --argjson slurm_job_id "$SLURM_JOB_ID" \
        --arg compute_hostname "$COMPUTE_HOSTNAME" \
        --arg start_time "$(date -d "@$SLURM_JOB_START_TIME" -Iseconds)" \
        --arg stop_time "$(date -d "@$SLURM_JOB_END_TIME" -Iseconds)" \
        '.status = "initialising" | .slurmJobId = $slurm_job_id | .computeHostname = $compute_hostname | .startTime = $start_time | .stopTime = $stop_time | .vllmPid = $vllm_pid' \
        "$lockfile" > "$lockfile.tmp"

        mv "$lockfile.tmp" "$lockfile"
    fi
}

# Runs on head compute node.
# running status implies vllm has started and is serving requests
# Expects job.
update_status_running() {
    local lockfile=$(resolve_lockfile "$1")

    if ((${SLURM_NODEID:-0} == 0)); then
        echo "[startup] job $1 is running."
        jq \
        '.status = "running"' \
        "$lockfile" > "$lockfile.tmp"

        mv "$lockfile.tmp" "$lockfile"
    fi
}

# Runs on compute node.
# running status implies vllm has started and is serving requests
# Expects job.
request_cancel() {
    local lockfile=$(resolve_lockfile "$1")
    if ((${SLURM_NODEID:-0} == 0)); then
        echo "[shutdown] requesting cancel for job $1."
        jq \
        '.status = "cancel"' \
        "$lockfile" > "$lockfile.tmp"

        mv "$lockfile.tmp" "$lockfile"
    fi
}

# Runs on compute node in a trap.
# failed status implies vllm failed to start.
# Expects a description of the failure reason
# and an error code.
# Expects job. reason for error, and exit code
update_status_unclean_shutdown() {
    local lockfile=$(resolve_lockfile "$1")
    if ((${SLURM_NODEID:-0} == 0)); then
        echo "[shutdown] unclean shutdown for job $1 due to $2 ($3)."
        jq \
        --arg error "$2" \
        --argjson exit_code "$3" \
        --arg stop_time "$(date -Iseconds)" \
        '.status = "failed" | .reason = $error | .stopTime = $stop_time | .exitCode = $exit_code' \
        "$lockfile" > "$lockfile.tmp"

        mv "$lockfile.tmp" "$lockfile"
    fi
}

# Runs on compute node in a trap.
# triggered by end of job, user cancel, idle timeout
# reason must be set separately
# Expects job.
update_status_clean_shutdown() {
    local lockfile=$(resolve_lockfile "$1")
    if ((${SLURM_NODEID:-0} == 0)); then
        echo "[shutdown] clean shutdown for job $1."
        jq \
        --arg stop_time "$(date -Iseconds)" \
        '.status = "stopped" | .stopTime = $stop_time | .exitCode = "0"' \
        "$lockfile" > "$lockfile.tmp"

        mv "$lockfile.tmp" "$lockfile"
    fi
}

# Runs on compute node before clean shutdown.
# expects lockfile and reason for shutdown
# Expects job, and shutdown reason.
update_reason() {
    local lockfile=$(resolve_lockfile "$1")
    if ((${SLURM_NODEID:-0} == 0)); then
        echo "[shutdown] reason for shutdown for job $1: $2."
        jq \
        --arg reason "$2" \
        '.reason = $reason' \
        "$lockfile" > "$lockfile.tmp"

        mv "$lockfile.tmp" "$lockfile"
    fi
}

# checks whether job has a given status
# expects job and status to check
is_status() {
    local lockfile=$(resolve_lockfile "$1")
    jq \
    --arg test "$2" \
    -e 'has("status") and .status == $test' "$lockfile" > /dev/null
}

##################
### SHUTDOWN  ####
##################

# Ensures vllm job is killed and updates reason.
# This runs in a trap in the subshell but can be run completely independently
# Is called from an EXIT trap and the exit code defines behaviour.
# expects job, and exit code, everything else is read from the status.
tidy_up() {
    local job=$1
    local pid=$(resolve_setting "$job" "vllmPid")
    local slurmJobId=$(resolve_setting "$job" "slurmJobId")
    local exit_code=$2

    echo "[shutdown] shutting down job $1 (vllm: $pid, slurm: $slurmJobId, exit: $exit_code)."

    # Force clean shutdown of the actual vLLM background process and any
    # spawned children if still alive:

    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
        echo "[shutdown] killing process $pid shutdown for job $1."
        kill -15 "$pid" 2>/dev/null

        # Optional: Give it 2 seconds to close cleanly, otherwise force kill
        sleep 2
        if kill -0 "$pid" 2>/dev/null; then
            kill -9 "$pid" 2>/dev/null
        fi
    fi

    case "$exit_code" in
        200)
            # SIGUSR1 - slurm timeout - no reason will have been given yet
            echo "[shutdown] received SIGUSR1: slurm timeout"
            update_reason "$job" "SLURM timeout"
            update_status_clean_shutdown "$job"
            ;;
        201)
            # SIGUSR2 - other reasons
            echo "[shutdown] received SIGUSR2: idle shutdown or user cancel"
            # reason for shutdown has already been recorded
            update_status_clean_shutdown "$job"
            ;;
        0)
            echo "[shutdown] vllm parent terminated normally"
            ;;
        *)
            if is_status "$job" "initialising"; then
                echo "[shutdown] exit code $exit_code: vllm error on startup"
                update_status_unclean_shutdown "$job" "failed to start" "$exit_code"
                # recover logs?
            else
                echo "[shutdown] exit code $exit_code: vllm error during inference"
                update_status_unclean_shutdown "$job" "crashed during inference" "$exit_code"
            fi
            ;;
    esac

    echo "[shutdown] scancel slurm job $slurmJobId for job $1."
    scancel "$slurmJobId"

    exit 0
}

# setup exit traps
# expects job_name
setup_traps() {
    trap 'tidy_up "'"$1"'" 200' SIGUSR1 # slurm timeout
    trap 'tidy_up "'"$1"'" 201' SIGUSR2 # idle timeout or user cancel request
    trap 'tidy_up "'"$1"'" $?' ERR
    trap 'tidy_up "'"$1"'" $?' EXIT
}

# ensures localdir is empty and exists
clear_localdir() {
    local localdir=$(resolve_localdir "$1")
    echo "[cache] clean working directory for job $1: $localdir"
    if [[ -d "$localdir" ]]; then
        rm -rf "${localdir:?no localdir defined}/*"
    else
        mkdir -p "$localdir"
    fi
}

##################
### MONITORS  ####
##################

# report the processes and JIT cache memory usage
# expects job
report_memory() {
    local localdir=$(resolve_localdir "$1")
    local node=${SLURM_NODEID:-0}

    # 1. Get the raw process memory footprint data
    local raw_ps=$(ps -u "$USER" -o rss=,comm=)

    # 2. Extract the grand TOTAL of all user processes combined
    local total_ram=$(echo "$raw_ps" | awk '{sum+=$1} END{if(sum>1024) printf "%dM", sum/1024; else printf "%dK", sum}')

    # 3. Extract and format the TOP 6 heaviest processes
    local top_6_stats=$(echo "$raw_ps" | awk '{m[$2]+=$1} END{for(c in m) printf "%d %s\n", m[c], c}' | sort -rn | head -n 6 | awk '{if($1>1024) printf "%s=%dM ",$2,$1/1024; else printf "%s=%dK ",$2,$1}')

    printf "[%s-node %s] Cache: %sK | Total RAM: %s | Top 6: %s\n" \
        "$(date +%H:%M:%S)" \
        "$node" \
        "$(du -sk "$localdir" 2>/dev/null | cut -f1)" \
        "$total_ram" \
        "$top_6_stats"
}

# intended to be started in foreground and will block until the health endpoint
# responds. This is only run on head node.
# expects job and vllm parent pid
monitor_startup() {
    local job=$1
    local vllm_parent=$2
    local serverPort=$(resolve_setting "$job" "serverPort")
    local model=$(resolve_setting "$job" "model")

    if ((${SLURM_NODEID:-0} == 0)); then

        while true; do

            if is_status "$job" "pending"; then
                echo "[startup] waiting for job $job to initialize"
                sleep $CHECK_INTERVAL_SECS
                continue
            fi

            if is_status "$job" "initialising"; then

                if curl -sf "http://localhost:$serverPort/health" > /dev/null 2>&1; then
                    echo "[startup] vllm /health api active: saving compilation cache"

                    save_cache "$job"

                    local max_retries=5
                    local attempt=1
                    local warmup_success=1 # 0 = success, 1 = failure

                    echo "[startup] Sending warmup token to trigger JIT cache compilation..."

                    while (( attempt <= max_retries )); do
                        # Check if the endpoint responds correctly
                        if curl -sf "http://localhost:$serverPort/v1/chat/completions" \
                            -H "Content-Type: application/json" \
                            -d "{
                                \"model\": \"$model\",
                                \"messages\": [{\"role\": \"user\", \"content\": \"Hello.\"}]
                            }" > /dev/null 2>&1; then

                            warmup_success=0
                            break # Exit retry loop immediately on success!
                        else
                            echo "[startup] WARNING: warmup attempt $attempt/$max_retries transiently failed or timed out. Retrying in 10s..."
                            (( attempt++ ))
                            sleep 10
                        fi
                    done

                    # Evaluate ultimate warmup success state
                    if (( warmup_success == 0 )); then
                        echo "[startup] job $job /v1/chat/completions api active: saving jit cache"
                        save_cache "$job"
                        echo "[startup] job $job startup complete."
                        update_status_running "$job"
                        break
                    else
                        # If all 5 attempts fail back-to-back, drop out and let external monitoring catch the failure
                        echo "[startup] ERROR: Warmup permanently failed after $max_retries consecutive transient errors."
                        update_reason "$job" "vllm warmup failed after retries"
                        kill -s SIGUSR2 "$vllm_parent" 2>/dev/null || exit 1
                        exit 1
                    fi

                else
                    echo "[startup] job $job waiting for vllm /health api."
                    report_memory "$job"
                    continue
                fi

            else

                local status=$(resolve_setting "$job" "status")
                echo "[startup] failed to detect vllm start. job $job is in status $status."
                exit 1

            fi

            sleep $CHECK_INTERVAL_SECS
        done

    fi

    echo "[startup] job $job started up normally. startup monitor detached."
}

# intended to be started as a background process on the head node.
# it looks for lockfile presence, checks for "cancel" status in lockfile,
# ensures head vllm process is still present, and that either
# 1) vllm status is "initialising" or 2) vllm status is running & logs contain
# a recent query to a supported endpoint expects job, vllm parent pid
monitor_head() {
    local job=$1
    local lockfile=$(resolve_lockfile "$job")
    local vllm_parent=$2
    local log=$(resolve_logfile "$job")
    local idle_timeout=$(resolve_setting "$job" "idleTimeout")

    if [[ ! -f "$lockfile" ]]; then
        # If the lockfile is missing we panic shutdown vllm.
        echo "[head] status file $lockfile is missing: shutting down head"
        # Cant update missing lockfile! Sigterm skips attempt to write to lockfile.
        kill -s SIGTERM "$vllm_parent"
        exit 1
    fi

    echo "[head] starting idle shutdown monitor..."
    while true; do

        # 1. Check lockfile: if it has gone skip straight to shutdown
        if [[ ! -f "$lockfile" ]]; then

            # If the lockfile is missing we panic shutdown vllm.
            # This happens only if it originally existed then was deleted.
            echo "[head] status file $lockfile has been deleted: shutting down head (node $node)"
            # Cant update missing lockfile! Sigterm skips attempt to write to lockfile.
            kill -s SIGTERM "$vllm_parent"
            exit 1

        else

            if is_status "$job" "pending"; then
                # job status is not yet updated
                # This skips all checks until the node is assigned
                sleep "$CHECK_INTERVAL_SECS"
                continue
            fi

            if is_status "$job" "failed" || is_status "$job" "stopped"; then
                break
            fi

            local vllm_pid=$(resolve_setting "$job" "vllmPid")

            # 0. check vllm process is still active
            if ! kill -0 "$vllm_pid" > /dev/null 2>&1; then
                echo "[head] vllm process ($vllm_pid) has gone away"
                update_reason "$job" "lost contact with vllm process"
                break
            fi

            # 1. check for user requested shutdown via lockfile:
            if is_status "$job" "cancel"; then
                echo "[head] user cancel request detected: shutting down head."
                update_reason "$job" "user cancel"
                break
            fi

            # 2a. process is still initialising:
            if is_status "$job" "initialising"; then
                sleep "$CHECK_INTERVAL_SECS"
                # This skips the active traffic checks:
                continue
            fi

            if is_status "$job" "running"; then
                # 2b. filter server logs for active traffic
                if [[ ! -f "$log" ]]; then

                    # If the log is missing we panic shutdown vllm.
                    echo "[head] log file $log is missing: shutting down head"
                    update_reason "$job" "missing log file"
                    break

                else

                    # Generate regex patterns for the last X minutes to narrow down grep search
                    time_patterns=()
                    for i in $(seq 0 "$idle_timeout"); do
                        time_patterns+=("-e" "$(date -d "$i minutes ago" "$VLLM_TIME_FMT")")
                    done

                    endpoint_patterns=()
                    for endpoint in "${TARGET_ENDPOINTS[@]}"; do
                        endpoint_patterns+=("-e" "$endpoint")
                    done

                    # Check if the target endpoint appears in the logs within the idle window
                    if tail -n 5000 "$log" | grep -F "${time_patterns[@]}" | grep -q -F "${endpoint_patterns[@]}"; then
                        echo "[head] system active: recent api request found."
                        sleep "$CHECK_INTERVAL_SECS"
                        continue
                    fi

                    echo "[head] no api requests seen for $idle_timeout minutes: shutting down head."
                    update_reason "$job" "idle timeout"
                    break

                fi
            fi
        fi
    done

    # If we have got here (via a break in the loop above) then something has
    # caused a condition requiring a shutdown. This is signalled to the vllm
    # parent:
    kill -s SIGUSR2 "$vllm_parent"

    while [[ -n "$vllm_parent" ]] && kill -0 "$vllm_parent" 2>/dev/null; do
        echo "[head] waiting for vllm parent process $vllm_parent to exit for job $1."
        sleep 2
    done

    # Tidy up the caches.
    clear_localdir "$job"

    # 3. Trigger parent slurm job shutdown.
    echo "[head] monitor shutting down for job $1."
    exit 0

}


# A process monitor that keeps track of a lockfile status and kills process if
# lockfile is missing or lockfile status is not pending, initialising, or running.
# This is for worker nodes only.
# expects lockfile, vllm parent pid, and jit cachedir localdir
monitor_worker() {
    local job=$1
    local lockfile=$(resolve_lockfile "$job")
    local vllm_parent=$2
    local log=$(resolve_logfile "$job")
    local node=${SLURM_NODEID:-0}

    # designed for worker nodes. Does not update lockfile. purely tracks lockfile.
    if [[ "$node" -eq 0 ]]; then
        echo "[worker $node] worker monitor started for head node! this is a design error."
        kill -s SIGTERM "$vllm_parent"
        exit 1
    fi

    # Lockfile is missing to begin with
    if [[ ! -f "$lockfile" ]]; then
        # If the lockfile is missing we panic shutdown the worker.
        echo "[worker $node] status file $lockfile is missing: shutting down worker"
        # Cant update missing lockfile!
        kill -s SIGTERM "$vllm_parent"
        exit 1
    fi

    echo "[worker $node] starting lockdown file monitor..."
    while true; do

        # 1. Check lockfile: if it has gone skip straight to shutdown
        if [[ ! -f "$lockfile" ]]; then

            # If the lockfile is missing we panic shutdown the worker.
            # This happens only if it originally existed then was deleted.
            echo "[worker $node] status file $lockfile has been deleted: shutting down worker"
            # Cant update missing lockfile!
            kill -s SIGTERM "$vllm_parent"
            exit 1

        else

            # pending status is unusual. potentially a race condition where the
            # head node has yet to startup.
            if is_status "$job" "pending"; then
                continue
            fi

            # initialising means vllm is starting up.
            # check memory and cache status of the node while starting up
            if is_status "$job" "initialising"; then
                report_memory "$job"
                continue
            fi

            # Of the other statuses only "running" is OK, any other statuses
            # imply worker node needs to shutdown via lockfile:
            if ! is_status "$job" "running" ; then
                echo "[worker $node] head vllm process is not active: shutting down worker"
                break
            fi

            sleep "$CHECK_INTERVAL_SECS"
        fi
    done

    # status in lockfile is not pending, initialising or running. This means
    # the process has stopped (or cancel is requested) and we shutdown the
    # worker:
    kill -s SIGTERM "$vllm_parent"

    while [[ -n "$vllm_parent" ]] && kill -0 "$vllm_parent" 2>/dev/null; do
        echo "[worker $node] waiting for vllm parent process $vllm_parent to exit for job $1."
        sleep 2
    done

    clear_localdir "$job"

    echo "[worker $node] monitor shutting down for job $1."
    # 3. Trigger parent slurm job shutdown.
    exit 0
}

##################
### CACHE     ####
##################

# restore the JIT cache directory
# expects the lockfile location and the jit cache location
# cache file will be sibling to lockfile `./jit-cache.tar.gz`.
# expects job only
restore_cache() {

    # Archive the tmpfs $HOME to shared storage for next cold start.
    local job=$1
    local cachetar=$(resolve_cachetar "$job")
    local localdir=$(resolve_localdir "$job")

    # 3. Try to restore cached JIT compilations
    if [ -f "$cachetar" ]; then
        echo "[cache] restoring JIT cache from shared storage..."
        # --no-same-permissions (or -m) forces tar to map files to the current user's umask
        tar xzf "$cachetar" --no-same-permissions -C "$localdir" 2>/dev/null && \
        echo "[cache] cache restored" || \
        echo "[cache] cache corrupt — recompiling"
    fi

}

# save the JIT cache directory
# expects the lockfile location and the jit cache location
# cache file will be sibling to lockfile `./jit-cache.tar.gz`.
# expects job only
save_cache() {

    # Archive the tmpfs $HOME to shared storage for next cold start.
    local job=$1
    local cachetar=$(resolve_cachetar "$job")
    local localdir=$(resolve_localdir "$job")

    if ((${SLURM_NODEID:-0} == 0)); then

        echo "[cache] archiving JIT cache to shared storage..."

        # 1. Force permissions inside the directory to be group-accessible before archiving
        chmod -R g+rwX "$localdir" 2>/dev/null || true

        # 2. Tar the cache while wiping out individual user metadata
        tar czf "${cachetar}.tmp" \
            --owner=0 \
            --group=0 \
            --mode='g+rwX,o-rwx' \
            -C "$localdir" . 2>/dev/null && \
            mv "$cachetar.tmp" "$cachetar" && \
            chmod 664 "$cachetar" && \
            echo "[cache] saved jit cache as: $(du -sh "$cachetar" | cut -f1)" || \
            echo "[cache] failed to save jit cache"

    fi
}



