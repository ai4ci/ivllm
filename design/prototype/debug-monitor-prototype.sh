#!/bin/bash
# shellcheck disable=SC2155,SC2153
#
# debug-monitor-prototype.sh — PROTOTYPE, not wired into the build.
#
# Implements two things discussed in design/active-issues.md's GLM-5.2 entry
# and design/ivllm-environment.md's "Debugging flags" section:
#
#   1. set_debugging_env() — the not-yet-built IVLLM_DEBUG_LEVEL master-flag
#      function that ivllm-environment.md describes but utils.sh doesn't
#      implement yet. Three layers, matching that doc's Level 3/4 spec:
#        Layer 1 — vLLM's own logger (VLLM_LOGGING_LEVEL)
#        Layer 2 — NCCL / torch.distributed (NCCL_DEBUG(_SUBSYS) + the
#                  TORCH_NCCL_* flight-recorder/desync-debug family)
#        Layer 3 — libfabric / CXI (FI_LOG_LEVEL/_PROV/_SUBSYS)
#      All file-based debug artifacts route into the job's shared debug/
#      directory (resolve_job_dir "$job" "debug"), matching the existing
#      pyspy-dump convention in report_memory().
#
#   2. Stall-triggered flight-recorder dump — monitor_head() and
#      wait_report() modified to cooperate: monitor_head (head node, tails
#      the aggregated log already) detects the
#      "No available shared memory broadcast block found" warning and
#      manually triggers a TORCH_NCCL_DEBUG_INFO_PIPE_FILE dump on every
#      node, rather than waiting for (or racing) PyTorch's own NCCL
#      watchdog. wait_report() (runs on every node) notices the same
#      stall signal via a shared sentinel file and grabs an extra,
#      synchronized report_memory() snapshot the moment it's seen.
#
# WHY the pipe-trigger, not just DUMP_ON_TIMEOUT: vLLM's default
# --distributed-timeout-seconds is 600s (design/references/vllm-serve-cli.md).
# TORCH_NCCL_DUMP_ON_TIMEOUT only fires once that full window elapses.
# GLM-5.2's confirmed hangs so far have all been manually cancelled well
# inside 600s (e.g. ~3.5 minutes in logs/glm52q/20260812_213446/) — the
# automatic dump would never have fired. See active-issues.md for the full
# writeup. report_memory() itself needs NO changes for any of this — it's
# included below unmodified, for context, since wait_report() just calls it
# an extra time when a stall is detected.
#
# HOW TO APPLY: this file is NOT sourced by anything. It's a diff target —
# copy each function into src/engine/lib/utils.sh, replacing the existing
# one of the same name (report_memory is identical to today's; only its
# call site in wait_report changes). See "WIRING NOTES" at the bottom for
# the three call sites — set_debugging_env()'s signature changed in v2 (see
# below), so those call sites need updating, not just adding.
#
# v2 UPDATE (2026-08-14), from a real run — logs/glm52q/20260814_203641/ —
# after v1 was merged into utils.sh verbatim and actually exercised at
# IVLLM_DEBUG_LEVEL=4. The trigger mechanism worked (3 stall episodes
# detected at the intended ~5-minute cooldown spacing; writing to the
# per-rank pipes reliably produced "ProcessGroupNCCL preparing to dump debug
# info" in the log), but the captured dumps themselves were useless. Two
# confirmed bugs fixed below, replacing what v1 flagged as open questions:
#
#   1. THREE of the TORCH_NCCL_* names are deprecated in this PyTorch build
#      (v2.10, confirmed from the dump's own `version` field) —
#      TORCH_NCCL_TRACE_BUFFER_SIZE, TORCH_NCCL_TRACE_CPP_STACK, and
#      TORCH_NCCL_DEBUG_INFO_TEMP_FILE. PyTorch prints a per-rank warning
#      naming the replacement (TORCH_FR_BUFFER_SIZE, TORCH_FR_CPP_STACK,
#      TORCH_FR_DUMP_TEMP_FILE) but the old names are evidently NOT honored
#      for effect — every dump produced had `entries: []` (the entire flight
#      recorder ring buffer empty), consistent with the buffer never
#      actually being sized. TORCH_NCCL_DESYNC_DEBUG, TORCH_NCCL_DUMP_ON_TIMEOUT,
#      and TORCH_NCCL_DEBUG_INFO_PIPE_FILE got no deprecation warning, so
#      those three are still current. Fix: set both old and new names
#      together below — cheap, and covers whichever PyTorch version is
#      actually in play without needing a version check.
#   2. set_debugging_env() used `${SLURM_NODEID:-0}` to build a per-node
#      pipe/temp-file path, inconsistent with report_memory()/wait_report()
#      (which take node rank as an explicit parameter, sourced by each
#      caller from $IVLLM_NODE_RANK). On the real run, every rank on BOTH
#      physical nodes ended up sharing the identical `node0`-prefixed path
#      (confirmed: `ls debug/` showed only `torch_nccl_dump_trigger_node00.pipe`
#      through `node07.pipe`, never any `node1*` variant) — meaning
#      $SLURM_NODEID resolved to 0 in both node's call to this function,
#      unlike the reliable, already-proven $IVLLM_NODE_RANK convention used
#      everywhere else. Fix: set_debugging_env() now takes an explicit node
#      parameter, matching wait_report()'s own signature/defaulting.
#
# Also confirmed and simplified: PyTorch creates its OWN per-rank
# `<given-path><rank>.pipe` files rather than reading from the single FIFO
# this function used to `mkfifo` — that manual FIFO was never read by
# anything and is removed below. trigger_torch_nccl_dump()'s glob still
# catches the real, PyTorch-created files without it.
#
# REMAINING CAVEATS, still not fully resolved:
#   - The stall-episode dedup in monitor_head() (only trigger once, then
#     cooldown) is intentionally simple. It re-arms after
#     IVLLM_STALL_COOLDOWN_SECS regardless of whether the prior dump
#     succeeded — fine for a prototype, not necessarily production-grade.
#   - Writing to a FIFO with no reader blocks indefinitely in bash; every
#     write below is wrapped in `timeout` for exactly this reason. Do not
#     remove those wrappers.
#   - Only 4 of 8 expected trace-dump files landed on disk in the real run
#     (node00-03; 04-07 never appeared despite log evidence their ranks also
#     received the trigger) — not yet root-caused. Once bug 2 above is
#     fixed and node1's ranks get their own correctly-named path, re-check
#     whether this was actually the same "shared node0 path" confusion
#     (ranks 4-7 silently overwriting/racing on node0's files) or a genuine
#     separate cross-node write issue.

# ── New: stall indicators (alongside the existing IVLLM_CRASH_INDICATORS) ──
export IVLLM_STALL_INDICATORS=(
    "No available shared memory broadcast block found"
)

# How long to wait before re-arming the stall trigger after firing once.
# Chosen to comfortably exceed one hang "episode" at the ~60s message
# repeat rate seen in logs/glm52q/20260812_213446/, without re-triggering
# on every single repeat of the same still-ongoing hang.
export IVLLM_STALL_COOLDOWN_SECS="${IVLLM_STALL_COOLDOWN_SECS:-300}"


# ── New: set_debugging_env ──────────────────────────────────────────────

# Configure vLLM/NCCL/libfabric debugging verbosity from a single master
# flag (IVLLM_DEBUG_LEVEL), per design/ivllm-environment.md's "Debugging
# flags" section. Levels 0-2 remain report_memory()'s own concern (RAM/GPU/
# pyspy) and are untouched here. Levels 3-4 export third-party env vars
# across three layers, routing file-based artifacts into the job's shared
# debug/ directory.
# Args: $1 — job name (for resolving the debug output directory);
#       $2 — node rank (default 0). Pass $IVLLM_NODE_RANK at call sites that
#       have it (ray-setup.sh, run-worker-vllm.sh) — do NOT read
#       $SLURM_NODEID directly here, v1 did and both physical nodes ended up
#       resolving to the same value on a real run (see v2 UPDATE at top of
#       file). Matches wait_report()'s own explicit-parameter convention.
# Must be called AFTER `eval "$envExports"` (job-config env: block) so that
# a job-level IVLLM_DEBUG_LEVEL is already set — see WIRING NOTES.
# No-op if IVLLM_DEBUG_LEVEL < 3 (i.e. does nothing beyond what
# report_memory() already handles for levels 0-2).
# Usage: set_debugging_env "$job" "$node_rank"
set_debugging_env() {
    local job=${1:?must set job name}
    local node="${2:-0}"
    local debug_level="${IVLLM_DEBUG_LEVEL:-0}"

    (( debug_level < 3 )) && return 0

    local dumpdir
    dumpdir=$(resolve_job_dir "$job" "debug")
    mkdir -p "$dumpdir"

    echo "[debug] IVLLM_DEBUG_LEVEL=$debug_level — enabling verbose third-party diagnostics"

    # ── Layer 1: vLLM's own logger ──────────────────────────────────────
    export VLLM_LOGGING_LEVEL=DEBUG

    # ── Layer 2: NCCL / torch.distributed ────────────────────────────────
    export NCCL_DEBUG=INFO

    # ── Layer 3: libfabric / CXI ──────────────────────────────────────────
    # Deliberately NOT trace here — confirmed on this platform that `info`
    # is more informative than `trace` at level 3 (see active-issues.md /
    # ivllm-environment.md — non-monotonic verbosity, don't "fix" this).
    export FI_LOG_LEVEL=info
    export FI_LOG_PROV=cxi

    if (( debug_level < 4 )); then
        return 0
    fi

    # ── Level 4: targeted trace, for actively chasing a live hang ────────
    # High log volume — only reached at the top debug level.

    # Layer 2, upgraded: trace-level NCCL plus PyTorch's flight recorder /
    # desync-debug family. This is the new, not-yet-community-precedented
    # diagnostic identified in active-issues.md as the current top priority
    # for the GLM-5.2 shm_broadcast hang.
    #
    # Both old and new names are set for TRACE_BUFFER_SIZE/TRACE_CPP_STACK/
    # DEBUG_INFO_TEMP_FILE — confirmed on a real run (logs/glm52q/20260814_203641/,
    # PyTorch 2.10) that the TORCH_NCCL_* forms of exactly these three are
    # deprecated in favour of TORCH_FR_*, and that the old names are NOT
    # honored for effect (every dump came back with an empty `entries: []`
    # ring buffer despite TORCH_NCCL_TRACE_BUFFER_SIZE being set). Setting
    # both costs nothing and covers whichever name this PyTorch build
    # actually reads. DESYNC_DEBUG/DUMP_ON_TIMEOUT/DEBUG_INFO_PIPE_FILE got
    # no deprecation warning on that same run, so those three are unchanged.
    export NCCL_DEBUG=TRACE
    export NCCL_DEBUG_SUBSYS=COLL,PROXY
    export TORCH_NCCL_DESYNC_DEBUG=1
    export TORCH_NCCL_DUMP_ON_TIMEOUT=1
    export TORCH_NCCL_TRACE_BUFFER_SIZE=2000
    export TORCH_FR_BUFFER_SIZE=2000
    export TORCH_NCCL_TRACE_CPP_STACK=1
    export TORCH_FR_CPP_STACK=1
    export TORCH_NCCL_DEBUG_INFO_TEMP_FILE="$dumpdir/torch_nccl_trace_node${node}"
    export TORCH_FR_DUMP_TEMP_FILE="$dumpdir/torch_nccl_trace_node${node}"
    export TORCH_NCCL_DEBUG_INFO_PIPE_FILE="$dumpdir/torch_nccl_dump_trigger_node${node}"

    # No manual mkfifo here (v1 had one) — confirmed on the real run that
    # PyTorch creates its own per-rank `<given-path><rank>.pipe` files and
    # never reads a pre-created FIFO at the literal given path. Nothing
    # would ever have opened the old manual FIFO for reading.

    # Layer 3, upgraded: targeted libfabric subsystem trace.
    export FI_LOG_SUBSYS=cq,ep_data,mr

    echo "[debug] level 4 diagnostics armed — flight-recorder trigger base: $TORCH_NCCL_DEBUG_INFO_PIPE_FILE (node $node)"
}


# ── New: trigger_torch_nccl_dump (helper, called from monitor_head) ─────

# Best-effort, non-blocking trigger of every node's torch_nccl_dump_trigger
# FIFO under a job's debug/ directory. Safe to call even if no node ever
# reached IVLLM_DEBUG_LEVEL>=4 (glob simply matches nothing).
# Args: $1 — job name.
# Usage: trigger_torch_nccl_dump "$job"
trigger_torch_nccl_dump() {
    local job=${1:?must set job name}
    local dumpdir
    dumpdir=$(resolve_job_dir "$job" "debug")

    local pipe
    local found=0
    for pipe in "$dumpdir"/torch_nccl_dump_trigger_node*; do
        [[ -p "$pipe" ]] || continue
        found=1
        echo "[head] triggering flight-recorder dump via $pipe"
        # Confirmed on a real run: PyTorch creates one distinct .pipe file
        # per rank (<given-path><rank>.pipe), so this glob already targets
        # one reader per file, not several processes sharing one FIFO as
        # v1 worried it might. Still writing several times, cheaply, as a
        # defensive margin against the write racing the reader's own
        # open()/read() setup rather than for a multiple-readers reason.
        local i
        for i in 1 2 3 4 5; do
            timeout 2 bash -c 'echo dump > "$1" 2>/dev/null' _ "$pipe"
            sleep 0.2
        done
    done

    if (( found == 0 )); then
        echo "[head] no torch_nccl_dump_trigger pipes found under $dumpdir — IVLLM_DEBUG_LEVEL<4 this run?"
    fi
}


# ── Modified: monitor_head ────────────────────────────────────────────────

# Background monitor on slurm step node that runs for the entire job lifetime.
# CHANGES from production utils.sh: adds a stall-detection check alongside
# the existing crash-detection check. On first sight of an
# IVLLM_STALL_INDICATORS match since the last cooldown window, writes a
# shared sentinel file (debug/stall_detected, containing the trigger
# timestamp — wait_report() on every node watches for this) and calls
# trigger_torch_nccl_dump(). Does NOT change monitor_head's control flow or
# exit codes otherwise — a stall is not itself a crash or a reason to shut
# down; ivllm cancel remains a manual/operator decision.
# Usage: monitor_head "$job" &
monitor_head() {
    local job="$1"
    local lockfile
    local log
    local idle_timeout
    local server_port
    local model
    local last_stall_trigger_epoch=0

    lockfile=$(resolve_job_status "$job")
    log=$(resolve_job_log "$job")
    idle_timeout=$(get_job_status_setting "$job" ".idleTimeout")

    server_port=$(get_job_status_setting "$job" ".serverPort")
    model=$(get_job_status_setting "$job" ".model")

    local debug_level=$(get_job_config_setting "$job" ".ivllm-debug-level")
    debug_level=${debug_level:-0}

    if [ ! -f "$lockfile" ]; then
        echo "[head] FATAL: lockfile $lockfile missing on startup"
        return 250
    fi

    echo "[head] starting monitor (idle_timeout=$idle_timeout)..."

    while true; do

        # Lockfile deleted
        if [ ! -f "$lockfile" ]; then
            echo "[head] lockfile $lockfile has been deleted — shutting down"
            return 250
        fi

        local status
        status=$(get_job_status_setting "$job" ".status")

        # Terminal states — exit loop
        if [[ $status == "failed" ]]; then
            echo "[head] WARNING: monitor detected failed status"
            return 251
        fi

        if [[ $status == "stopped" ]]; then
            echo "[head] WARNING: monitor detected stopped status"
            return 203
        fi

        # Still pending — wait for SLURM allocation
        if [[ $status == "pending" ]]; then
            sleep "$IVLLM_CHECK_INTERVAL_SECS"
            continue
        fi

        # User requested cancel
        if [[ $status ==  "cancel" ]]; then
            echo "[head] user cancel request detected"
            return 201
        fi

        # User requested abort
        if [[ $status ==  "abort" ]]; then
            echo "[head] user abort request detected"
            return 254
        fi

        # Exit at any time (running or initialising if we detect a crash in logs)
        local crash_patterns=()
        for crash in "${IVLLM_CRASH_INDICATORS[@]}"; do
            crash_patterns+=("-e" "$crash")
        done

        local recent_log
        recent_log=$(tail -n 5000 "$log" 2>/dev/null)

        # Check for recent crash related messages
        if grep -q -F "${crash_patterns[@]}" <<< "$recent_log"; then
            echo "[head] monitor detected a crash in head log file."
            sleep 60
            echo "[head] monitor shutting down."
            return 253
        fi

        # ── New: stall detection ─────────────────────────────────────────
        # Does not affect control flow — a stall is a diagnostic event,
        # not a shutdown condition. Only meaningful once IVLLM_DEBUG_LEVEL
        # >= 4 has actually created the trigger pipes; trigger_torch_nccl_dump
        # is a harmless no-op otherwise.
        local stall_patterns=()
        for stall in "${IVLLM_STALL_INDICATORS[@]}"; do
            stall_patterns+=("-e" "$stall")
        done

        if grep -q -F "${stall_patterns[@]}" <<< "$recent_log"; then
            local now_epoch
            now_epoch=$(date +%s)
            if (( now_epoch - last_stall_trigger_epoch >= IVLLM_STALL_COOLDOWN_SECS )); then
                echo "[head] stall indicator seen in log — triggering flight-recorder dump"
                local dumpdir
                dumpdir=$(resolve_job_dir "$job" "debug")
                # Shared sentinel: wait_report() on every node (including
                # this one) watches this file's contents to grab an extra,
                # synchronized report_memory() snapshot at the same moment.
                date +%s > "$dumpdir/stall_detected"
                trigger_torch_nccl_dump "$job"
                last_stall_trigger_epoch=$now_epoch
            fi
        fi

        # No crash marker detected - has vllm come up?

        # Still initialising — skip idle checks
        if [[ $status ==  "initialising" ]]; then
            if curl -sf "http://localhost:$server_port/health" > /dev/null 2>&1; then

                echo "[startup] vLLM /health active — saving JIT cache"
                save_cache "$job"

                local max_retries=5
                local attempt=1
                local warmup_ok=1

                echo "[startup] sending warmup request..."
                while (( attempt <= max_retries )); do

                    local tmp_status=$(get_job_status_setting "$job" ".status")
                    if [[ ! $tmp_status == "initialising" ]]; then
                        warmup_ok=1
                        echo "[head] status $tmp_status during warmup"
                        break;
                    fi

                    if run_vllm_warmup "$model" "$server_port"; then
                        warmup_ok=0
                        break;
                    fi
                    echo "[startup] WARNING: warmup attempt $attempt/$max_retries failed, retrying..."
                    (( attempt++ ))
                    sleep 10
                done

                if (( warmup_ok == 0 )); then
                    echo "[startup] warmup complete"
                    echo "[startup] job $job startup complete."
                    update_status_running "$job"
                    echo "[startup] startup complete: vLLM is running."
                    continue
                else
                    echo "[startup] ERROR: warmup failed after $max_retries attempts"
                    echo "[startup] signalling for vllm to shut down."
                    echo "[startup] startup complete: vLLM failed warmup."
                    return 252
                fi
            else
                echo "[startup] job $job waiting for vLLM /health"
                sleep "$IVLLM_CHECK_INTERVAL_SECS"
                continue
            fi
        fi

        # Running — check idle timeout
        if [[ $status == "running" && -n "$idle_timeout" && "$idle_timeout" -ge 0 ]]; then

            local time_patterns=()
            for i in $(seq 0 "$idle_timeout"); do
                time_patterns+=("-e" "$(date -d "$i minutes ago" "$IVLLM_TIME_FMT")")
            done

            local endpoint_patterns=()
            for endpoint in "${IVLLM_TARGET_ENDPOINTS[@]}"; do
                endpoint_patterns+=("-e" "$endpoint")
            done

            if grep -F "${time_patterns[@]}" <<< "$recent_log" | grep -q -F "${endpoint_patterns[@]}"; then
                sleep "$IVLLM_CHECK_INTERVAL_SECS"
                continue
            else
                echo "[head] no API requests for $idle_timeout minutes — shutting down"
                return 202
            fi
        fi

        sleep "$IVLLM_CHECK_INTERVAL_SECS"
    done

    echo "[head] monitor shutting down for job $job."
    return 0
}


# ── Modified: wait_report ─────────────────────────────────────────────────

# Waits for a process to finish whilst reporting on its memory usage.
# CHANGES from production utils.sh: in addition to the existing
# initialising/IVLLM_RUNTIME_DEBUG-gated periodic report_memory() calls,
# also watches the shared debug/stall_detected sentinel (written by
# monitor_head() above, on whichever node runs it) and — the first time it
# sees a *new* timestamp there since this loop started watching — calls
# report_memory() immediately, once, out of the normal cadence. Runs on
# every node (head and workers alike), so a stall gives one synchronized
# pyspy/GPU snapshot from every node at close to the same real moment,
# rather than relying on independent ~10s ticks to happen to land close
# together.
# Args: $1 — job; $2 - pid for the process to monitor; $3 - node id.
# Usage: wait_report "$job" "$pid" "$node"
wait_report() {
    local job=${1:?must provide job}
    local pid=${2:?must provide pid}
    local node=${3:-0}
    local elapsed=0
    local tick_ms=100
    local target_ms
    (( target_ms = IVLLM_CHECK_INTERVAL_SECS * 1000 ))

    local dumpdir
    dumpdir=$(resolve_job_dir "$job" "debug")
    local stall_sentinel="$dumpdir/stall_detected"
    local last_seen_stall=""

    while ! process_died "$pid"; do
        if [ "$elapsed" -ge "$target_ms" ]; then
            if is_status "$job" "initialising" || [[ "${IVLLM_RUNTIME_DEBUG:-0}" == "1" ]]; then
                report_memory "$job" "$node"
            fi
            elapsed=0
        fi

        # ── New: stall-triggered extra snapshot ──────────────────────────
        if [[ -f "$stall_sentinel" ]]; then
            local current_stall
            current_stall=$(<"$stall_sentinel")
            if [[ -n "$current_stall" && "$current_stall" != "$last_seen_stall" ]]; then
                echo "[serve-$node] stall detected (sentinel: $current_stall) — capturing extra snapshot"
                report_memory "$job" "$node"
                last_seen_stall="$current_stall"
            fi
        fi

        sleep 0.1
        ((elapsed += tick_ms))
    done

    wait "$pid" 2>/dev/null
    local code=$?

    if [[ $code != 0 ]]; then
        echo "[serve-$node] vllm crashed with exit code $code"
        return $code
    else
        echo "[serve-$node] vllm exited normally"
        sleep 1
    fi

    return 1
}


# ── Unchanged: report_memory (included for context only — no edits needed) ──

# Report memory and JIT cache usage for the current node.
# No changes from production utils.sh. wait_report() above just calls this
# an extra time on stall detection; report_memory() itself doesn't need to
# know why it was called.
# Args: $1 — job name; $2 - node id.
# Usage: report_memory "$job" "$node"
report_memory() {
      local job="${1:?must provide job}"
      local localdir
      local node

      localdir=$(resolve_localdir "$job")
      node=${2:-0}

      local raw_ps
      raw_ps=$(ps -u "$USER" -o pid=,rss=,comm= 2>/dev/null || true)

      local total_ram
      total_ram=$(echo "$raw_ps" | awk '{sum+=$2} END{if(sum>1024) printf "%dM", sum/1024; else printf "%dK", sum}')

      local top_6
      top_6=$(echo "$raw_ps" | awk '{m[$3]+=$2} END{for(c in m) printf "%d %s\n", m[c], c}' | sort -rn | head -n 6 | awk '{if($1>1024) printf "%s=%dM ",$2,$1/1024; else printf "%s=%dK ",$2,$1}')

      printf "[%s-node %s] Cache: %sK | RAM: %s | Top: %s\n" \
          "$(date +%H:%M:%S)" "$node" \
          "$(du -sk "$localdir" 2>/dev/null | cut -f1)" "$total_ram" "$top_6"

      local debug_level="${IVLLM_DEBUG_LEVEL:-0}"
      (( debug_level < 1 )) && return 0

      if command -v nvidia-smi &>/dev/null; then
          local gpu_line
          gpu_line=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total \
              --format=csv,noheader,nounits 2>/dev/null | \
              awk -F', ' '{printf "gpu%s=%s%%/%sM ", $1, $2, $3}')
          printf "[%s-node %s] GPU: %s\n" "$(date +%H:%M:%S)" "$node" "$gpu_line"
      fi

      (( debug_level < 2 )) && return 0

      if command -v py-spy &>/dev/null; then
          local dumpdir
          dumpdir=$(resolve_job_dir "$job" "debug")
          mkdir -p "$dumpdir"
          local dumpfile="$dumpdir/pyspy-node${node}.log"
          local dumped=0
          local pid
          local rss
          local comm

          {
              echo "### $(date +%Y-%m-%dT%H:%M:%S) ###"
              while read -r pid rss comm; do
                  case "$comm" in
                      *RayWorkerP*|*EngineCor*|vllm|*VLLM*)
                          echo "=== pid $pid rss=${rss}K comm=$comm ==="
                          py-spy dump --pid "$pid" --nonblocking 2>&1
                          echo
                          (( dumped++ ))
                          ;;
                  esac
              done <<< "$raw_ps"
          } >> "$dumpfile"

          (( dumped > 0 )) && printf "[%s-node %s] [debug] appended %d py-spy dump(s) to %s\n" \
              "$(date +%H:%M:%S)" "$node" "$dumped" "$dumpfile"
      else
          printf "[%s-node %s] [debug] IVLLM_DEBUG_LEVEL=%s requested py-spy but it is not installed\n" \
              "$(date +%H:%M:%S)" "$node" "$debug_level"
      fi
  }


# ── WIRING NOTES ───────────────────────────────────────────────────────────
#
# 1. Copy set_debugging_env(), trigger_torch_nccl_dump(), and the
#    IVLLM_STALL_INDICATORS/IVLLM_STALL_COOLDOWN_SECS exports into
#    src/engine/lib/utils.sh (near set_jit_caches()/IVLLM_CRASH_INDICATORS
#    respectively).
#
# 2. Replace monitor_head() and wait_report() in utils.sh with the versions
#    above. report_memory() is unchanged — no edit needed there, it's only
#    included for context.
#
# 3. Add ONE call to set_debugging_env() in each of the three node-level
#    scripts, placed AFTER `eval "$envExports"` — NOT alongside
#    set_jit_caches() near the top of each script. IVLLM_DEBUG_LEVEL is a
#    job-config env: block setting, so it doesn't exist yet at the point
#    set_jit_caches() currently runs; calling set_debugging_env() before
#    envExports is evaluated would silently always see level 0.
#
#      run-head-vllm.sh:   after line 76 (`eval "$envExports"`), before the
#                           "Selected environment exports" echo / vllm serve launch.
#                           Call: `set_debugging_env "$IVLLM_JOB" 0`
#                           (head node is always rank 0 — this script has no
#                           $IVLLM_NODE_RANK variable of its own).
#      run-worker-vllm.sh: same point, after its own `eval "$envExports"`.
#                           Call: `set_debugging_env "$IVLLM_JOB" "$IVLLM_NODE_RANK"`
#                           ($IVLLM_NODE_RANK is already a required arg to
#                           this script — line 13 — reuse it, do not read
#                           $SLURM_NODEID; see v2 UPDATE at top of file for why.)
#      ray-setup.sh:       after its `eval "$envExports"` (line 61), before
#                           `source "$vllmVersionDir/bin/activate"` / `ray start`.
#                           Call: `set_debugging_env "$IVLLM_JOB" "$IVLLM_NODE_RANK"`
#                           ($IVLLM_NODE_RANK is set at line 20 of this
#                           script from its own $3 positional arg — reuse it.)
#
#    v1 called `set_debugging_env "$IVLLM_JOB"` (one arg) at all three sites
#    — if updating an existing merge rather than applying fresh, all three
#    call sites need their second argument added, not just the function body.
#
# 4. Nothing else calls trigger_torch_nccl_dump() directly — only
#    monitor_head() does, on the head node, since that's the only place the
#    aggregated log (containing EngineCore's stall warning) is ever tailed.
#
# 5. Confirmed on the real run (logs/glm52q/20260814_203641/) and still
#    worth checking again after applying the v2 fixes above: (a) the
#    trigger mechanism itself fires correctly — "ProcessGroupNCCL preparing
#    to dump debug info" appears in the log right after each triggered
#    write; (b) whether all 8 expected trace-dump files land this time
#    (only 4 of 8 did pre-fix — see REMAINING CAVEATS at top of file); (c)
#    whether `entries` in the dumped pickle is non-empty now that
#    TORCH_FR_BUFFER_SIZE is set alongside the deprecated name.
