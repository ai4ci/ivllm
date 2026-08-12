#!/bin/bash
# design/prototype/ivllm-bench.sh — PROTOTYPE, not production (see AGENTS.md:
# scripts in design/ are instructional examples, rewrite before shipping).
#
# Draft of the ADR-118 (revised) "shadow project directory" benchmarking
# design, taken one step further per discussion: instead of adding an
# IVLLM_BENCH_MODE hook inside monitor_head() (a change to an existing
# production file), this version is a **standalone login-node orchestrator**
# that drives the existing external CLI surface — ivllm-serve.sh,
# ivllm-cancel.sh, and utils.sh's status/config helpers (sourced, not
# reimplemented). One small production change was needed alongside this:
# capture_job_diagnostics() (utils.sh) now does `cp -rf "$job_dir"/*
# "$diag_dir/"` — sweep everything in the job directory, not a hardcoded
# vllm.yaml/status.json/vllm*.log list — specifically so this script's
# bench.json (and IVLLM_DEBUG_LEVEL=2's debug/pyspy dumps) get archived too
# without needing a parallel, separately-timestamped copy step of their own.
#
# The login node cannot reach a compute node's port directly (confirmed —
# Isambard's compute nodes are network-isolated from the login node except
# via SLURM itself), so the `vllm bench serve` CLIENT is launched via `srun
# --overlap --jobid=<slurmJobId>` (the job ID read straight out of the
# lockfile) — riding on the already-running job's existing allocation to
# execute the bench client ON the compute node hosting the API server, no
# new resource request. This is the same `--overlap` pattern
# slurm-vllm-serve.sh/slurm-ray-vllm-serve.sh already use internally for
# their own head/worker srun steps, just invoked here from outside the job.
# Net effect: this ends up MORE honest than a login-node network hop would
# have been — genuine same-node `localhost`, not a workaround, a bonus.
#
# What this does, end to end, for `ivllm-bench.sh -c <comparison_directory>`:
#   1. Reads every `*.yaml`/`*.yml` file in <comparison_directory> — one
#      candidate vLLM config per file, job name = filename stem.
#   2. Creates a "benchmark" shadow project directory as a subfolder of
#      <comparison_directory> itself (not of the real project dir — each
#      comparison run gets its own fully independent shadow tree this way,
#      so two different comparisons can reuse the same config/job names
#      without colliding). The REAL project directory (read from
#      $PROJECTDIR/$IVLLM_PROJECTDIR, which must already be set in the
#      environment — same convention utils.sh itself uses) is only
#      consulted as the symlink *target*: engine/vllm, engine/nvhpc,
#      engine/rdma, and model are symlinked back to it so nothing is
#      re-downloaded or recompiled; engine/jobs and engine/diagnostics are
#      real, independent directories under the shadow tree.
#   3. For each config: copies it into place at the path utils.sh expects.
#      Prior state for the same job name is deliberately NOT wiped —
#      rerunning a comparison adds a new timestamped entry alongside
#      previous ones in diagnostics rather than overwriting history. Then,
#      before anything is submitted, every UNIQUE `.model` across all
#      configs is prefetched once, sequentially (`ivllm-get-model.sh`,
#      same tool `ivllm-serve.sh` already calls inline) — deliberately not
#      "hard fail if not cached", since we can just fetch it, but a
#      genuinely-undownloadable model aborts the whole comparison here
#      rather than after every dependent job separately fails post-queue.
#      This matters more than it sounds: sharing one base model across
#      configs with different parallelism/quant settings is the COMMON
#      case for a tuning comparison, not an edge case — without this,
#      concurrently-submitted jobs needing the same uncached model would
#      race to download it to the same destination simultaneously.
#   4. Submits all jobs concurrently via the real `ivllm-serve.sh -j <job>
#      -b` (non-interactive batch partition — already skips the interactive
#      reservation's 1-job limit, no new code needed for that).
#   5. Polls each job's lockfile (via utils.sh's own `is_status`/
#      `get_job_status_setting` — sourced, not reimplemented) until it
#      reaches "running", with a wall-clock giving-up point per job so a
#      stuck/hung job (see design/active-issues.md — this exists for a
#      reason) can't hang the whole orchestrator forever.
#   6. Once running: via `srun --overlap --jobid=<slurmJobId>` targeting the
#      job's own `computeHostname`, runs a health check then `vllm bench
#      serve --host localhost ...` co-located with the API server (see
#      above), saving JSON output straight into the job's own working
#      directory (swept into diagnostics in step 7, alongside everything
#      else). The venv activation happens INSIDE the srun'd remote shell,
#      not on the login node — needs to execute where `vllm` actually needs
#      to resolve on PATH.
#   7. Requests a graceful cancel (`ivllm-cancel.sh -j <job>`), waits for the
#      job to actually reach a terminal status (cancel is fire-and-forget —
#      shutdown happens asynchronously once monitor_head notices it), then
#      calls `capture_job_diagnostics()` explicitly — its cancel/timeout/
#      idle-timeout paths (exit codes 200-203, utils.sh:837-895) deliberately
#      don't call it themselves, only crash paths do, so a normal successful
#      benchmark run would never get archived without this.
#   8. Once every job has finished, copies the whole diagnostics tree back
#      into <comparison_directory>/results/ and prints a plain status table
#      — no verdict, per ADR-118's original design intent.
#
# All 8 job-lifecycle steps run in parallel, one per config, as independent
# background subshells — this is the "trivially parallelisable across
# configs" property ADR-118 always wanted.
#
# Still untested/worth checking on a real run before trusting this fully:
#   - `srun --overlap --jobid=<id>` targeting a job submitted by a
#     DIFFERENT `sbatch` invocation (this script's own, not the running
#     job's own orchestrator subshell) — used internally elsewhere in this
#     codebase only from within a job targeting its OWN allocation, never
#     yet from an external script attaching to a job by ID after the fact.
#     Should work (that's what --overlap + --jobid is for), but hasn't
#     actually been run.
#   - Exact `vllm bench serve` flags below may need adjusting per vLLM
#     version — check `vllm bench serve --help` against whatever
#     min-vllm-version each config declares.
#
# Usage (run on the Isambard login node, with $PROJECTDIR/$IVLLM_PROJECTDIR
# already set to the REAL project directory — same as any other ivllm
# command expects):
#   bash ivllm-bench.sh -c ~/bench-configs/nemotron-tuning/
#
# where ~/bench-configs/nemotron-tuning/ contains e.g. tp4dp2.yaml,
# tp8.yaml, wide-ep.yaml — one candidate config per file.
#
# Structure note: everything below is either a plain constant/function
# DEFINITION (safe to `source` this file for testing — see
# tests/bash/sandboxed/test-ivllm-bench.sh, which does exactly that to unit
# test write_status_summary()/the model-prefetch loop against real
# status.json/vllm.yaml fixtures without ever touching SLURM) or lives
# inside main(), which only actually runs when this file is EXECUTED
# directly, via the source-guard at the very bottom. Nothing at the top
# level of the file has a side effect or can exit the sourcing shell.

set -uo pipefail

IVLLM_BIN="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REAL_PROJECTDIR="${IVLLM_PROJECTDIR:-${PROJECTDIR:-}}"
JOB_TIME="${IVLLM_BENCH_TIME:-02:00:00}"          # matches ADR-118's 2h default
MAX_WAIT_SECS="${IVLLM_BENCH_MAX_WAIT_SECS:-3600}" # give up waiting for "running" after this long
POLL_INTERVAL_SECS=15

usage() {
    echo "Usage: $0 -c <comparison_directory>"
    echo ""
    echo "  -c dir   Directory containing one vLLM config (*.yaml/*.yml) per"
    echo "           candidate to benchmark. Job name = filename stem."
    echo ""
    echo "Requires \$PROJECTDIR or \$IVLLM_PROJECTDIR to already be set to the"
    echo "REAL (non-benchmark) project directory in the environment."
    exit 1
}

# ── Steps 3-7: one job, start to finish. Runs as an independent background
# subshell per config — this is where the "trivially parallelisable across
# configs" property comes from. Only calls utils.sh functions at CALL time,
# not definition time, so it's safe to define here even though utils.sh
# isn't sourced until main() runs. ──
process_job() {
    local job="$1"
    local jobDir vllmVersion vllmVersionDir minVllmVersion model computeHostname serverPort slurmJobId
    local waited=0

    echo "[bench:$job] preparing"

    # Config is already copied into place (and its model already prefetched)
    # by the discovery/prefetch pass in main(), before any process_job
    # started — see the comment there for why.
    jobDir=$(resolve_job_dir "$job")

    echo "[bench:$job] submitting (batch partition, -t $JOB_TIME)"
    if ! "$IVLLM_BIN/ivllm-serve.sh" -j "$job" -b -t "$JOB_TIME" >"$jobDir/bench-submit.log" 2>&1; then
        echo "[bench:$job] FAILED to submit — see $jobDir/bench-submit.log"
        return 1
    fi

    echo "[bench:$job] waiting for status=running (up to ${MAX_WAIT_SECS}s)"
    while true; do
        if is_status "$job" "running"; then
            break
        fi
        if is_status "$job" "failed" || is_status "$job" "stopped"; then
            echo "[bench:$job] FAILED to reach running (status: $(get_job_status_setting "$job" ".status"), reason: $(get_job_status_setting "$job" ".reason"))"
            return 1
        fi
        if (( waited >= MAX_WAIT_SECS )); then
            echo "[bench:$job] TIMED OUT waiting for running after ${MAX_WAIT_SECS}s — force cancelling"
            "$IVLLM_BIN/ivllm-cancel.sh" -j "$job" -f >>"$jobDir/bench-submit.log" 2>&1
            return 1
        fi
        sleep "$POLL_INTERVAL_SECS"
        (( waited += POLL_INTERVAL_SECS ))
    done

    model=$(get_job_status_setting "$job" ".model")
    computeHostname=$(get_job_status_setting "$job" ".computeHostname")
    serverPort=$(get_job_status_setting "$job" ".serverPort")
    slurmJobId=$(get_job_status_setting "$job" ".slurmJobId")
    echo "[bench:$job] running on $computeHostname:$serverPort ($model, slurm job $slurmJobId)"

    minVllmVersion=$(get_job_config_setting "$job" ".min-vllm-version")
    vllmVersion=$(select_closest_version "$minVllmVersion")
    vllmVersionDir=$(resolve_vllm_version_dir "$vllmVersion")

    # The login node cannot reach a compute node's port directly (confirmed —
    # not just an untested assumption, see conversation history). So the
    # bench client must actually EXECUTE on the compute node, not just be
    # aimed at it from here. `srun --overlap --jobid=<slurmJobId>` rides on
    # the ALREADY-RUNNING job's existing allocation (no new resource
    # request, same pattern slurm-vllm-serve.sh/slurm-ray-vllm-serve.sh
    # already use internally for their own head/worker srun steps — just
    # invoked here from outside the job, against a job ID read back out of
    # the lockfile) to run a lightweight extra step on the head node.
    # Bonus: this also means the client hits genuine same-node `localhost`,
    # which is a MORE honest measurement than the original login-node/
    # network-hop design, not just a workaround for the reachability gap.
    local overlapArgs=(--overlap --jobid="$slurmJobId" --nodelist="$computeHostname" --nodes=1 --ntasks=1)

    echo "[bench:$job] health check via srun --overlap (confirms the compute node is actually reachable this way)"
    if ! srun "${overlapArgs[@]}" curl -sf --max-time 10 "http://localhost:$serverPort/health" >>"$jobDir/bench-submit.log" 2>&1; then
        echo "[bench:$job] ERROR: srun --overlap health check failed — see $jobDir/bench-submit.log for srun's own error"
        "$IVLLM_BIN/ivllm-cancel.sh" -j "$job" >>"$jobDir/bench-submit.log" 2>&1
        return 1
    fi

    # capture_job_diagnostics() now does `cp -rf "$job_dir"/* "$diag_dir/"`
    # (changed to sweep everything, incl. IVLLM_DEBUG_LEVEL=2's debug/pyspy
    # dumps, not just a hardcoded vllm.yaml/status.json/vllm*.log list) — so
    # writing straight into $jobDir and archiving once at the end is enough;
    # no separate bench-specific folder needed.
    if srun "${overlapArgs[@]}" bash -c "
        source '$vllmVersionDir/bin/activate' &&
        vllm bench serve \
            --host localhost \
            --port '$serverPort' \
            --model '$model' \
            --dataset-name random \
            --save-result \
            --result-dir '$jobDir' \
            --result-filename bench.json
        " > "$jobDir/bench.log" 2>&1
    then
        echo "[bench:$job] bench complete — $jobDir/bench.json"
    else
        echo "[bench:$job] vllm bench serve FAILED — see $jobDir/bench.log"
    fi

    echo "[bench:$job] requesting graceful cancel"
    "$IVLLM_BIN/ivllm-cancel.sh" -j "$job" >>"$jobDir/bench-submit.log" 2>&1

    # ivllm-cancel.sh only writes the cancel request and returns immediately
    # — the real shutdown (and the status.json update we actually want to
    # read) happens asynchronously once monitor_head notices it. Capturing
    # diagnostics before that finishes would archive a stale in-flight
    # snapshot instead of the final status/reason.
    local cancelWaited=0
    while ! is_status "$job" "stopped" && ! is_status "$job" "failed"; do
        if (( cancelWaited >= 300 )); then
            echo "[bench:$job] WARNING: graceful cancel did not complete within 300s — force cancelling so the allocation doesn't sit there indefinitely"
            "$IVLLM_BIN/ivllm-cancel.sh" -j "$job" -f >>"$jobDir/bench-submit.log" 2>&1
            break
        fi
        sleep 5
        (( cancelWaited += 5 ))
    done

    # tidy_up()'s cancel/timeout/idle-timeout paths (exit codes 200-203)
    # deliberately do NOT call this themselves (see utils.sh:837-895) — only
    # crash paths do. A benchmark run's logs/config are the actual point of
    # the run, not just failure evidence, so archive unconditionally here.
    capture_job_diagnostics "$job"
}

# ── Fire-and-forget client contract ──────────────────────────────────────
#
# This script is meant to be launched DETACHED and NOT waited on
# synchronously by its caller — a comparison across several large
# multi-node configs can run for hours. What a future TypeScript
# `ivllm compare --submit` needs to implement:
#
# 1. Launch, non-blocking. Don't sbatch/srun this script itself — it does
#    no GPU work of its own, just submits and polls other jobs, so making
#    it a SLURM job would occupy a scheduler slot for no benefit. Instead,
#    over the existing SSH connection (RemoteOps.runRemote()), detach it
#    the standard way — nohup + closed stdin + disown, so the remote
#    process survives the SSH command/session closing:
#      ssh <login-host> 'nohup bash ivllm-bench.sh -c <remoteComparisonDir> \
#        > <remoteComparisonDir>/orchestrator.log 2>&1 < /dev/null & disown; \
#        echo started'
#    runRemote() returns as soon as "started" comes back — well before any
#    benchmark job even reaches "running".
#
# 2. Poll progress, cheap and safe to call often (plain file read, no SLURM
#    calls of its own): read <comparison_directory>/benchmarking_status.json
#    (rsync/cat it down). Schema:
#      {
#        "pid": <int>,               // this script's own PID on the login node
#        "updated": "<ISO8601 UTC>", // when this file was last written
#        "complete": <bool>,         // true once EVERY job has a terminal status
#        "counts": {"pending": N, "initialising": N, "running": N,
#                   "stopped": N, "failed": N},
#        "jobs": {"<jobName>": {"status": "...", "reason": "..." | null}, ...}
#      }
#    `complete` is the ONLY field that authoritatively means "safe to fetch
#    results" — treat `counts`/`jobs` as progress display only.
#
# 3. Detect a dead orchestrator (file present, `complete` stuck false).
#    Check BOTH staleness of `updated` (no update for several times
#    POLL_INTERVAL_SECS) AND whether `pid` is still alive (a remote
#    `kill -0 <pid>` check, e.g. via
#    `ssh <login-host> "kill -0 <pid> 2>/dev/null && echo alive || echo dead"`).
#    Staleness alone isn't reliable — one slow `vllm bench serve` call can
#    legitimately leave the file looking old for a while with nothing wrong.
#
# 4. Retrieve results, only once `complete: true`:
#    copyDirectory(<comparison_directory>/results/, ..., 'down') — one
#    subdirectory per job, each in turn containing EVERY timestamped
#    capture_job_diagnostics() folder ever produced for that job name
#    across however many times this comparison directory has been rerun
#    (status.json, bench.json, vllm.yaml, vllm.*.log, and at
#    IVLLM_DEBUG_LEVEL>=2, debug/pyspy dumps, per run) — history is kept
#    deliberately, not collapsed to the latest, since comparing a rerun
#    against a previous attempt after tweaking a config is the actual
#    point. The printed summary table (one row per job PER RUN, oldest
#    first) is ALSO saved as <comparison_directory>/results/summary.txt for
#    the same reason nothing is watching this script's stdout live in
#    fire-and-forget mode.
#
# Not implemented here, left for the CLI layer: cancelling a whole
# comparison early. Today that means killing the orchestrator PID (stops it
# submitting/polling further) AND separately `ivllm-cancel.sh -j <job>` per
# already-submitted job — no single command does both yet.
#
# $STATUS_FILE and $ORCHESTRATOR_PID are globals set by main() (they depend
# on $COMPARISON_DIR, only known once main() parses argv) — both functions
# below only read them at CALL time, so defining them here, before main(),
# is safe.

# Rewrites $STATUS_FILE from scratch each call — cheap enough (a handful of
# jobs, each just a jq read of its own already-small status.json) to call
# on every poll tick rather than trying to diff/patch incrementally.
write_status_summary() {
    local job status reason
    local pending=0 initialising=0 running=0 stopped=0 failed=0 unknownCount=0
    local jobsJson="{}"

    for job in "${jobNames[@]}"; do
        status=$(get_job_status_setting "$job" ".status" 2>/dev/null)
        [[ -z "$status" ]] && status="unknown"
        reason=$(get_job_status_setting "$job" ".reason" 2>/dev/null)

        case "$status" in
            pending) (( pending++ )) ;;
            initialising) (( initialising++ )) ;;
            running) (( running++ )) ;;
            stopped) (( stopped++ )) ;;
            failed) (( failed++ )) ;;
            *) (( unknownCount++ )) ;;
        esac

        jobsJson=$(jq -n --argjson prev "$jobsJson" --arg job "$job" --arg status "$status" --arg reason "$reason" \
            '$prev + {($job): {status: $status, reason: (if $reason == "" then null else $reason end)}}')
    done

    # Terminal = every job has left the pending/initialising/running states.
    # "unknown" (e.g. a job whose lockfile briefly doesn't exist yet) is
    # deliberately NOT terminal — better to keep polling than declare done
    # on a transient read glitch.
    local complete="false"
    (( pending + initialising + running + unknownCount == 0 )) && complete="true"

    jq -n \
        --argjson pid "$ORCHESTRATOR_PID" \
        --arg updated "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        --argjson complete "$complete" \
        --argjson pending "$pending" --argjson initialising "$initialising" \
        --argjson running "$running" --argjson stopped "$stopped" --argjson failed "$failed" \
        --argjson jobs "$jobsJson" \
        '{pid: $pid, updated: $updated, complete: $complete,
          counts: {pending: $pending, initialising: $initialising, running: $running,
                   stopped: $stopped, failed: $failed},
          jobs: $jobs}' > "$STATUS_FILE.tmp" && mv "$STATUS_FILE.tmp" "$STATUS_FILE"
}

# Backgrounded once, alongside the per-job process_job() subshells. Exits
# itself once complete — the plain `wait` in main() waits for this AND
# every process_job subshell, so the two converge naturally with no extra
# bookkeeping needed.
status_writer_loop() {
    while true; do
        write_status_summary
        [[ "$(jq -r '.complete // false' "$STATUS_FILE" 2>/dev/null)" == "true" ]] && break
        sleep "$POLL_INTERVAL_SECS"
    done
}

# Prefetch every UNIQUE model across the given job names, sequentially,
# before any job is submitted. Args: job names (already copied into place
# via resolve_job_config(), so get_job_config_setting can read them).
#
# Not a rare edge case: for a tuning comparison it's the COMMON case that
# every config in a batch shares the same base model with different
# parallelism/quant settings. Without this, N concurrently-submitted
# process_job() subshells would each independently notice the model is
# missing (ivllm-serve.sh's own inline auto-download check) and race to
# `hf download` the identical destination at the same time — not
# necessarily a hard failure, but wasted bandwidth/time at best and an
# unverified assumption about concurrent-write safety at worst. Doing it
# here, once per unique model, sequentially, removes the race entirely —
# this is deliberately NOT "hard fail if not downloaded" (that would be
# unfriendly when we could just fetch it), but a genuinely-undownloadable
# model (bad name, auth/gating issue) returns non-zero, which main() turns
# into aborting the whole comparison rather than letting every dependent
# job separately fail after sitting in the SLURM queue first.
#
# Uses $COMPARISON_DIR (global, set by main()) only for the download log's
# location — the dedup/download logic itself only depends on its arguments
# and the sourced utils.sh helpers, which is what keeps it unit-testable
# with a plain job-name list rather than needing a full main() invocation.
prefetch_unique_models() {
    local job model dlLog
    declare -A seenModels=()
    for job in "$@"; do
        model=$(get_job_config_setting "$job" ".model")
        [[ -n "${seenModels[$model]:-}" ]] && continue
        seenModels["$model"]=1

        if [[ -d "$(resolve_model_dir "$model")" ]]; then
            echo "[bench] model already cached: $model"
            continue
        fi

        echo "[bench] model not yet cached, downloading once for every config that needs it: $model"
        dlLog="${COMPARISON_DIR:-.}/model-download-$(echo "$model" | tr -c 'A-Za-z0-9._-' '_').log"
        if ! (source "$IVLLM_BIN/ivllm-get-model.sh" -m "$model" -l "$dlLog"); then
            # shellcheck disable=2031
            echo "[bench] ERROR: failed to download $model — see $dlLog — aborting, every job needing it would fail anyway" >&2
            return 1
        fi
    done
}

# ── main(): everything with a side effect. Only runs when this file is
# EXECUTED directly (see the source-guard at the bottom) — never when
# sourced for testing. ──
main() {
    COMPARISON_DIR=""
    local OPTIND opt
    while getopts "c:h" opt; do
        case $opt in
            c) COMPARISON_DIR="$OPTARG" ;;
            h) usage ;;
            \?) usage ;;
        esac
    done

    [[ -z "$COMPARISON_DIR" ]] && { echo "[bench] ERROR: -c <comparison_directory> is required" >&2; usage; }
    [[ -d "$COMPARISON_DIR" ]] || { echo "[bench] ERROR: not a directory: $COMPARISON_DIR" >&2; exit 1; }
    [[ -z "$REAL_PROJECTDIR" ]] && { echo "[bench] ERROR: \$PROJECTDIR/\$IVLLM_PROJECTDIR not set — point it at the real project dir first" >&2; exit 1; }

    # Benchmarking project dir for each comparison.
    BENCH_PROJECTDIR="$COMPARISON_DIR/benchmark"

    echo "[bench] real project dir:      $REAL_PROJECTDIR"
    echo "[bench] benchmark shadow dir:  $BENCH_PROJECTDIR"
    echo "[bench] comparison directory:  $COMPARISON_DIR"

    # ── Step 2: shadow project directory + symlinks (idempotent) ──
    mkdir -p "$BENCH_PROJECTDIR/engine"
    local shared target link
    for shared in vllm nvhpc rdma; do
        target="$REAL_PROJECTDIR/engine/$shared"
        link="$BENCH_PROJECTDIR/engine/$shared"
        if [[ -L "$link" ]]; then
            :  # already symlinked — leave it
        elif [[ -e "$link" ]]; then
            echo "[bench] ERROR: $link exists and is not a symlink — refusing to touch it" >&2
            exit 1
        else
            ln -s "$target" "$link"
            echo "[bench] symlinked engine/$shared -> $target"
        fi
    done
    if [[ -L "$BENCH_PROJECTDIR/model" ]]; then
        :
    elif [[ -e "$BENCH_PROJECTDIR/model" ]]; then
        echo "[bench] ERROR: $BENCH_PROJECTDIR/model exists and is not a symlink — refusing to touch it" >&2
        exit 1
    else
        ln -s "$REAL_PROJECTDIR/model" "$BENCH_PROJECTDIR/model"
        echo "[bench] symlinked model -> $REAL_PROJECTDIR/model"
    fi
    mkdir -p "$BENCH_PROJECTDIR/engine/jobs" "$BENCH_PROJECTDIR/engine/diagnostics"
    # rerunning a benchmarking job will add into existing results in diagostics

    # Everything below runs against the SHADOW project dir. utils.sh is
    # sourced purely to reuse its existing path/status helpers instead of
    # re-deriving path conventions by hand here (see the
    # capture_job_diagnostics() change noted at the top — the one
    # intentional exception to "unmodified").
    # shellcheck disable=2031
    export IVLLM_PROJECTDIR="$BENCH_PROJECTDIR"
    export PROJECTDIR="$BENCH_PROJECTDIR"
    # shellcheck disable=SC1091
    source "$IVLLM_BIN/lib/utils.sh"

    # Read by write_status_summary()/status_writer_loop() (defined above,
    # before COMPARISON_DIR was known) — global, not local, deliberately.
    STATUS_FILE="$COMPARISON_DIR/benchmarking_status.json"
    ORCHESTRATOR_PID=$$

    # ── Steps 1-4/8: discover configs, prefetch models, start status reporting, fan out, wait, collect ──

    declare -a jobNames=()
    local f job
    shopt -s nullglob
    for f in "$COMPARISON_DIR"/*.yaml "$COMPARISON_DIR"/*.yml; do
        job=$(basename "$f")
        job="${job%.*}"
        jobNames+=("$job")
        echo "[bench] discovered config: $f -> job \"$job\""
        # Copied into place now (not inside process_job) so every job's
        # .model can be read via the normal get_job_config_setting helper
        # below, before anything is submitted.
        cp "$f" "$(resolve_job_config "$job")"
    done
    shopt -u nullglob

    if [[ ${#jobNames[@]} -eq 0 ]]; then
        echo "[bench] ERROR: no *.yaml/*.yml files found in $COMPARISON_DIR" >&2
        exit 1
    fi

    # See prefetch_unique_models()'s own doc comment (defined above main())
    # for why this matters more than it looks like it should.
    prefetch_unique_models "${jobNames[@]}" || exit 1

    echo "[bench] writing initial $STATUS_FILE"
    write_status_summary
    status_writer_loop &

    echo "[bench] ${#jobNames[@]} job(s) running in parallel: ${jobNames[*]}"
    for job in "${jobNames[@]}"; do
        process_job "$job" &
    done
    wait

    # Guarantee the final state is captured immediately rather than relying
    # on status_writer_loop's own last cycle (which the `wait` above
    # already waited for anyway — this is just belt-and-braces).
    write_status_summary

    local resultsDir="$COMPARISON_DIR/results"
    mkdir -p "$resultsDir"
    cp -r "$BENCH_PROJECTDIR/engine/diagnostics/." "$resultsDir/" 2>/dev/null || true

    # Piped through tee, not just printed — nothing is watching this
    # script's stdout live once it's launched detached (see the
    # fire-and-forget contract above), so the summary needs to actually
    # land in $resultsDir alongside the diagnostics it's describing, not
    # just scroll past in orchestrator.log.
    {
    local jobResultsDir runDir runName status reason benchFile reqPerSec outTokPerSec ttft
    echo
    echo "=== Comparison complete — results copied to $resultsDir ==="
    printf '%-20s %-16s %-10s %-22s %-10s %-10s %-10s %s\n' \
        "JOB" "RUN" "STATUS" "REASON" "REQ/S" "OUT TOK/S" "TTFT(ms)" "BENCH FILE"
    for job in "${jobNames[@]}"; do
        jobResultsDir="$resultsDir/$job"

        # Deliberately read from the ARCHIVED copy under $resultsDir, not
        # the live $IVLLM_PROJECTDIR/engine/jobs/ lockfile — this is what
        # actually gets shipped back once a future TypeScript
        # `compare --analyse` only rsyncs engine/diagnostics/, not
        # engine/jobs/, so the summary here should be self-sufficient from
        # the same tree a user/agent would actually receive.
        #
        # Every timestamped capture_job_diagnostics() folder for this job,
        # not just the newest — rerunning a comparison after tweaking a
        # config is the expected workflow, and comparing against what a
        # previous attempt scored is the actual point, so history is
        # printed rather than collapsed to one row. The RUN column (the
        # folder's own timestamp) is what disambiguates "which attempt was
        # this" as the config underneath a job name changes across reruns.
        while IFS= read -r runDir; do
            [[ -z "$runDir" ]] && continue
            runName=$(basename "$runDir")

            status="unknown"; reason=""
            if [[ -f "$runDir/status.json" ]]; then
                status=$(jq -r '.status // "unknown"' "$runDir/status.json")
                reason=$(jq -r '.reason // ""' "$runDir/status.json")
            fi

            benchFile=""; reqPerSec="-"; outTokPerSec="-"; ttft="-"
            if [[ -f "$runDir/bench.json" ]]; then
                benchFile="$runDir/bench.json"
                # Field names below are current as of vLLM's `vllm bench
                # serve --save-result` output at the time this was written
                # — check `jq keys` on an actual bench.json if these come
                # back as "?" on a different vLLM version.
                reqPerSec=$(jq -r '.request_throughput // "?"' "$benchFile")
                outTokPerSec=$(jq -r '.output_throughput // "?"' "$benchFile")
                ttft=$(jq -r '.mean_ttft_ms // "?"' "$benchFile")
            fi

            printf '%-20s %-16s %-10s %-22s %-10s %-10s %-10s %s\n' \
                "$job" "$runName" "$status" "$reason" "$reqPerSec" "$outTokPerSec" "$ttft" "${benchFile:-<no result>}"
        done < <(find "$jobResultsDir" -maxdepth 1 -mindepth 1 -type d 2>/dev/null | sort)
    done
    echo "=== No verdict computed — read the bench.json files to compare numbers yourself ==="
    } | tee "$resultsDir/summary.txt"
}

# Only run main() when this file is EXECUTED, not when it's sourced (e.g. by
# tests/bash/sandboxed/test-ivllm-bench.sh to unit test process_job(),
# write_status_summary(), and the model-prefetch loop directly).
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
