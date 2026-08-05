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
#      previous ones in diagnostics rather than overwriting history.
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

# TODO: This needs to be possible to run in fire and forget mode. Likely use is
# setup a directory put some files in it. Call the benchmarking, go away, come
# back much later. Retrieve results. The typescript CLI will not be waiting for
# this to finish. It woudl be nice if this could be a slurm job but realistically
# since it is launching slurm jobs that is unlikely to be simple.
#
# retrieving results is likely just look in the results directory. more
# interesting is the check status of running benchmarking job. Is some level of
# 8 jobs pending; 3 initialising; 1 running; 0 failed; 1 stopped (likely complete);
# feedback possible through a benchmarking_status.json (could be in the $resultDir)

set -uo pipefail

IVLLM_BIN="${IVLLM_BIN:-$HOME/.local/bin}"
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

COMPARISON_DIR=""
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

# Everything below runs against the SHADOW project dir. utils.sh is sourced
# purely to reuse its existing path/status helpers instead of re-deriving
# path conventions by hand here (see the capture_job_diagnostics() change
# noted at the top — the one intentional exception to "unmodified").
export IVLLM_PROJECTDIR="$BENCH_PROJECTDIR"
export PROJECTDIR="$BENCH_PROJECTDIR"
# shellcheck disable=SC1091
source "$IVLLM_BIN/lib/utils.sh"

# ── Steps 3-7: one job, start to finish. Runs as an independent background
# subshell per config — this is where the "trivially parallelisable across
# configs" property comes from. ──
process_job() {
    local job="$1" configFile="$2"
    local jobDir vllmVersion vllmVersionDir minVllmVersion model computeHostname serverPort slurmJobId
    local waited=0

    echo "[bench:$job] preparing"

    jobDir=$(resolve_job_dir "$job")
    cp "$configFile" "$(resolve_job_config "$job")"

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
            echo "[bench:$job] WARNING: job did not reach a terminal state within 300s of cancel request — archiving whatever state exists now"
            # TODO: scancel the slurm job by id for this job if its not going down cleanly.
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

# ── Steps 1-4/8: discover configs, fan out, wait, collect ──

# TODO: For consideration: hard fail if any of the models are not yet downloaded.
# why - not the benchmarkers job to download models? why not - no reason really?

declare -a jobNames=()
shopt -s nullglob
for f in "$COMPARISON_DIR"/*.yaml "$COMPARISON_DIR"/*.yml; do
    job=$(basename "$f")
    job="${job%.*}"
    jobNames+=("$job")
    echo "[bench] discovered config: $f -> job \"$job\""
    process_job "$job" "$f" &
done
shopt -u nullglob

if [[ ${#jobNames[@]} -eq 0 ]]; then
    echo "[bench] ERROR: no *.yaml/*.yml files found in $COMPARISON_DIR" >&2
    exit 1
fi

echo "[bench] ${#jobNames[@]} job(s) running in parallel: ${jobNames[*]}"
wait

resultsDir="$COMPARISON_DIR/results"
mkdir -p "$resultsDir"
cp -r "$BENCH_PROJECTDIR/engine/diagnostics/." "$resultsDir/" 2>/dev/null || true

# TODO: capture this summary output to a file in $resultsDir
echo
echo "=== Comparison complete — results copied to $resultsDir ==="
printf '%-20s %-10s %-22s %-10s %-10s %-10s %s\n' \
    "JOB" "STATUS" "REASON" "REQ/S" "OUT TOK/S" "TTFT(ms)" "BENCH FILE"
for job in "${jobNames[@]}"; do
    jobResultsDir="$resultsDir/$job"

    # Deliberately read from the ARCHIVED copy under $resultsDir, not the
    # live $IVLLM_PROJECTDIR/engine/jobs/ lockfile — this is what actually
    # gets shipped back once a future TypeScript `compare --analyse` only
    # rsyncs engine/diagnostics/, not engine/jobs/, so the summary here
    # should be self-sufficient from the same tree a user/agent would
    # actually receive. One timestamped folder per job run (capture_job_
    # diagnostics() sweeps the whole job dir — status.json, bench.json,
    # debug/pyspy dumps, everything — into it in one shot), named with a
    # timestamp it generates internally, so find the newest one rather than
    # trying to predict the exact name.

    # TODO: The summary should include all previous runs
    # why? - tweaking things and rerunning comparison is what is going to happen
    # comparing newest run with other runs is naturally going to be what we want
    # and changing configuration is sometimes going to make things worse, so we
    # want to compare what we had.
    # why not? - semantics of what one "job" is changes over time. need to have a
    # indication of the run - via the timestamp of the diagnostics directory.

    latestRunDir=$(find "$jobResultsDir" -maxdepth 1 -mindepth 1 -type d 2>/dev/null | sort | tail -n1)

    status="unknown"; reason=""
    if [[ -n "$latestRunDir" && -f "$latestRunDir/status.json" ]]; then
        status=$(jq -r '.status // "unknown"' "$latestRunDir/status.json")
        reason=$(jq -r '.reason // ""' "$latestRunDir/status.json")
    fi

    benchFile=""; reqPerSec="-"; outTokPerSec="-"; ttft="-"
    if [[ -n "$latestRunDir" && -f "$latestRunDir/bench.json" ]]; then
        benchFile="$latestRunDir/bench.json"
        # Field names below are current as of vLLM's `vllm bench serve
        # --save-result` output at the time this was written — check
        # `jq keys` on an actual bench.json if these come back as "?" on a
        # different vLLM version.
        reqPerSec=$(jq -r '.request_throughput // "?"' "$benchFile")
        outTokPerSec=$(jq -r '.output_throughput // "?"' "$benchFile")
        ttft=$(jq -r '.mean_ttft_ms // "?"' "$benchFile")
    fi

    printf '%-20s %-10s %-22s %-10s %-10s %-10s %s\n' \
        "$job" "$status" "$reason" "$reqPerSec" "$outTokPerSec" "$ttft" "${benchFile:-<no result>}"
done
echo "=== No verdict computed — read the bench.json files to compare numbers yourself ==="

# TODO: more for the typescript client $resultDir is what we want to rsync back
# to the local machine to analyse results.
