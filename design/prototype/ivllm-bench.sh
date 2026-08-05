#!/bin/bash
# design/prototype/ivllm-bench.sh — PROTOTYPE, not production (see AGENTS.md:
# scripts in design/ are instructional examples, rewrite before shipping).
#
# Draft of the ADR-118 (revised) "shadow project directory" benchmarking
# design, taken one step further per discussion: instead of adding an
# IVLLM_BENCH_MODE hook inside monitor_head() (a change to an existing
# production file), this version is a **standalone login-node orchestrator**
# that drives the existing, completely UNMODIFIED external CLI surface —
# ivllm-serve.sh, ivllm-cancel.sh, and utils.sh's read-only status/config
# helpers (sourced, never edited). Zero existing files touched. This trades
# a little bit of measurement purity (the `vllm bench serve` client runs on
# the login node, reaching the job's `computeHostname:serverPort` over
# Isambard's internal network — a real hop, not literal in-process
# `localhost` — though still nothing like the WAN/SSH-tunnel latency ADR-118
# explicitly rejected as "meaningless") for a design that is easy to reason
# about, easy to integrate into the TypeScript CLI later (it's just another
# external command the backend shells out to), and carries zero risk to any
# currently-working script.
#
# What this does, end to end, for `ivllm-bench.sh -c <comparison_directory>`:
#   1. Reads every `*.yaml`/`*.yml` file in <comparison_directory> — one
#      candidate vLLM config per file, job name = filename stem.
#   2. Creates a "benchmark" shadow project directory as a subfolder of the
#      REAL project directory (read from $PROJECTDIR/$IVLLM_PROJECTDIR,
#      which must already be set in the environment — same convention
#      utils.sh itself uses). Symlinks the expensive shared subdirectories
#      (engine/vllm, engine/nvhpc, engine/rdma, model) back to the real
#      project dir so nothing is re-downloaded or recompiled; engine/jobs
#      and engine/diagnostics are real, independent directories so
#      benchmark runs never collide with real job state.
#   3. For each config: wipes any stale prior benchmark state for that job
#      name, copies the config into place at the path utils.sh expects.
#   4. Submits all jobs concurrently via the real `ivllm-serve.sh -j <job>
#      -b` (non-interactive batch partition — already skips the interactive
#      reservation's 1-job limit, no new code needed for that).
#   5. Polls each job's lockfile (via utils.sh's own `is_status`/
#      `get_job_status_setting` — sourced, not reimplemented) until it
#      reaches "running", with a wall-clock giving-up point per job so a
#      stuck/hung job (see design/active-issues.md — this exists for a
#      reason) can't hang the whole orchestrator forever.
#   6. Once running: activates that job's own vLLM venv (same
#      resolve_vllm_version_dir()+source pattern ivllm-serve.sh itself
#      uses) and runs `vllm bench serve` against the job's
#      `computeHostname:serverPort`, saving JSON output into that job's
#      diagnostics directory.
#   7. Requests a graceful cancel (`ivllm-cancel.sh -j <job>`) — driving the
#      existing, unmodified shutdown/diagnostics-capture path.
#   8. Once every job has finished (benched-and-cancelled, or failed/timed
#      out on its own), copies the whole diagnostics tree back into
#      <comparison_directory>/results-<timestamp>/ and prints a plain
#      status table — no verdict, per ADR-118's original design intent.
#
# All 8 job-lifecycle steps run in parallel, one per config, as independent
# background subshells — this is the "trivially parallelisable across
# configs" property ADR-118 always wanted.
#
# Untested assumptions worth checking on a real run before trusting this:
#   - The login node can reach a compute node's hostname:port directly
#     (plain HTTP, no SSH tunnel) — this is what the SSH tunnel in
#     `IsambardBareMetalBackend.ts` exists to provide for a machine OUTSIDE
#     Isambard's network; it's assumed (not yet confirmed here) that the
#     login node itself doesn't need that tunnel to reach a sibling compute
#     node.
#   - Activating a vLLM venv (for the `vllm bench serve` CLIENT only — no
#     GPU/CUDA needed for that side) works fine on the login node's
#     architecture — should be true (same aarch64 family) but unconfirmed.
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

BENCH_PROJECTDIR="$REAL_PROJECTDIR/benchmark"

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

# Everything below runs against the SHADOW project dir. utils.sh is sourced
# (never edited) purely to reuse its existing, already-correct path/status
# helpers instead of re-deriving path conventions by hand here.
export IVLLM_PROJECTDIR="$BENCH_PROJECTDIR"
export PROJECTDIR="$BENCH_PROJECTDIR"
# shellcheck disable=SC1091
source "$IVLLM_BIN/lib/utils.sh"

# ── Steps 3-7: one job, start to finish. Runs as an independent background
# subshell per config — this is where the "trivially parallelisable across
# configs" property comes from. ──
process_job() {
    local job="$1" configFile="$2"
    local jobDir diagDir vllmVersion vllmVersionDir minVllmVersion model computeHostname serverPort
    local waited=0

    echo "[bench:$job] preparing"

    # Clean slate — benchmark jobs are disposable, don't inherit a stale
    # result from a previous comparison run under the same config name.
    rm -rf "${IVLLM_PROJECTDIR:?}/engine/jobs/${job:?}" "${IVLLM_PROJECTDIR:?}/engine/diagnostics/${job:?}"

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
    echo "[bench:$job] running on $computeHostname:$serverPort ($model) — starting vllm bench serve"

    # Reachability check first — fail fast with a clear message rather than
    # letting `vllm bench serve` fail confusingly deep inside its own client.
    if ! curl -sf --max-time 10 "http://$computeHostname:$serverPort/health" >/dev/null; then
        echo "[bench:$job] ERROR: cannot reach http://$computeHostname:$serverPort/health from the login node"
        echo "[bench:$job]        (untested assumption in this prototype — see header comment)"
        "$IVLLM_BIN/ivllm-cancel.sh" -j "$job" >>"$jobDir/bench-submit.log" 2>&1
        return 1
    fi

    minVllmVersion=$(get_job_config_setting "$job" ".min-vllm-version")
    vllmVersion=$(select_closest_version "$minVllmVersion")
    vllmVersionDir=$(resolve_vllm_version_dir "$vllmVersion")
    # shellcheck disable=SC1091
    source "$vllmVersionDir/bin/activate"

    diagDir=$(resolve_diagnostics_dir "$job")
    if vllm bench serve \
        --host "$computeHostname" \
        --port "$serverPort" \
        --model "$model" \
        --dataset-name random \
        --save-result \
        --result-dir "$diagDir" \
        --result-filename "bench.json" \
        > "$diagDir/bench.log" 2>&1
    then
        echo "[bench:$job] bench complete — $diagDir/bench.json"
    else
        echo "[bench:$job] vllm bench serve FAILED — see $diagDir/bench.log"
    fi

    echo "[bench:$job] requesting graceful cancel"
    "$IVLLM_BIN/ivllm-cancel.sh" -j "$job" >>"$jobDir/bench-submit.log" 2>&1
    cp "$configFile" "$diagDir/" 2>/dev/null || true
}

# ── Steps 1-4/8: discover configs, fan out, wait, collect ──
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

resultsDir="$COMPARISON_DIR/results-$(date +%Y%m%d_%H%M%S)"
mkdir -p "$resultsDir"
cp -r "$BENCH_PROJECTDIR/engine/diagnostics/." "$resultsDir/" 2>/dev/null || true

echo
echo "=== Comparison complete — results copied to $resultsDir ==="
printf '%-20s %-12s %s\n' "JOB" "STATUS" "BENCH RESULT"
for job in "${jobNames[@]}"; do
    status=$(get_job_status_setting "$job" ".status" 2>/dev/null || echo "unknown")
    benchFile=$(find "$resultsDir/$job" -name "bench.json" 2>/dev/null | head -n1)
    printf '%-20s %-12s %s\n' "$job" "$status" "${benchFile:-<no result>}"
done
echo "=== No verdict computed — read the bench.json files to compare numbers yourself ==="
