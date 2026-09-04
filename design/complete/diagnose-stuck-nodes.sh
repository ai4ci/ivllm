#!/bin/bash
# design/prototype/diagnose-stuck-nodes.sh — collect SLURM diagnostics for a
# running job, to hand to Isambard admins when some nodes in a multi-node
# allocation never come up (symptom: srun keeps printing "Job step creation
# temporarily disabled, retrying (Requested nodes are busy)" for specific
# nodes, indefinitely, while other nodes in the same allocation work fine).
#
# Standalone — no ivllm/vLLM environment required. Run directly on a login
# node once the job is already running and you've noticed the symptom.
#
# What it does, per node in the job's allocation:
#   - scontrol show node          (State, CPULoad, AllocMem/RealMemory, Gres)
#   - a bare `srun --overlap ps auxf`     (any leftover/orphaned process?)
#   - a bare `srun --overlap nvidia-smi`  (are the GPUs actually free?)
#   - a GPU/mem-shaped `srun --immediate` probe, sized from that node's own
#     `scontrol` Gres/RealMemory — reproduces the exact "resource-bearing
#     step gets rejected, resource-free step succeeds" split that actually
#     diagnosed this the last time it happened, without hardcoding any
#     specific job's GPU/mem numbers.
# Plus job-level: scontrol show job, squeue -s (which nodes have a step at
# all — the real tell: stuck nodes have NONE), sacct history, reservation
# info (if any).
#
# Every srun call is wrapped in --immediate=$IMMEDIATE_SECS so this script
# can never itself get stuck retrying — a probe that can't get a step within
# that window is recorded as a failure and the script moves on.
#
# Usage: diagnose-stuck-nodes.sh <slurm-job-name>
#   <slurm-job-name>  e.g. "glm52" — matched against squeue's own job name,
#                     for the current user, assumed to already be running.
#
# Output: a timestamped directory under $HOME/slurm-diag/, tarred at the end
# — the printed tarball path is what to attach to a support ticket.

IMMEDIATE_SECS=20

usage() {
    echo "Usage: $0 <slurm-job-name>" >&2
    echo "  <slurm-job-name>  e.g. \"glm52\" — matched via squeue --me --all -n <name>," >&2
    echo "                    assumed to already be running." >&2
    exit 1
}

# Args: $1 - node name. Writes ps-auxf.txt, nvidia-smi.txt, and
# gpu-probe.txt into $ndir. Never hangs (every srun call is --immediate).
# Usage: collect_node_diagnostics "$node"
collect_node_diagnostics() {
    local node="$1"
    local ndir="$OUTDIR/$node"
    mkdir -p "$ndir"

    echo "  [$node] scontrol show node"
    scontrol show node -o "$node" >"$ndir/scontrol-node.txt" 2>&1

    echo "  [$node] bare ps auxf (any leftover process? kernel threads filtered)"
    # Drop pure kernel-thread lines ("[kthreadd]", "[kworker/0:1]", ...) but
    # keep anything else with a bracket in it — notably "slurmstepd: [<step>]",
    # which is exactly how you spot which steps are alive on this node.
    srun --overlap --jobid="$JOBID" --nodelist="$node" --immediate="$IMMEDIATE_SECS" \
        ps auxf 2>&1 | awk '/\[/ && !/slurmstepd/ {next} {print}' >"$ndir/ps-auxf.txt"
    echo "exit=${PIPESTATUS[0]}" >>"$ndir/ps-auxf.txt"

    echo "  [$node] nvidia-smi (are the GPUs actually free?)"
    srun --overlap --jobid="$JOBID" --nodelist="$node" --immediate="$IMMEDIATE_SECS" \
        nvidia-smi >"$ndir/nvidia-smi.txt" 2>&1
    echo "exit=$?" >>"$ndir/nvidia-smi.txt"

    # Size the reproduction probe from this node's own advertised resources —
    # not from any specific job's numbers — so it generalises to other jobs.
    local gres mem
    gres=$(grep -oP '(?:^| )Gres=gpu:\K[0-9]+' "$ndir/scontrol-node.txt" | head -n1 || true)
    mem=$(grep -oP '(?:^| )RealMemory=\K[0-9]+' "$ndir/scontrol-node.txt" | head -n1 || true)

    local probe_args=(--overlap --jobid="$JOBID" --nodelist="$node"
        --immediate="$IMMEDIATE_SECS" --nodes=1 --ntasks-per-node=1)
    [[ -n "$gres" ]] && probe_args+=(--gpus="$gres")
    [[ -n "$mem" ]] && probe_args+=(--mem="${mem}M")

    echo "  [$node] GPU/mem-shaped step probe (gpus=${gres:-none} mem=${mem:-none}M)"
    {
        echo "probe args: ${probe_args[*]}"
        srun "${probe_args[@]}" true
        echo "exit=$?"
    } >"$ndir/gpu-probe.txt" 2>&1
}

main() {
    local job_name="${1:-}"
    [[ -z "$job_name" ]] && usage

    # --all is required on every squeue call here, not just this one: some
    # sites (e.g. Isambard's "interactive" reservation/partition) configure
    # certain partitions as Hidden=YES, and squeue silently omits anything
    # in a hidden partition from its default output — even when scoped to a
    # specific known JOBID (`squeue -j` below) — unless --all overrides it.
    # scontrol/sacct are unaffected; this is a squeue-specific behaviour.
    JOBID=$(squeue --me --all -h -n "$job_name" -o "%i" | head -n1)
    if [[ -z "$JOBID" ]]; then
        echo "ERROR: no running job named '$job_name' found for $(whoami)" >&2
        exit 1
    fi
    local match_count
    match_count=$(squeue --me --all -h -n "$job_name" -o "%i" | wc -l)
    if (( match_count > 1 )); then
        echo "WARNING: $match_count jobs named '$job_name' found — using the first ($JOBID)" >&2
    fi

    local timestamp
    timestamp=$(date +%Y%m%d_%H%M%S)
    OUTDIR="$HOME/slurm-diag/${job_name}_${JOBID}_${timestamp}"
    mkdir -p "$OUTDIR"
    echo "=== job '$job_name' (id $JOBID) — diagnostics -> $OUTDIR ==="

    echo "--- job-level info ---"
    scontrol show job -o "$JOBID" >"$OUTDIR/scontrol-job.txt" 2>&1
    squeue -j "$JOBID" -s --all >"$OUTDIR/steps.txt" 2>&1
    sacct -j "$JOBID" --format=JobID,NodeList,State,ExitCode,Start,End -X \
        >"$OUTDIR/sacct.txt" 2>&1

    # Note: `scontrol show job -o` packs many "Key=value" fields onto one
    # line, and some field names are suffixes of others (e.g. "ReqNodeList="
    # ends in "NodeList="). An unanchored `grep -oP 'NodeList=\K\S+'` matches
    # inside "ReqNodeList=" too, and since that field is usually "(null)"
    # (no explicit --nodelist at submission) and appears *before* the real
    # "NodeList=" on the line, `head -n1` would silently pick up the wrong
    # one. Anchoring on "start-of-line or preceding space" avoids that for
    # every field pulled from this file.
    local reservation
    reservation=$(grep -oP '(?:^| )Reservation=\K\S+' "$OUTDIR/scontrol-job.txt" | head -n1 || true)
    if [[ -n "$reservation" && "$reservation" != "(null)" ]]; then
        scontrol show reservation "$reservation" >"$OUTDIR/scontrol-reservation.txt" 2>&1
    fi

    local nodelist
    nodelist=$(grep -oP '(?:^| )NodeList=\K\S+' "$OUTDIR/scontrol-job.txt" | head -n1)
    if [[ -z "$nodelist" || "$nodelist" == "(null)" ]]; then
        local job_state
        job_state=$(grep -oP '(?:^| )JobState=\K\S+' "$OUTDIR/scontrol-job.txt" | head -n1 || echo "unknown")
        echo "ERROR: job $JOBID has no nodes allocated yet (JobState=$job_state)." >&2
        echo "This tool is for a job that's already running with some nodes stuck —" >&2
        echo "wait for the allocation to complete, or inspect $OUTDIR/scontrol-job.txt directly." >&2
        exit 1
    fi
    local nodes=()
    mapfile -t nodes < <(scontrol show hostnames "$nodelist")
    echo "nodes in allocation: ${nodes[*]}"
    printf '%s\n' "${nodes[@]}" >"$OUTDIR/nodes.txt"

    # Nodes with at least one step registered against them right now.
    local nodes_with_steps
    nodes_with_steps=$(awk 'NR>1 {print $NF}' "$OUTDIR/steps.txt" |
        xargs -r -n1 scontrol show hostnames 2>/dev/null | sort -u)

    echo "--- per-node diagnostics ---"
    for node in "${nodes[@]}"; do
        collect_node_diagnostics "$node"
    done

    {
        echo "SLURM stuck-node diagnostics"
        echo "job: $job_name (id $JOBID)"
        echo "collected: $(date -Iseconds) by $(whoami) on $(hostname)"
        echo
        echo "Nodes in allocation and whether SLURM has ever created a step on them:"
        for node in "${nodes[@]}"; do
            if grep -qx "$node" <<<"$nodes_with_steps"; then
                echo "  $node : HAS a step registered"
            else
                echo "  $node : ** NO step registered — this is the stuck-node signature **"
            fi
        done
        echo
        echo "Per-node probe results:"
        for node in "${nodes[@]}"; do
            local probe_exit
            probe_exit=$(grep -oP '^exit=\K[0-9]+' "$OUTDIR/$node/gpu-probe.txt" | tail -n1)
            local ps_lines
            ps_lines=$(grep -vc '^exit=' "$OUTDIR/$node/ps-auxf.txt")
            echo "  $node : GPU-probe exit=${probe_exit:-?} (0=succeeded)  ps-auxf non-daemon-line-count=$ps_lines"
        done
        echo
        echo "Raw detail in: $OUTDIR/<node>/{scontrol-node,ps-auxf,nvidia-smi,gpu-probe}.txt"
        echo "Job-level detail in: $OUTDIR/{scontrol-job,steps,sacct}.txt"
    } | tee "$OUTDIR/summary.txt"

    local tarball="${OUTDIR}.tar.gz"
    tar czf "$tarball" -C "$(dirname "$OUTDIR")" "$(basename "$OUTDIR")"
    echo
    echo "=== done — attach this to the support ticket: $tarball ==="
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then main "$@"; fi
