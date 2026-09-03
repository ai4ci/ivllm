#!/bin/bash
# design/prototype/slingshot-tp-reprex.sh — PROTOTYPE, not production (see
# AGENTS.md: scripts in design/ are instructional examples, rewrite before
# shipping).
#
# Question this answers: is the GLM-5.2 hang (design/active-issues.md — all
# 8 TP ranks eventually frozen at gpu_model_runner.py's transfer_event
# synchronize(), traced via nccl-debug.log to NCCL electing exactly one rank
# per node, local slot 3, as the sole inter-node Slingshot/CXI network
# bridge for the model's AllGather-based MoE/EP dispatch, with that rank
# pair reproducibly falling behind and never catching back up) a Slingshot
# network problem, or something specific to vLLM/Ray's own process/executor
# machinery?
#
# Approach: request the IDENTICAL resource shape and environment vLLM's
# real 2-node/8-GPU deployment uses (same node count, same GPUs/node, same
# --cpus-per-gpu as the fixed CPU-affinity bug, same module loads /
# NCCL+libfabric tuning via common-env.sh + vllm-env.sh, same installed
# vLLM venv's torch/nccl/aws-ofi-nccl versions) — but run nothing except a
# plain torch.distributed NCCL AllGather loop (slingshot_tp_reprex.py),
# sized to match the exact collective observed in the real hangs
# (count=19360, bf16). No model weights, no Ray, no vLLM. If the same
# rank-3/rank-7 divergence reproduces here, that's decisive: the problem is
# in NCCL/Slingshot for this specific 8-rank/2-node topology, not in vLLM.
# If it *doesn't* reproduce after a real stress run, that's equally
# decisive the other way, and re-opens vLLM/Ray-executor-level causes.
#
# Because this does no model loading or warmup, a full run (tens of
# thousands of iterations of a tiny AllGather) takes on the order of a
# minute if healthy — versus 10-15+ minutes to reach the equivalent amount
# of real collective traffic via a full vLLM startup. This is the fast
# iteration loop for testing NCCL/FI_* tuning knobs going forward: change
# an env var, resubmit, get an answer in about a minute instead of paying
# for a full model load every time.
#
# Usage:
#   sbatch design/prototype/slingshot-tp-reprex.sh [vllm-venv-dir]
#
# To sweep a specific NCCL/libfabric variable without editing this file,
# export it before submitting — sbatch's default --export=ALL forwards your
# shell's environment into the job, and vllm-env.sh's own exports only take
# effect if the variable isn't already set:
#   NCCL_NET_GDR_LEVEL=SYS sbatch design/prototype/slingshot-tp-reprex.sh
#   REPREX_ITERS=100000 REPREX_ELEMS=4096 sbatch design/prototype/slingshot-tp-reprex.sh
#
# Expected PASS output (near the end of the job's --output log): all 8
# ranks' final "DONE ... total=Xs" lines appear within a few seconds of each
# other, and rank 0's per-rank summary table shows comparable mean/p99/max
# latencies across all 8 ranks — no rank's numbers wildly higher than its
# neighbors, and no "STALL" lines anywhere in the log.
#
# Expected FAIL output (reproducing the real hang's signature): one or more
# "⚠️ STALL at iteration ..." lines from a rank in local slot 3 (or its
# node-1 counterpart), that rank's heartbeats falling behind or stopping
# entirely while its 3 same-node neighbors' heartbeats keep advancing, and/or
# the job never reaching "clean exit" for the affected rank(s) before its
# own SLURM time limit or --gpus-per-task's srun step is killed.
#
#SBATCH --job-name=slingshot-tp-reprex
#SBATCH --nodes=2
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-gpu=72
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --ntasks-per-node=4
#SBATCH --reservation=interactive
#SBATCH --partition=interactive
#SBATCH --time=00:15:00
#SBATCH --output=%x-%j.log
#SBATCH --error=%x-%j.log

set -uo pipefail

ENGINE_LIB="$HOME/.local/bin/lib"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

resolve_venv() {
    # Args: $1 - optional explicit venv dir (containing bin/activate).
    # Falls back to the highest-versioned venv under $PROJECTDIR/engine/vllm
    # — same convention as ray-4node-smoketest.sh, so any vLLM install
    # already set up for real jobs works here with no extra steps.
    local explicit="${1:-}"
    if [[ -n "$explicit" && -f "$explicit/bin/activate" ]]; then
        echo "$explicit"
        return 0
    fi
    local base="${PROJECTDIR:-}/engine/vllm"
    if [[ ! -d "$base" ]]; then
        echo "ERROR: no venv given and \$PROJECTDIR/engine/vllm ($base) doesn't exist." >&2
        echo "Pass a venv path explicitly: sbatch $0 /path/to/vllm/0.26.0" >&2
        return 1
    fi
    find "$base" -mindepth 1 -maxdepth 1 -type d -print0 |
        xargs -0 -r -n1 basename | sort -V | tail -n1 |
        xargs -I{} echo "$base/{}"
}

main() {
    local venv_dir
    venv_dir=$(resolve_venv "${1:-}") || exit 1
    echo "[reprex] using venv: $venv_dir"

    # Same NCCL/libfabric/module setup real vLLM jobs get — this is the
    # whole point: byte-identical stack, minus vLLM/Ray, so a result here
    # is directly attributable to NCCL/Slingshot rather than confounded by
    # a different library version or missing tuning flag.
    # shellcheck disable=SC1091
    source "$ENGINE_LIB/common-env.sh"
    # shellcheck disable=SC1091
    source "$ENGINE_LIB/vllm-env.sh"
    # shellcheck disable=SC1091
    source "$venv_dir/bin/activate"

    python3 -c "import torch, torch.distributed" || {
        echo "ERROR: torch not importable after activating $venv_dir" >&2
        exit 1
    }

    # Full topology/channel-election visibility, written per-node — same
    # subsystem set as ivllm-debug-level:4's set_debugging_env(), so the
    # resulting log is directly diffable against a real job's
    # debug/nccl-debug.log.
    export NCCL_DEBUG=INFO
    export NCCL_DEBUG_SUBSYS=INIT,BOOTSTRAP,ENV,GRAPH,COLL,NET
    export NCCL_DEBUG_FILE="$SCRIPT_DIR/slingshot-tp-reprex-nccl-${SLURM_JOB_ID}-node%h.log"

    local head_node head_ip
    head_node=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
    head_ip=$(dig +short "$head_node")
    echo "[reprex] head node: $head_node ($head_ip)"

    export MASTER_ADDR="$head_ip"
    export MASTER_PORT="${MASTER_PORT:-29511}"  # arbitrary, unused by any real vLLM job

    echo "[reprex] launching 8 ranks (2 nodes x 4 GPUs) — plain torch.distributed NCCL, no Ray/vLLM"
    srun --export=ALL --gpus-per-task=1 bash -c '
        export RANK="$SLURM_PROCID"
        export WORLD_SIZE="$SLURM_NTASKS"
        export LOCAL_RANK="$SLURM_LOCALID"
        exec python3 -u "'"$SCRIPT_DIR"'/slingshot_tp_reprex.py"
    '
    local rc=$?
    echo "[reprex] srun exit code: $rc"
    return "$rc"
}

main "$@"
