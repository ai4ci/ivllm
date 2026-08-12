#!/bin/bash
# design/prototype/ray-4node-smoketest.sh — minimal N-node Ray cluster
# smoke test, to check whether the "3rd+ node's srun step never gets
# created" symptom (see design/active-issues.md / logs/slurm-diag-*) is
# specific to ivllm/vLLM's real job scripts, or a general property of any
# job on this cluster that launches this many overlapping `srun --overlap`
# steps under one sbatch allocation.
#
# Deliberately NOT sourced from src/engine/lib/*.sh and has no ivllm job
# directory / vllm.yaml dependency — a from-scratch minimal reproduction
# using the same resource shape ivllm's real dispatch (slurm-ray-vllm-serve.sh)
# requests: --gpus-per-node=4 --cpus-per-gpu=64 --mem=0 --exclusive, one
# `srun --overlap` per node (head + one per worker), fired in a loop with a
# short stagger between launches — but running a trivial `ray start` +
# a one-line Python payload instead of vLLM.
#
# Usage:
#   sbatch design/prototype/ray-4node-smoketest.sh
#
# To vary node count without editing this file, pass --nodes on the sbatch
# command line — sbatch CLI flags override the #SBATCH pragmas below:
#   sbatch --nodes=2 design/prototype/ray-4node-smoketest.sh
#
# Optional: pass a specific vLLM venv to reuse its installed `ray` +
# `python3`, as the first script argument (defaults to auto-detecting the
# highest-versioned venv under $PROJECTDIR/engine/vllm/<version>/):
#   sbatch design/prototype/ray-4node-smoketest.sh /projects/<proj>/engine/vllm/0.26.0
#
# Expected PASS output (near the end of the job's --output log):
#   [smoketest] ray.nodes() alive count: 4 / 4 expected -> PASS
# If it prints fewer than expected, that reproduces the symptom with no
# vLLM/ray involved beyond `ray start` itself — pointing at SLURM/srun
# step-creation, not anything in ivllm's own scripts.
#
#SBATCH --job-name=ray4smoke
#SBATCH --nodes=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-gpu=64
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --ntasks-per-node=1
#SBATCH --reservation=interactive
#SBATCH --time=00:10:00
#SBATCH --output=%x-%j.log
#SBATCH --error=%x-%j.log

set -uo pipefail

RAY_PORT=6380  # deliberately not vLLM's usual 6379, to avoid colliding with a real job
LAUNCH_STAGGER_SECS=2
CLUSTER_SETTLE_SECS=30
NGPUS_PER_NODE=${SBATCH_GPUS_PER_NODE:-4}
NMEM_PER_NODE=0

resolve_venv() {
    # Args: $1 - optional explicit venv dir (containing bin/activate).
    # Falls back to the highest-versioned venv under $PROJECTDIR/engine/vllm.
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
    echo "[smoketest] using venv: $venv_dir"
    # shellcheck disable=SC1091
    source "$venv_dir/bin/activate"
    command -v ray >/dev/null || { echo "ERROR: 'ray' not found after activating $venv_dir" >&2; exit 1; }

    local head_node head_ip
    head_node=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
    head_ip=$(dig +short "$head_node")
    echo "[smoketest] head node: $head_node ($head_ip)"

    local srun_pids=()

    # A long-lived background process alongside the srun steps, to mirror
    # ivllm's real monitor_head() coexisting with the srun launches — in
    # case a competing non-srun process on the batch step matters.
    ( while true; do sleep 5; done ) & local monitor_pid=$!
    echo "[smoketest] background monitor pid: $monitor_pid"

    echo "[smoketest] launching ray head on $head_node"
    srun --overlap --nodelist="$head_node" --nodes=1 \
        --gpus="$NGPUS_PER_NODE" --mem="$NMEM_PER_NODE" --cpus-per-gpu=64 \
        --ntasks-per-node=1 --export=ALL \
        ray start --head --port="$RAY_PORT" --node-ip-address="$head_ip" --block &
    srun_pids+=("$!")
    echo "[smoketest] head srun pid: $!"

    sleep 5  # give the head a moment before workers try to join

    local worker_nodes count=1
    worker_nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | tail -n+2)
    for worker in $worker_nodes; do
        local worker_ip
        worker_ip=$(dig +short "$worker")
        echo "[smoketest] launching ray worker $count on $worker ($worker_ip)"
        srun --overlap --nodelist="$worker" --nodes=1 \
            --gpus="$NGPUS_PER_NODE" --mem="$NMEM_PER_NODE" --cpus-per-gpu=64 \
            --ntasks-per-node=1 --export=ALL \
            ray start --address="${head_ip}:${RAY_PORT}" --node-ip-address="$worker_ip" --block &
        srun_pids+=("$!")
        echo "[smoketest] worker $count srun pid: $!"
        ((count++))
        sleep "$LAUNCH_STAGGER_SECS"
    done

    echo "[smoketest] all srun launches issued — which nodes actually got a step:"
    squeue -j "$SLURM_JOB_ID" -s --all

    echo "[smoketest] waiting ${CLUSTER_SETTLE_SECS}s for the ray cluster to settle..."
    sleep "$CLUSTER_SETTLE_SECS"

    echo "[smoketest] re-checking steps after settle time:"
    squeue -j "$SLURM_JOB_ID" -s --all

    echo "[smoketest] querying cluster from the head node"
    srun --overlap --nodelist="$head_node" --nodes=1 --ntasks-per-node=1 --export=ALL \
        python3 - "$SLURM_JOB_NUM_NODES" <<'PYEOF'
import sys
import ray

expected = int(sys.argv[1])
ray.init(address="auto")
nodes = ray.nodes()
alive = [n for n in nodes if n.get("Alive")]
print(f"[smoketest] ray.nodes() total entries: {len(nodes)}")
for n in nodes:
    print(f"  {n.get('NodeManagerAddress')}: alive={n.get('Alive')}")
print(f"[smoketest] ray.nodes() alive count: {len(alive)} / {expected} expected -> "
      f"{'PASS' if len(alive) == expected else 'FAIL'}")
PYEOF

    echo "[smoketest] tearing down"
    for pid in "${srun_pids[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    kill "$monitor_pid" 2>/dev/null || true
    scancel "$SLURM_JOB_ID"
}

main "$@"
