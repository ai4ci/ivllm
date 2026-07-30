#!/bin/bash
# slurm-vllm-serve.sh — Multi-node vLLM job launcher.
#
# Submitted by ivllm-serve.sh via sbatch. Resolves environment,
# starts vLLM with proper NCCL/Ray configuration, and runs the
# monitor triad (startup, head, worker).
#SBATCH --export=ALL
# asks SLURM to send the SIGUSR1 signal 120 seconds before end of the time limit
#SBATCH --signal=B:SIGUSR1@120

# shellcheck disable=SC2155,SC1091
# N.b. set the log in the wrapper script (sbatch --output and --error flags)
# and most parameters

source "$SLURM_SUBMIT_DIR/lib/utils.sh"

IVLLM_JOB=${1?must supply job name}
IVLLM_GPUS_PER_NODE=${2?must specify gpus per node}
IVLLM_MEM_PER_NODE=${3?must specify mem per node}

export IVLLM_HEAD_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export IVLLM_HEAD_NODE_IP=$(dig +short "$IVLLM_HEAD_NODE")

# launch and background long running process which orchestrates all srun tasks
# this is the central process in the whole server side and this is running on
# the slurm step host. signal propagation to compute nodes is via slurm srun.
# exit codes from failing vllm processes will bubble up here.
# traps in this parent process of the vllm srun commands manage orchestration.
# but mostly intercept signals and call tidy_up before passing them downwards.
(
    SRUN_PIDS=()

    srun \
        --overlap \
        --nodelist="$IVLLM_HEAD_NODE" \
        --nodes=1 \
        --gpus="$IVLLM_GPUS_PER_NODE" \
        --mem="$IVLLM_MEM_PER_NODE" \
        --cpus-per-gpu=64 \
        --ntasks-per-node=1 \
        --export=ALL \
        "$SLURM_SUBMIT_DIR/lib/run_head_vllm.sh" \
        "$IVLLM_JOB" \
        "$IVLLM_HEAD_NODE_IP" \
        & IVLLM_MONITOR_PID=$!

    SRUN_PIDS+=("$IVLLM_MONITOR_PID")

    echo "[serve-0] vllm head srun process $IVLLM_MONITOR_PID"

    declare -i count=1  # Declare as type integer

    IVLLM_WORKER_NODES=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | tail -n+2)
    for IVLLM_WORKER in $IVLLM_WORKER_NODES; do

        srun \
            --overlap \
            --nodelist="$IVLLM_WORKER" \
            --nodes=1 \
            --gpus="$IVLLM_GPUS_PER_NODE" \
            --mem="$IVLLM_MEM_PER_NODE" \
            --cpus-per-gpu=64 \
            --ntasks-per-node=1 \
            --export=ALL \
            "$SLURM_SUBMIT_DIR/lib/run_worker_vllm.sh" \
            "$IVLLM_JOB" \
            "$IVLLM_HEAD_NODE_IP" \
            "$count" \
            & IVLLM_MONITOR_PID=$!

        SRUN_PIDS+=("$IVLLM_MONITOR_PID")

        echo "[serve-$count] vllm worker srun process $IVLLM_MONITOR_PID"
        count+=1          # Increment

    done

    update_status_initialise "$IVLLM_JOB"
    echo "[serve-0] initialised job $IVLLM_JOB"

    setup_traps "$IVLLM_JOB" "${SRUN_PIDS[@]}"
    echo "[serve-0] setting traps on process $$"

    wait_all "${SRUN_PIDS[@]}"
) & IVLLM_PARENT_PID=$!

# Waits for user cancel via lockfile instruction, slurm timeout, or idle timeout:
# This will idle timeout eventually
echo "[serve] initialised head monitor for job $IVLLM_JOB"

# watch lockfile and detect cancel, failure, or idle timeout events.
# communicates to IVLLM_PARENT_PID by signalling.
# signals propagated to nodes via srun.
# monitor process will die with IVLLM_PARENT_PID
monitor_head "$IVLLM_JOB" "$IVLLM_PARENT_PID" & IVLLM_MONITOR_PID=$!
echo "[serve-0] vllm head monitor for job $IVLLM_JOB: process $IVLLM_MONITOR_PID"

# poll health until api responding, warms up vllm, manages cache saving
# startup process will die with IVLLM_PARENT_PID
monitor_startup "$IVLLM_JOB" "$IVLLM_PARENT_PID" & IVLLM_STARTUP_MONITOR_PID=$!
echo "[serve] vllm startup monitor for job $IVLLM_JOB: process $IVLLM_STARTUP_MONITOR_PID"

wait $IVLLM_PARENT_PID

