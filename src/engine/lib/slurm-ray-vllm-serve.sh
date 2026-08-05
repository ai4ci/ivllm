#!/bin/bash
# slurm-ray-vllm-serve.sh — Multi-node vLLM job launcher.
# This should only be used for multinode jobs
#
# Submitted by ivllm-serve.sh via sbatch. Resolves environment,
# starts vLLM with proper NCCL/Ray configuration
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
export RAY_PORT=6379

export IVLLM_HEAD_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export IVLLM_HEAD_NODE_IP=$(dig +short "$IVLLM_HEAD_NODE")

# Need to activate the venv in this script to allow us to use ray status and
# nothing else :-(
minVllmVersion=$(get_job_config_setting "$IVLLM_JOB" ".min-vllm-version")
vllmVersion=$(select_closest_version "$minVllmVersion")
vllmVersionDir=$(resolve_vllm_version_dir "$vllmVersion")
source "$vllmVersionDir/bin/activate"

# launch and background long running process which orchestrates all srun tasks
# this is the central process in the whole server side and this is running on
# the slurm step host. signal propagation to compute nodes is via slurm srun.
# exit codes from failing ray workers will bubble up here.
# traps in this parent process of the srun commands manage orchestration.
# but mostly intercept signals and call tidy_up before passing them downwards.
(
    SRUN_PIDS=()

    # Waits for user cancel via lockfile instruction, slurm timeout, or idle timeout:
    # This will idle timeout eventually
    echo "[serve] initialised head monitor for job $IVLLM_JOB"
    # watch lockfile and detect cancel, failure, or idle timeout events.
    # monitor process will die with IVLLM_PARENT_PID
    monitor_head "$IVLLM_JOB" & IVLLM_MONITOR_PID=$!
    echo "[serve-0] vllm head monitor for job $IVLLM_JOB: process $IVLLM_MONITOR_PID"

    setup_traps "$IVLLM_JOB" "$IVLLM_MONITOR_PID" "${SRUN_PIDS[@]}"

    # TODO: https://docs.ray.io/en/latest/cluster/vms/user-guides/community/slurm.html
    # ray symmetric-run

    # setup the ray head and workers
    # The ray-setup.sh script is responsible for setting all environment variables.
    # and jit caches etc.
    # We cannot assume anything defined here is passed onto the ray nodes
    # despite the export=ALL directive.
    srun \
        --overlap \
        --nodelist="$IVLLM_HEAD_NODE" \
        --nodes=1 \
        --gpus="$IVLLM_GPUS_PER_NODE" \
        --mem="$IVLLM_MEM_PER_NODE" \
        --cpus-per-gpu=64 \
        --ntasks-per-node=1 \
        --export=ALL \
        "$SLURM_SUBMIT_DIR/lib/ray-setup.sh" \
        "$IVLLM_JOB" \
        "$IVLLM_HEAD_NODE_IP" \
        & IVLLM_HEAD_PID=$!

    SRUN_PIDS+=("$IVLLM_HEAD_PID")

    setup_traps "$IVLLM_JOB" "$IVLLM_MONITOR_PID" "${SRUN_PIDS[@]}"

    echo "[serve-0] ray head start srun process $IVLLM_HEAD_PID on $IVLLM_HEAD_NODE_IP"

    declare -i count=1  # Declare as type integer

    IVLLM_WORKER_NODES=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | tail -n+2)
    for IVLLM_WORKER in $IVLLM_WORKER_NODES; do

        export IVLLM_WORKER_NODE_IP=$(dig +short "$IVLLM_WORKER")

        srun \
            --overlap \
            --nodelist="$IVLLM_WORKER" \
            --nodes=1 \
            --gpus="$IVLLM_GPUS_PER_NODE" \
            --mem="$IVLLM_MEM_PER_NODE" \
            --cpus-per-gpu=64 \
            --ntasks-per-node=1 \
            --export=ALL \
            "$SLURM_SUBMIT_DIR/lib/ray-setup.sh" \
            "$IVLLM_JOB" \
            "$IVLLM_HEAD_NODE_IP" \
            "$count" \
            "$IVLLM_WORKER_NODE_IP" \
            & IVLLM_WORKER_PID=$!

        SRUN_PIDS+=("$IVLLM_WORKER_PID")

        setup_traps "$IVLLM_JOB" "$IVLLM_MONITOR_PID" "${SRUN_PIDS[@]}"

        echo "[serve-$count] ray worker start srun process $IVLLM_WORKER_PID on $IVLLM_WORKER_NODE_IP"
        count+=1          # Increment

    done

    sleep 20
    # Ray cluster setup - submit the vllm job to the head node of the cluster
    # We submit this to one node desite the fact that it may be bigger than
    # that node.
    srun \
        --overlap \
        --nodelist="$IVLLM_HEAD_NODE" \
        --nodes=1 \
        --gpus="$IVLLM_GPUS_PER_NODE" \
        --mem="$IVLLM_MEM_PER_NODE" \
        --cpus-per-gpu=64 \
        --ntasks-per-node=1 \
        --export=ALL \
        "$SLURM_SUBMIT_DIR/lib/ray-run-vllm.sh" \
        "$IVLLM_JOB" \
        & IVLLM_RAY_VLLM_PID=$!

    SRUN_PIDS+=("$IVLLM_RAY_VLLM_PID")
    setup_traps "$IVLLM_JOB" "$IVLLM_MONITOR_PID" "${SRUN_PIDS[@]}"

    update_status_initialise "$IVLLM_JOB"
    echo "[serve-0] ray vllm initialised job $IVLLM_JOB"

    wait_all "$IVLLM_MONITOR_PID" "${SRUN_PIDS[@]}"

) & IVLLM_PARENT_PID=$!

# Redirect slurm out of time signals to the parent orchestrator
# Slurm errors also.
trap 'kill -s SIGUSR1 "$IVLLM_PARENT_PID" 2>/dev/null' SIGUSR1
trap 'kill -s SIGUSR1 "$IVLLM_PARENT_PID" 2>/dev/null' ERR

while ! process_died "$IVLLM_PARENT_PID"; do
    sleep 1
done

echo "[head] process ($IVLLM_PARENT_PID) has exited. Shutting down."
echo "[head] monitor shutting down for job $IVLLM_JOB."
wait $IVLLM_PARENT_PID # reap zombie

