#!/bin/bash
# ivllm-cancel.sh — Cancel a running vLLM job.
#
# Graceful cancel (default): writes 'cancel' to the job's lockfile.
# The compute-side monitor detects the request and shuts down vLLM cleanly.
# Force cancel (-f): runs scancel on the SLURM job directly.

ivllm_cancel_usage() {
    # Print usage instructions and exit with error code 1.
    echo "Usage: $0 [-j job] [-f]"
    echo ""
    echo "Options:"
    echo "  -j job      The name of the job to start."
    echo "  -a          Abort the job and register as failed."
    echo "  -f          Force removal of lockfile and scancel of job."
    echo "  -h          Show this help message"
    exit 1
}

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

source "$here/lib/utils.sh"

export IVLLM_JOB=""
export IVLLM_FORCE=""
export IVLLM_ABORT="cancel"

OPTIND=1
while getopts "j:fah" opt; do
    case $opt in
        j) IVLLM_JOB="$OPTARG" ;;
        f) IVLLM_FORCE=true ;;
        a) IVLLM_ABORT="abort" ;;
        h) ivllm_cancel_usage ;;
        \?) echo "Error: Invalid option -$OPTARG" >&2; ivllm_cancel_usage ;;
        :)  echo "Error: Option -$OPTARG requires an argument" >&2; ivllm_cancel_usage ;;
    esac
done

if [[ -z "$IVLLM_JOB" ]]; then
    echo "ERROR: No job parameter supplied" >&2
    exit 1
fi

lockfile=$(resolve_job_status "$IVLLM_JOB")
if [[ -f $lockfile ]]; then

    slurmJobId=$(get_job_status_setting "$IVLLM_JOB" ".slurmJobId")
    echo "cancelling slurm job: ${slurmJobId:-unknown}"

    if is_status "$IVLLM_JOB" "failed"; then
        echo "[shutdown] cleaning up failed job $IVLLM_JOB"
        if [[ -n "$slurmJobId" ]]; then
            scancel "$slurmJobId" || echo "WARNING: cancel slurm job: $slurmJobId" >&2
        fi
        rm -f "$lockfile"
        exit 0
    elif is_status "$IVLLM_JOB" "stopped"; then
        echo "[shutdown] cleaning up stopped job $IVLLM_JOB"
        if [[ -n "$slurmJobId" ]]; then
            scancel "$slurmJobId" || echo "WARNING: failed to force cancel slurm job: $slurmJobId" >&2
        fi
        rm -f "$lockfile"
        exit 0
    else


        owner=$(get_job_status_setting "$IVLLM_JOB" ".user")
        status=$(get_job_status_setting "$IVLLM_JOB" ".status")

        if [[ -z $IVLLM_FORCE ]]; then

            echo "[shutdown] shutting down job $IVLLM_JOB with status: $status"
            echo "[shutdown] requesting automatic cancel for $IVLLM_JOB"
            request_cancel "$IVLLM_JOB" "$IVLLM_ABORT"

        elif [[ $owner != $(whoami) ]]; then

            echo "[shutdown] not possible to force cancel job owned by $owner: $IVLLM_JOB with status $status" >&2
            echo "[shutdown] requesting automatic cancel for $IVLLM_JOB"
            request_cancel "$IVLLM_JOB" "$IVLLM_ABORT"

        else

            echo "[shutdown] force cancel job: $IVLLM_JOB with status $status"
            if [[ $type == "cancel" ]]; then
            echo "[cancel] job $IVLLM_JOB force cancelled."
                tidy_up "$IVLLM_JOB" 201
            else
                echo "[cancel] job $IVLLM_JOB force aborted."
                tidy_up "$IVLLM_JOB" 254
            fi
            if [[ -n "$slurmJobId" ]]; then
                scancel "$slurmJobId" || echo "WARNING: could not scancel slurm job: $slurmJobId it may be already dead" >&2
            fi
            rm -f "$lockfile"

        fi

    fi
else
    echo "[shutdown] no job $IVLLM_JOB to cancel"
    exit 1
fi


