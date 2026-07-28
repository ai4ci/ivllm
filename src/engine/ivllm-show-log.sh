#!/bin/bash
# ivllm-show-log.sh — Tail the vLLM output log for a job.
#
# Streams the remote log file (vllm.<nodeid>.log) to stdout.
# Supports tailing from a specific node or from the beginning.

ivllm_show_logs_usage() {
    echo "Usage: $0 [-j job] [-n] [-m match]"
    echo ""
    echo "Options:"
    echo "  -j job      The name of the job to start."
    echo "  -n node     Which node to show or quoted '*' for all"
    echo "  -m match    A string when matched terminates the tail"
    echo "  -h          Show this help message"
    exit 1
}

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$here/lib/utils.sh"

export IVLLM_JOB=""
export IVLLM_NODE="0"
export IVLLM_MATCH=""

OPTIND=1
while getopts "j:n:m:h" opt; do
    case $opt in
        j) IVLLM_JOB="$OPTARG" ;;
        n) IVLLM_NODE="$OPTARG" ;;
        m) IVLLM_MATCH="$OPTARG" ;;
        h) ivllm_show_logs_usage ;;
        \?) echo "Error: Invalid option -$OPTARG" >&2; ivllm_show_logs_usage ;;
        :)  echo "Error: Option -$OPTARG requires an argument" >&2; ivllm_show_logs_usage ;;
    esac
done

if [[ -z "$IVLLM_JOB" ]]; then
    echo "ERROR: No job parameter supplied" >&2
    exit 1
fi

jobDir=$(resolve_job_dir "$IVLLM_JOB")
logsGlob="$jobDir/vllm.$IVLLM_NODE.log"

# Safely expand the glob into an array of existing files
# This handles the literal '*' case correctly
shopt -s nullglob
# shellcheck disable=SC2206
files=($logsGlob)
shopt -u nullglob

if (( ${#files[@]} == 0 )); then
    echo "Error: No log files matched '$logsGlob'" >&2
    exit 1
fi

# Pass the array of files unquoted into tail
# tail -f handles multiple files natively, printing headers for each
# tail -n +1 -f "${files[@]}"

stdbuf -oL tail -f "${files[@]}" || awk -v target="$IVLLM_MATCH" '
    target != "" && index($0, target) { print; exit }
    1;
    { fflush() }
'
