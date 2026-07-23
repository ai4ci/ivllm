#!/usr/bin/env bash
# shellcheck disable=SC2059
# ivllm-status.sh — Show status of a vLLM job or all jobs.
#
# Reads status.json from the lockfile and outputs JSON or a formatted row.

ivllm_status_usage() {
    echo "status: $0 [-j job] [-p]"
    echo ""
    echo "Options:"
    echo "  -j job      Pareseable json for this job only (ignores -p)"
    echo "  -p          Return pareseable json."
    echo "  -h          Show this help message"
    exit 1
}

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$here/lib/utils.sh"

export IVLLM_JOB=""
export IVLLM_JSON=""

while getopts "j:ph" opt; do
    case $opt in
        j) IVLLM_JOB="$OPTARG" ;;
        p) IVLLM_JSON=true ;;
        h) ivllm_status_usage ;;
        \?) echo "Error: Invalid option -$OPTARG" >&2; ivllm_status_usage ;;
        :)  echo "Error: Option -$OPTARG requires an argument" >&2; ivllm_status_usage ;;
    esac
done

if [[ -n $IVLLM_JOB ]]; then

    JOB_FILE=$(resolve_job_status "$IVLLM_JOB")
    if [[ ! -f $JOB_FILE ]]; then
        echo "No job file found for $IVLLM_JOB" >&2
        exit 1
    fi
    cat "$JOB_FILE";
    exit 0

fi

# Define the base jobs directory relative to script or absolute path
JOBS_DIR=$(resolve_job_root_dir)

# Check if the jobs directory exists
if [[ ! -d "$JOBS_DIR" ]]; then
    echo "Error: Directory '$JOBS_DIR' not found." >&2
    exit 1
fi

# Find all matching files and store them in an array
mapfile -t status_files < <(find "$JOBS_DIR" -type f -name "status.json" 2>/dev/null)

if [[ -n $IVLLM_JSON ]]; then

  # Request json output
  if (( ${#status_files[@]} == 0 )); then
    echo '[]'
  else
    jq -s '.' "${status_files[@]}"
  fi

else

  # Request table output
  # Check if the array is empty
  if (( ${#status_files[@]} == 0 )); then
      echo "No job status files found."

  else

    FMT="%-8s %-10s %-8s %-20s %-8s %s\n"

    # Print table headers (tabs used as delimiters for column command)
    printf "$FMT" "JOB" "STATUS" "USER" "MODEL" "UNTIL" "INFO"
    printf "$FMT" "===" "======" "====" "=====" "=====" "===="

    # Loop through all status.json files in the hierarchy
    for status_file in "${status_files[@]}"; do
        # Ensure the file is not empty and contains valid JSON
        if [[ -s "$status_file" ]]; then
            jq -r '[
                (.jobName // "N/A"),
                (.status // "N/A"),
                (.user // "N/A"),
                (.model // "N/A" | toString),
                (.stopTime // "N/A"),
                (.reason // ""),
            ] | @tsv' "$status_file"
        fi
    done | awk -F'\t' -v fmt="$FMT" '{printf fmt, $1, $2, $3, $4, $5, $6}'

  fi
fi


