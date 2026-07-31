#!/bin/bash

usage() {
    echo "Usage: $0 [-v] [-o output] [-n count] file..."
    echo ""
    echo "Options:"
    echo "  -v          Enable verbose output"
    echo "  -o output   Write results to output file"
    echo "  -n count    Number of lines to process"
    echo "  -h          Show this help message"
    exit 1
}

VERBOSE=false
OUTPUT=""
COUNT=0

while getopts ":vo:n:h" opt; do
    case $opt in
        v) VERBOSE=true ;;
        o) OUTPUT="$OPTARG" ;;
        n) COUNT="$OPTARG" ;;
        h) usage ;;
        \?) echo "Error: Invalid option -$OPTARG" >&2; usage ;;
        :)  echo "Error: Option -$OPTARG requires an argument" >&2; usage ;;
    esac
done

shift $((OPTIND - 1))

if [ $# -eq 0 ]; then
    echo "Error: No input files specified" >&2
    usage
fi

if [ "$VERBOSE" = true ]; then
    echo "Verbose: ON"
    echo "Output: ${OUTPUT:-stdout}"
    echo "Count: ${COUNT:-all}"
    echo "Files: $@"
    echo ""
fi

for file in "$@"; do
    if [ ! -f "$file" ]; then
        echo "Warning: '$file' not found, skipping" >&2
        continue
    fi

    if [ -n "$OUTPUT" ]; then
        if [ "$COUNT" -gt 0 ] 2>/dev/null; then
            head -n "$COUNT" "$file" >> "$OUTPUT"
        else
            cat "$file" >> "$OUTPUT"
        fi
    else
        if [ "$COUNT" -gt 0 ] 2>/dev/null; then
            head -n "$COUNT" "$file"
        else
            cat "$file"
        fi
    fi
done
