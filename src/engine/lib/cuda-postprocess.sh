#!/bin/bash
#
# Invoked directly by the CUDA driver as the target of a piped
# CUDA_COREDUMP_FILE: the driver forks this command and streams the raw
# GPU coredump into its stdin. We materialize that stream to a node-local
# temp file (read() on a pipe naturally blocks until EOF, i.e. until the
# driver finishes writing — no size-polling needed), decode it immediately
# with whatever cuda-gdb this node already has loaded (guaranteed to match
# this node's exact host arch + CUDA driver version, unlike any offline
# analysis elsewhere), then discard the multi-GB binary and keep only the
# small text digest.
#
# Args (baked into CUDA_COREDUMP_FILE's command line at export time, plus
# %h/%p/%t substituted by the driver before it's handed to the shell):
#   $1 = node-local scratch dir (fast tmpfs, for the throwaway raw file)
#   $2 = shared debug output dir (where the final .txt digest lands)
#   $3 = debug level (for differential post processing - not currently used)
set -uo pipefail

scratch="$1" outdir="$2" debug_level="$3"

host="$(hostname)"
ts="$(date +%s)"

# N.B. node local scratch is a limited space ramfs in CPU memory.
tmp_core="$scratch/coredump-$$-${ts}.raw"

i=1; while [ -e "$outdir/cuda-dump-${host}-${ts}.$i.txt" ]; do ((i++)); done;
out="$outdir/cuda-dump-${host}-${ts}.$i.txt"
trap 'rm -f "$tmp_core"' EXIT

# blocks the process until each dump is processed.
# lock is released when this script exits.
# This poses a risk of bringing the inference to its knees and is a tradeoff
# that possibly not needed unless capture causes a massive issue
exec 9>"$scratch/cuda-postprocess.lock"
flock 9

cat > "$tmp_core"  # blocks until the driver closes its end of the pipe

cuda-gdb -q --batch \
    -ex "target cudacore $tmp_core" \
    -ex "info inferiors" \
    -ex "info cuda devices" \
    -ex "info cuda contexts" \
    -ex "info cuda kernels" \
    -ex "cuda thread apply all backtrace" \
    >> "$out" 2>&1

echo "[cuda-coredump] level: $debug_level; decoded host=$host -> $out ($(wc -l < "$out" 2>/dev/null || echo 0) lines)" >&2
