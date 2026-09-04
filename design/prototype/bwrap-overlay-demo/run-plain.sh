#!/bin/bash
# run-plain.sh — baseline: "run vllm" today, no isolation, no patching.
#
# This is the shape of every current ivllm job: run-head-vllm.sh /
# run-worker-vllm.sh eventually just `exec vllm serve ...` against whatever
# is installed at $IVLLM_PROJECTDIR/engine/vllm/<version>/ — the SHARED
# venv, used as-is by every job/model on that venv. fake_vllm/app.py stands
# in for that installed vLLM package.
#
# Compare against run-bwrap-overlay.sh in this same directory — same
# fake_vllm/, same underlying idea (run something against an installed
# package), different question: does a model-specific patch on top of it
# persist beyond this one run, and does it require mutating the shared
# install to apply at all?
set -euo pipefail
DEMO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== run-plain.sh: no isolation, no patch, straight against the shared install ==="
python3 "$DEMO_DIR/fake_vllm/app.py"
