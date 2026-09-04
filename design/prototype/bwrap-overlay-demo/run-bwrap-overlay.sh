#!/bin/bash
# run-bwrap-overlay.sh — shape of the idea raised in design/priorities.md's
# "Formalise patching method in engine" item: apply a model-specific vLLM
# patch (like minimax-m3-indexer-unfuse.v0.26.0.v1.patch) inside a bwrap
# overlayfs mount, so it exists ONLY for the lifetime of this one process
# tree and the real, shared vLLM install on disk (fake_vllm/app.py here) is
# never touched — no apply/revert dance against a shared venv, no risk of
# one job's patch leaking into another job that happens to land on the same
# node afterward, and no drift between "what's patched" and "what's
# recorded as patched" (the exact gap that made the disable-flashinfer-env
# v0.25.1 patch's tracked diff drift from what was actually deployed — see
# design/prototype/patch/README.md).
#
# THE MECHANISM: bwrap's native overlayfs support (`bwrap --help`, added
# well before 0.11.0 — confirmed present in both bubblewrap 0.11.0
# (Isambard) and 0.11.1 (this sandbox)):
#
#   --overlay-src SRC        add SRC as a read-only lower layer (repeatable —
#                             could stack multiple patches this way, see
#                             WIRING NOTES below)
#   --tmp-overlay DEST        mount the overlay at DEST, with an invisible
#                             tmpfs as the upper (writable) layer
#
# Any write inside the sandbox to a path under DEST copies-up into that
# tmpfs — the lower layer (the real install, --overlay-src) is opened
# read-only by the kernel's overlayfs driver and structurally cannot be
# mutated through this mount, regardless of what the sandboxed process
# tries to do to it. When bwrap exits, the tmpfs upper layer (and every
# copied-up/patched file in it) is gone — there is nothing to revert.
#
# THE ANALOGY TO A REAL JOB: --overlay-src fake_vllm/ stands in for
# --overlay-src "$IVLLM_PROJECTDIR/engine/vllm/<version>/" (the real,
# shared install every job currently runs against unmodified);
# patches/minimax-indexer-unfuse-demo.patch stands in for a real
# design/prototype/patch/diffs/*.patch file; `patch -p1` inside the sandbox
# stands in for apply-vllm-patch.sh's job (except it never needs a
# --revert step — there's nothing durable to revert); `python3
# /vllm/app.py` stands in for `vllm serve ...`.
set -euo pipefail
DEMO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ! command -v bwrap >/dev/null 2>&1; then
    echo "FATAL: bwrap not found on PATH" >&2
    exit 1
fi

echo "=== run-bwrap-overlay.sh: patch applied only inside an overlayfs mount ==="
echo "--- before: host copy of fake_vllm/app.py (should be UNPATCHED) ---"
grep "return \"" "$DEMO_DIR/fake_vllm/app.py"

bwrap \
    --ro-bind /usr /usr \
    --ro-bind /bin /bin \
    $( [[ -d /lib ]] && echo --ro-bind /lib /lib ) \
    $( [[ -d /lib64 ]] && echo --ro-bind /lib64 /lib64 ) \
    --ro-bind /etc /etc \
    --proc /proc \
    --dev /dev \
    --unshare-pid \
    --unshare-net \
    --die-with-parent \
    --new-session \
    --clearenv \
    --setenv PATH /usr/bin:/bin \
    --ro-bind "$DEMO_DIR/patches" /patches \
    --overlay-src "$DEMO_DIR/fake_vllm" \
    --tmp-overlay /vllm \
    -- \
    sh -c '
        set -eu
        echo "--- inside sandbox: applying patch to the OVERLAY, not the real install ---"
        patch -p1 -d /vllm < /patches/minimax-indexer-unfuse-demo.patch
        echo "--- inside sandbox: running the now-patched app ---"
        python3 /vllm/app.py
    '

echo "--- after: host copy of fake_vllm/app.py (should STILL be UNPATCHED) ---"
grep "return \"" "$DEMO_DIR/fake_vllm/app.py"
echo "=== done: the shared install on disk was never touched ==="

# ── WIRING NOTES (how this would become real ivllm behavior) ───────────────
#
# 1. Per-job patch selection: a job's yaml would list which
#    design/prototype/patch/diffs/*.patch files it needs (e.g. via a new
#    `vllm-patches:` key), resolved at the same point set_jit_caches()/
#    set_debugging_env() currently run (after the job's own env: block is
#    evaluated). Multiple patches for the same job = multiple `--overlay-src`
#    args in the order they should apply, or multiple sequential `patch -p1`
#    calls against the one overlay mount, same as apply-vllm-patch.sh
#    already does against the real venv today.
#
# 2. Where the overlay mount goes: real jobs already `source
#    "$vllmVersionDir/bin/activate"` and exec `vllm serve` from inside that
#    activated venv (run-head-vllm.sh/run-worker-vllm.sh/ray-run-vllm.sh).
#    The overlay's DEST would need to be the venv's own
#    site-packages/vllm/ path (or the whole venv root, if a patch ever
#    needs to touch something outside vllm/ itself), with --overlay-src
#    pointing at the CURRENT real path so unpatched files still resolve
#    through to it unmodified — only files the patch actually touches ever
#    copy up into the tmpfs upper layer.
#
# 3. This demo's bwrap invocation is deliberately minimal (--unshare-net,
#    clean env, no GPU/CUDA device binds) — a real one needs everything
#    tests/bash/lib/sandbox.sh's compute profile already binds (GPU device
#    nodes under /dev, NVIDIA driver libs, HCA/Slingshot device files,
#    SLURM_* env passthrough, etc.) plus --unshare-net would need dropping
#    entirely (real jobs need real network for NCCL/Slingshot and the
#    OpenAI-compatible HTTP server) — this prototype only needed to show
#    the patch-isolation SHAPE, not build a production sandbox profile.
#
# 4. Not addressed here: GPU device nodes and NVIDIA driver libraries
#    aren't namespaced the same way as plain files — bwrap's overlay only
#    needs to cover the vLLM *Python package* tree; CUDA/driver access
#    would still bind through normally (--dev-bind or equivalent), same as
#    it does for any bwrap-sandboxed process today.
