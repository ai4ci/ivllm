#!/bin/bash
# design/prototype/nccl-probe.sh — PROTOTYPE, not production (see AGENTS.md:
# scripts in design/ are instructional examples, rewrite before shipping).
#
# Question this answers: which libnccl.so does a real vLLM process tree
# actually load at runtime on Isambard — the pip-wheel copy pulled in as a
# torch/vllm dependency, the `brics/nccl` module's copy, or NVHPC 26.3's
# bundled copy — and does LD_PRELOAD actually win the race against however
# torch self-loads its bundled CUDA libraries?
#
# Why this matters: vLLM issue #46097 (github.com/vllm-project/vllm/issues/46097)
# documents a multi-node TP collective-desync deadlock traced to a specific
# *build* of NCCL (not a version number — two builds with the identical
# version string, 2.28.9, behaved completely differently: pip wheel deadlocked
# in ~20min, vendor/system package ran clean). The same thread found that
# NCCL_VERSION env vars and NCCL_DEBUG=WARN's log banner can both be stale or
# suppressed — i.e. you cannot trust env-var/log-based version reporting, you
# have to check which file the kernel actually mapped into the process
# (/proc/<pid>/maps) and hash it.
#
# Non-invasive: read-only checks against whatever's already installed
# (module-provided NCCL, NVHPC's bundled NCCL, the vllm venv's pip-wheel
# NCCL). Does NOT build or install anything. Safe to run repeatedly.
#
# Section [7] additionally answers a second question: swapping the core
# NCCL library is only half the story on Isambard — `aws-ofi-nccl` (the
# plugin bridging NCCL onto Slingshot's Cassini/libfabric transport) is
# version-paired with specific NCCL releases (see the Isambard NCCL guide),
# so a mismatch could silently fall back to plain TCP sockets instead of
# failing loudly — "works" but catastrophically slower (the guide's own
# numbers: 2.32 GB/s vs 162.69 GB/s). That only shows up once a real NCCL
# communicator actually initialises (ncclCommInitRank's transport
# selection), which sections [1]-[6] never trigger — importing torch alone
# loads the library but never negotiates a transport. Needs a genuine
# 2-node allocation (2 GPUs on the same node would likely just use NVLink
# and never touch the network transport at all).
#
# Usage (on an Isambard GPU compute node — needs CUDA):
#   # Sections [1]-[6] only, single node:
#   srun --partition=... --gpus=1 --pty bash design/prototype/nccl-probe.sh <vllm-version>
#   # All sections including [7]'s transport-selection test, 2 nodes:
#   srun --partition=... --nodes=2 --ntasks=2 --gpus-per-task=1 --pty \
#        bash design/prototype/nccl-probe.sh <vllm-version>
# e.g.
#   srun --reservation=interactive --gpus=1 --pty bash design/prototype/nccl-probe.sh 0.25.1
#   srun --reservation=interactive --nodes=2 --ntasks=2 --gpus-per-task=1 --pty \
#        bash design/prototype/nccl-probe.sh 0.25.1

set -uo pipefail

VLLM_VERSION=${1:?"usage: $0 <vllm-version> (e.g. 0.25.1) — must already be installed via 'ivllm setup'"}
ENGINE_LIB="$HOME/.local/bin/lib"

hash_of() {
    sha256sum "$1" 2>/dev/null | cut -d' ' -f1 || echo "<unreadable>"
}

# Every libnccl.so* reachable via the CURRENT $LD_LIBRARY_PATH, deduplicated.
list_ld_library_path_candidates() {
    echo "$LD_LIBRARY_PATH" | tr ':' '\n' | while read -r d; do
        [[ -d "$d" ]] || continue
        find "$d" -maxdepth 1 -name 'libnccl.so*' 2>/dev/null
    done | sort -u
}

print_candidates() {
    local any=0
    list_ld_library_path_candidates | while read -r f; do
        any=1
        printf '  %s\n    sha256: %s\n' "$f" "$(hash_of "$f")"
    done
    [[ "$any" -eq 0 ]] && echo "  (none found on current LD_LIBRARY_PATH)"
}

echo "=== [1] Loading modules + env exactly as common-env.sh does ==="
# shellcheck disable=SC1091
source "$ENGINE_LIB/utils.sh"       # resolve_* helpers only
# shellcheck disable=SC1091
source "$ENGINE_LIB/common-env.sh"  # module load brics/nccl, libfabric, gcc-native; builds LD_LIBRARY_PATH

echo
echo "=== [2] libnccl candidates on LD_LIBRARY_PATH after common-env.sh (module + NVHPC state, no venv yet) ==="
print_candidates

vllmVersionDir=$(resolve_vllm_version_dir "$VLLM_VERSION")
# shellcheck disable=SC1091
source "$vllmVersionDir/bin/activate"
# shellcheck disable=SC1091
source "$ENGINE_LIB/vllm-env.sh"

echo
echo "=== [3] libnccl candidates on LD_LIBRARY_PATH after venv activation ==="
print_candidates

echo
echo "=== [4] pip-wheel NCCL package location (NOT necessarily on LD_LIBRARY_PATH — torch loads it directly by path, this is just where it lives on disk) ==="
python3 - <<'PYEOF' || echo "  (nvidia-nccl-cu* package not found in this venv)"
import importlib.util, pathlib
spec = importlib.util.find_spec("nvidia.nccl")
if spec and spec.submodule_search_locations:
    libdir = pathlib.Path(spec.submodule_search_locations[0]) / "lib"
    for f in sorted(libdir.glob("libnccl.so*")):
        print(f"  {f}")
else:
    raise SystemExit(1)
PYEOF

# ── Ground truth: what does a real torch process actually map, and what does
# THAT LIBRARY'S OWN CODE say its version is? ──
#
# Two independent checks, deliberately not trusting either alone:
#   - torch.cuda.nccl.version() turned out to be misleading in practice: after
#     upgrading the venv's nvidia-nccl-cu12 package, the mapped file's hash
#     changed (proof the new file loads) but torch.cuda.nccl.version() kept
#     reporting the OLD version — it looks like a value baked into the torch
#     wheel at torch's own build time, not a live query of what's loaded.
#   - So the authoritative check calls ncclGetVersion() directly, via ctypes,
#     on whatever file(s) actually ended up mapped into /proc/self/maps —
#     that's the library's own binary reporting its own compiled-in version,
#     with no torch-level indirection to go stale.
#
# `import torch` alone triggers torch's bundled-CUDA-library self-loading
# (it explicitly ctypes.CDLL()s its own copies at import time) — no GPU
# collective, no distributed init, and no GPU strictly required to observe
# this, though this script is meant to run on a GPU node to match reality.
# Runs synchronously now (introspects its own /proc/self/maps from inside the
# same process) — no need to background it and race a `sleep` against a
# separate bash-side maps read.
probe_loaded_nccl() {
    local label="$1"
    echo
    echo "--- $label ---"
    python3 - <<'PYEOF'
import ctypes
import hashlib
import os

def sha256_of(path):
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError as e:
        return f"<unreadable: {e}>"

def nccl_version_of(path):
    # ncclGetVersion(int *version) encodes as MAJOR*10000 + MINOR*100 + PATCH.
    # Calling it directly on the already-mapped .so (ctypes.CDLL on a path
    # that's already loaded returns a handle to the SAME resident library,
    # it does not load a second copy) is the one check here that cannot be
    # stale — it's the library's own compiled-in answer about itself.
    try:
        lib = ctypes.CDLL(path)
        v = ctypes.c_int()
        rc = lib.ncclGetVersion(ctypes.byref(v))
        if rc != 0:
            return f"<ncclGetVersion returned error code {rc}>"
        n = v.value
        return f"{n // 10000}.{(n // 100) % 100}.{n % 100} (raw={n})"
    except Exception as e:
        return f"<ncclGetVersion call failed: {e}>"

import torch
try:
    print("torch.cuda.nccl.version() reports:", torch.cuda.nccl.version())
    print("  (treat this as untrustworthy — looks compile-time-baked into the")
    print("   torch wheel, not a live query; see the direct checks below instead)")
except Exception as e:
    print("torch.cuda.nccl.version() failed:", e)

print("pid:", os.getpid())

paths = set()
with open(f"/proc/{os.getpid()}/maps") as f:
    for line in f:
        if "nccl" in line.lower():
            path = line.split()[-1]
            if path.startswith("/"):
                paths.add(path)

print("libnccl.so file(s) actually mapped into this process (ground truth):")
if not paths:
    print("  (none found — unexpected, torch should have loaded one)")
for path in sorted(paths):
    print(f"  {path}")
    print(f"    sha256: {sha256_of(path)}")
    print(f"    ncclGetVersion() direct call: {nccl_version_of(path)}")
PYEOF
}

echo
echo "=== [5] Baseline: what actually loads with no LD_PRELOAD (current default behaviour) ==="
probe_loaded_nccl "no LD_PRELOAD"

echo
echo "=== [6] Does LD_PRELOAD override it? Try every candidate library found above ==="
list_ld_library_path_candidates | while read -r candidate; do
    LD_PRELOAD="$candidate" probe_loaded_nccl "LD_PRELOAD=$candidate"
done

echo
echo "=== Sections [5]/[6] verdict: compare the mapped path in [6] against each LD_PRELOAD target — ==="
echo "=== if they match, the mechanism works here; if [6] still shows the same file as [5]         ==="
echo "=== regardless, torch's own loading is winning and LD_PRELOAD alone won't be enough.          ==="

# ── [7] NCCL_DEBUG=INFO transport-selection smoke test ──
#
# Everything above only proves *which file* loads — it never initialises a
# real NCCL communicator, so it can't tell us whether transport selection
# at init time (NET/OFI via Cassini/libfabric vs a silent NET/Socket
# fallback to plain TCP) still works with whatever NCCL is now in effect.
# That only happens inside ncclCommInitRank, reached via a real multi-rank
# rendezvous — hence needing SLURM_NTASKS>=2 (see the usage note at the top
# of this file for the required --nodes=2 --ntasks=2 --gpus-per-task=1
# invocation).
if [[ "${SLURM_NTASKS:-1}" -ge 2 ]]; then
    echo
    echo "=== [7] NCCL_DEBUG=INFO transport-selection test (rank ${SLURM_PROCID:-?} of $SLURM_NTASKS) ==="

    headNode=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
    export MASTER_ADDR=$(dig +short "$headNode")
    export MASTER_PORT="${MASTER_PORT:-29500}"

    ncclLog=$(mktemp)
    NCCL_DEBUG=INFO python3 - "${SLURM_PROCID:-0}" "$SLURM_NTASKS" "${SLURM_LOCALID:-0}" \
        > "$ncclLog" 2>&1 <<'PYEOF'
import sys
import torch
import torch.distributed as dist

rank, world_size, local_rank = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
torch.cuda.set_device(local_rank)
dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

x = torch.ones(1, device="cuda") * (rank + 1)
dist.all_reduce(x)
expected = sum(range(1, world_size + 1))
status = "OK" if x.item() == expected else "MISMATCH"
print(f"[rank {rank}] all_reduce result: {x.item()} (expected {expected}) — {status}", flush=True)

dist.destroy_process_group()
PYEOF
    pyExit=$?

    echo "--- rank ${SLURM_PROCID:-?}: NCCL transport-selection lines ---"
    transportLines=$(grep -E 'NET/(OFI|Socket|IB)' "$ncclLog" || true)
    if [[ -n "$transportLines" ]]; then
        echo "$transportLines" | sed "s/^/  [rank ${SLURM_PROCID:-?}] /"
    else
        echo "  (no NET/ selection line found — see full log below)"
    fi

    if grep -q 'NET/Socket' "$ncclLog"; then
        echo "  *** WARNING: fell back to NET/Socket (TCP) — Slingshot/Cassini NOT in use, expect severe slowdown ***"
    elif grep -q 'NET/OFI' "$ncclLog"; then
        echo "  OK: NET/OFI selected — Cassini/libfabric transport is active"
    fi

    grep 'all_reduce result' "$ncclLog" | sed "s/^/  [rank ${SLURM_PROCID:-?}] /"

    if [[ "$pyExit" -ne 0 ]]; then
        echo "--- rank ${SLURM_PROCID:-?}: NCCL init/all_reduce FAILED (exit $pyExit) — full log follows ---"
        cat "$ncclLog"
    fi
    rm -f "$ncclLog"
else
    echo
    echo "=== [7] Skipping NCCL transport-selection test — needs a 2-node/2-task allocation ==="
    echo "    Rerun as: srun --reservation=interactive --nodes=2 --ntasks=2 --gpus-per-task=1 --pty \\"
    echo "                   bash design/prototype/nccl-probe.sh $VLLM_VERSION"
fi
