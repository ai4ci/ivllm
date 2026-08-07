# `patch/` — local patches to installed third-party packages

Scripts here modify files inside an already-installed vLLM venv on the cluster
(`$PROJECTDIR/engine/vllm/<version>/lib/python3.12/site-packages/...`), not
anything in this repo. They exist because a fix needs to land before an
upstream PR merges, or because the fix is a local workaround that shouldn't
be a permanent part of `slurm-vllm-setup.sh`. See `design/active-issues.md`
for the full background/investigation behind each one.

**Direction of travel**: `hybrid-trtllm.sh` was a one-off script per fix.
`apply-vllm-patch.sh` + `diffs/*.patch` is the first step towards treating
patching as a first-class, versioned ivllm concept — plain unified diffs,
applied/reverted generically by version rather than one bespoke script per
bug. Longer term (not built yet): apply patches into a bubblewrap overlayfs
layer instead of mutating the shared venv in place, so different patch sets
can be composed/switched per job without ever touching the underlying
install — `apply-vllm-patch.sh` is written narrowly enough (it only needs a
different target directory) that it should slot into that later.

## `apply-vllm-patch.sh` + `diffs/`

Generic apply/revert tool for the `.patch` files in `diffs/`. Each patch is
an ordinary unified diff with git-style `a/`/`b/` paths rooted at the vLLM
package's `site-packages` directory (e.g. `vllm/model_executor/layers/foo.py`),
applied via `patch -p1`. Idempotent in both directions — applying an
already-applied patch, or reverting one that isn't applied, is a clean
no-op rather than an error (checked via a dry-run in both directions before
touching anything).

```bash
bash design/prototype/patch/apply-vllm-patch.sh 0.25.1 \
    design/prototype/patch/diffs/skip-flashinfer-fusion-multinode.v0.25.1.patch

bash design/prototype/patch/apply-vllm-patch.sh 0.25.1 \
    design/prototype/patch/diffs/skip-flashinfer-fusion-multinode.v0.25.1.patch --revert
```

Requires `$IVLLM_PROJECTDIR` to already be set — it sources `utils.sh` and
reuses `resolve_vllm_version_dir()` rather than reconstructing the venv path
itself, so it stays correct if that resolution logic ever changes.

### `diffs/disable-flashinfer-env.v0.25.1.patch`

**Supersedes `skip-flashinfer-fusion-multinode.v0.25.1.patch` for MiniMax-M3**
now that the remaining crash (see `design/active-issues.md`) is confirmed
independent of allreduce backend — no need for node-count logic in vLLM
itself when the real fix is "don't use flashinfer's fused path for this
model at all." Simpler, and opt-in per job rather than baked into every
multi-node job on the shared venv:

```python
if os.environ.get("IVLLM_DISABLE_FLASHINFER") == "1":
    logger.info_once(...)
    return False, 0
```

Set `IVLLM_DISABLE_FLASHINFER: 1` in a job's yaml `env:` block to unconditionally
skip flashinfer's fused allreduce+RMSNorm for that job only — other
jobs/models sharing the same venv are unaffected unless they opt in too.
Because it's a plain env var (not a vLLM `--config` yaml key), it isn't
subject to the boolean-`false`-dropping bug documented in
`design/active-issues.md` — it shows up as a literal `IVLLM_DISABLE_FLASHINFER=1`
in the job's env-var dump in the log either way, and logs an explicit
`info_once` line when it actually takes effect.

### `diffs/skip-flashinfer-fusion-multinode.v0.25.1.patch`

**Problem it works around**: `hybrid-trtllm.sh` (below) fixes `trtllm`
selection for the case where the TP group is node-local but the *deployment*
spans multiple nodes (TP=4×DP=2). It does **not** help when the TP group
itself genuinely spans multiple nodes (e.g. TP=8 across 2×4-GPU nodes,
tested 2026-08-07) — there, `trtllm` is *correctly* rejected (it fundamentally
cannot do multi-node allreduce, patched or not), and the only other backend,
`mnnvl`, crashes the CUDA context outright on Slingshot (no real inter-node
NVLink/NVSwitch — confirmed, see `design/active-issues.md`). So for a
genuinely cross-node TP group there is currently no working flashinfer
allreduce backend at all, and no config flag on MiniMax-M3 to skip the fused
path (unlike Qwen3.5, which can disable it via `fuse_allreduce_rms: false`
since its call is routed through that compile pass — MiniMax-M3 calls
`fused_allreduce_gemma_rms_norm` directly from its own decoder layer).

**What the patch does**: `_can_use_flashinfer()`
(`vllm/model_executor/layers/fused_allreduce_gemma_rms_norm.py`) already has
a fully safe, numerically-identical fallback for every other case it can't
use the fused path (`tensor_model_parallel_all_reduce` + plain `GemmaRMSNorm`
— see the module's own docstring). This patch adds one more case to that list:
return `False, 0` immediately whenever the TP group itself spans more than
one node, using the same `_node_count(get_tp_group().cpu_group)` check as
`hybrid-trtllm.sh`. Single-node TP groups are completely unaffected and still
get the `trtllm` fusion; multi-node TP groups now skip flashinfer entirely
instead of hitting either the `trtllm` `ValueError` or the `mnnvl` CUDA crash.

**Relationship to `hybrid-trtllm.sh`**: both patches are useful independently
and don't conflict — `hybrid-trtllm.sh` fixes `get_fi_ar_workspace()`'s other
caller (`AllReduceFusionPass`, the torch.compile route other models use, e.g.
Qwen3.5/DeepSeek-V3.2) for the node-local-TP-inside-multi-node-deployment
case; this one specifically stops MiniMax-M3's direct, non-compile-pass call
from ever reaching a broken backend once its TP group is genuinely cross-node.
Apply both if unsure which case you're in — they only ever matter in disjoint
situations.

**Status (2026-08-07)**: not yet tested against a real run — drafted
immediately after diagnosing the TP=8×Ray dead end, before the next attempt.
If it works, the natural next step is folding it into `slurm-vllm-setup.sh`
(same as the `nvidia-nccl-cu12` pin) rather than requiring a manual apply step
per fresh venv.

## `hybrid-trtllm.sh`

**Problem it works around**: vLLM's `get_fi_ar_workspace()`
(`vllm/distributed/device_communicators/flashinfer_all_reduce.py`) refuses to
use the `trtllm` flashinfer allreduce backend whenever the *whole deployment*
spans more than one node (`get_node_count() > 1`), even when the specific
tensor-parallel group performing the allreduce never leaves a single node —
exactly the case for any `tensor-parallel-size` ≤ GPUs-per-node deployment
that adds `data-parallel-size`/`pipeline-parallel-size` > 1 to reach more
nodes (e.g. MiniMax-M3's TP=4×DP=2 across 2×4-GPU nodes). The check ignores
the `group` parameter it's already given and should be checking that group's
own node span, not the deployment's.

**What the patch does**: in the installed
`flashinfer_all_reduce.py`, changes the guard from

```python
if get_node_count() > 1 and backend == "trtllm":
```
to
```python
if _node_count(get_tp_group().cpu_group) > 1 and backend == "trtllm":
```

`_node_count()` is an existing vLLM helper (`parallel_state.py`) that computes
the node span of any process group — this just points it at the TP group
instead of the (irrelevant) whole-deployment world group. It must be called
with `get_tp_group().cpu_group`, not `.device_group` — `_node_count()` /
`in_the_same_node_as()` assert the group passed in is **not** an NCCL group
(a first version of this patch used `.device_group` directly and crashed
with `AssertionError: in_the_same_node_as should be tested with a non-NCCL
group`, since `.device_group` *is* the NCCL communicator).

**Usage**: run on the cluster, after the target vLLM venv exists:

```bash
bash design/prototype/patch/hybrid-trtllm.sh
```

It's idempotent — re-running it first restores `flashinfer_all_reduce.py`
from the `.orig` backup it made on first run, then re-applies the patch
cleanly, so it's safe to run again after any change to the script itself.
To revert permanently, just restore the backup:

```bash
mv "$PROJECTDIR/engine/vllm/0.25.1/lib/python3.12/site-packages/vllm/distributed/device_communicators/flashinfer_all_reduce.py.orig" \
   "$PROJECTDIR/engine/vllm/0.25.1/lib/python3.12/site-packages/vllm/distributed/device_communicators/flashinfer_all_reduce.py"
```

**Caveats**:
- Only patches the *currently installed* `0.25.1` venv. A version bump
  creates a fresh venv and silently loses the patch — if this proves out
  long-term, fold it into `slurm-vllm-setup.sh` as a post-install step
  (same pattern as the `nvidia-nccl-cu12` pin already there).
- The Python heredoc hardcodes `/projects/b6ax/engine/vllm/0.25.1/...`
  rather than expanding `$PROJECTDIR`/`$vllmVersion` like the bash wrapper
  around it does — fine for this project's current deployment, but not
  portable as written if reused elsewhere.
- Does **not** fix everything: it resolves the `trtllm`/`mnnvl` backend
  selection crash, but MiniMax-M3 under TP=4×DP=2 still crashes later, on
  something unrelated (a Triton kernel illegal-memory-access in the model's
  own sparse-attention decode kernel). See `design/active-issues.md` for the
  full state.
