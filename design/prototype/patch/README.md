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
bash design/prototype/patch/apply-vllm-patch.sh 0.26.0 \
    design/prototype/patch/diffs/shm-broadcast-lost-notify-fix.v0.26.0.v1.patch

bash design/prototype/patch/apply-vllm-patch.sh 0.26.0 \
    design/prototype/patch/diffs/shm-broadcast-lost-notify-fix.v0.26.0.v1.patch --revert
```

**Naming convention: `<descriptive-name>.v<vllm-version>.v<revision>.patch`.** Every patch file is versioned by filename, not edited in place — when a patch's content needs to change, save it as a new file with the revision number bumped (`...v1.patch` → `...v2.patch`) rather than overwriting the existing one. Superseded revisions and fully-retired patches move to `diffs/old/` (kept for reference, not deleted) rather than being removed outright.

This exists because editing an already-applied patch file in place is genuinely dangerous: the dry-run idempotency check only compares the *current* file content against the *current* venv state, with no way to know an *older* version of that same file is what's actually installed. If new content partially overlaps old (some hunks match, some don't), *both* the forward and reverse dry-runs can fail, and the script falls back to printing `"already applied — nothing to do"` and exits without applying anything — silently leaving the venv in whatever incomplete state the old version left it in, with no error. Confirmed by direct testing while what's now `shm-broadcast-stuck-queue-diagnostics.v0.26.0.v2.patch` was still being developed (see `design/active-issues.md`). `apply-vllm-patch.sh` now also cross-checks the `.ivllm-patches-applied` manifest for *other* revisions of the same base name and warns if one's still recorded as applied — but reverting the old revision before applying a new one remains the correct sequence regardless; the warning is a safety net, not a substitute.

Requires `$IVLLM_PROJECTDIR` to already be set — it sources `utils.sh` and
reuses `resolve_vllm_version_dir()` rather than reconstructing the venv path
itself, so it stays correct if that resolution logic ever changes.

### `diffs/shm-broadcast-stuck-queue-diagnostics.v0.26.0.v2.patch`

**Purpose**: a diagnostic-only patch, not a fix — enriches the existing
`"No available shared memory broadcast block found"` warning (which already
fires in every GLM-5.2-INT4 hang, but says nothing about *why*) with the
actual ring-buffer metadata state at the moment it fires. Every prior pyspy
capture this investigation left one question unanswered: was the response
actually written and something just failed to notice (a "lost notification"
shape — see `shm-broadcast-lost-notify-fix.v0.26.0.v1.patch` above), or was it
never written at all (pointing at a genuinely stuck collective upstream, in
NCCL/GPU work, not the IPC layer)? This patch answers that directly, every
time the warning fires, with zero overhead in the non-hanging case (the new
logging is gated behind the exact same existing warning condition).

**What it does**:
- On the reader side (`acquire_read`, hit by `EngineCore` waiting on worker
  responses *and* by workers waiting on the request-broadcast queue from
  `EngineCore`): logs `written_flag` (0 = the writer never got there; 1 =
  data is ready but this reader hasn't consumed it) and this reader's own
  read-flag, plus a label identifying *which* queue.
- On the writer side (`acquire_write`): logs `written_flag` and every
  reader's flag for the block it's trying to reuse, plus how many readers
  are outstanding.
- Tags each `MessageQueue` in `multiproc_executor.py` with a human-readable
  `_ivllm_debug_label` (`"request_broadcast"`, `"response[rank=N]"`) at the
  one place `rank` is already naturally available when the queues are
  constructed — `shm_broadcast.py` itself has no rank concept, so without
  this the diagnostic log would only be able to name a queue by its raw
  shared-memory segment name (still logged as a fallback if the attribute
  isn't set, so this degrades gracefully for any caller that doesn't tag
  its queues).

**Status (2026-08-11)**: prepared and verified to apply cleanly
(`git apply --check`) against `vendor/vllm-0.26.0`; not yet tested against a
real hang. Safe to apply alongside `shm-broadcast-lost-notify-fix.v0.26.0.v1.patch`
— they touch overlapping but non-conflicting regions of the same functions.
Grep job logs for `[ivllm-diag]` to find the new lines.

**v2 (2026-08-12)**: the first real run showed `queue=psm_<hex>` (the raw
shared-memory segment name) instead of a friendly label — `RayExecutorV2`
builds its own `rpc_broadcast_mq`/`response_mqs` independently of
`MultiprocExecutor`'s `__init__` (different code path, same base class), so
the v1 labeling in `multiproc_executor.py` never ran for this project's
actual deployment (which always uses the Ray executor). v2 adds the same
`_ivllm_debug_label` tagging to `ray_executor_v2.py`'s own construction
sites. Verified to still apply cleanly, alone and stacked with
`shm-broadcast-lost-notify-fix.v0.26.0.v1.patch`.

### `diffs/shm-broadcast-lost-notify-fix.v0.26.0.v1.patch`

**Problem it works around**: the GLM-5.2-AWQ-INT4 multi-node hang investigation
(see `design/active-issues.md`) traced every observed hang to
`vllm/distributed/device_communicators/shm_broadcast.py` — `EngineCore` or a
worker waiting indefinitely for a `shm_broadcast` response that, from the
other side's perspective, was already sent. Checking `vendor/vllm`'s full git
history found this exact file had two relevant bugfixes land upstream
*after* `0.26.0` (this project's currently-installed version) but *before*
`0.27.0`/`0.27.1` (both already tagged upstream as of 2026-08-11):

- [`10c75477b`](https://github.com/vllm-project/vllm/commit/10c75477b) (#45224) — an idle
  reader waiting indefinitely with no active warning/deadline could get an
  unbounded poll timeout on the ZMQ notify socket, with no periodic fallback
  recheck of the actual shared-memory flag. If that one wake-up notification
  is ever lost, the reader hangs forever even though the data was already
  correctly written. Fixed by capping every wait at a new
  `SHM_READER_RECHECK_INTERVAL_MS = 5000` regardless of state. Same commit
  also wraps the read-flag-setting code in `try`/`finally`, so an exception
  while a caller processes a dequeued buffer (e.g. handling an aborted
  request) can no longer permanently strand that reader's slot as "not read."
- [`48aa8d8d7`](https://github.com/vllm-project/vllm/commit/48aa8d8d7) (#41357) — a sign
  bug: a deadline computation going slightly negative produced a negative
  timeout passed to `zmq.Socket.poll()`, which treats negative timeouts as
  *block indefinitely* (libzmq's own `-1` convention) rather than *already
  expired*. Fixed via `max(0, ...)` clamping.

**What the patch does**: cherry-picks both commits' changes onto the
installed `0.26.0` tree (built by cherry-picking both onto a `v0.26.0`
worktree of `vendor/vllm` and diffing the result — both applied with zero
conflicts). Touches `shm_broadcast.py` (both fixes) and
`v1/executor/multiproc_executor.py` (the second fix's other call site).

**Status (2026-08-11)**: prepared and verified to apply cleanly
(`git apply --check`) against `vendor/vllm-0.26.0`; not yet tested against a
real hang. This is the current leading theory and immediate next test for
the GLM-5.2-INT4 hang — see `design/active-issues.md` for the full
evidence chain (including a related, still-unconfirmed weak-memory-ordering
angle in the same file, upstream issue
[`#27858`](https://github.com/vllm-project/vllm/issues/27858)).

### `diffs/disable-flashinfer-env.v0.25.1.v1.patch`

**Supersedes `old/skip-flashinfer-fusion-multinode.v0.25.1.patch` for MiniMax-M3**
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

### `diffs/old/skip-flashinfer-fusion-multinode.v0.25.1.patch` (archived, superseded)

Moved to `old/` — superseded by `disable-flashinfer-env.v0.25.1.v1.patch`
above, kept for reference only. Not meant to be applied going forward.

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
