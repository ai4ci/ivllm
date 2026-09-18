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

### `diffs/disable-flashinfer-import.v0.26.0.v1.patch`

**Companion to `disable-flashinfer-env`, not a superseder — both are needed for `IVLLM_DISABLE_FLASHINFER=1` to actually prevent flashinfer from touching a job.** Found 2026-09-03 while diagnosing a MiniMax-M3-AWQ-INT4 job that stalled silently (no crash, no NCCL/network involvement, GPU util pinned at 0%) for 5 minutes before being aborted — see `design/active-issues.md`. Root cause: `EngineCore`'s own process (not any GPU worker) hung inside `torch.cuda._lazy_init()`, reached via a completely different path than the one `disable-flashinfer-env` guards:

```
Scheduler.__init__ → supports_multimodal_inputs() → resolve_model_cls()
→ import vllm.models.minimax_m3 (model-class resolution, not kernel dispatch)
→ vllm/compilation/passes/fusion/allreduce_rms_fusion.py: import flashinfer.comm
→ flashinfer/jit/env.py's module-level torch.cuda.get_device_capability() call hangs
```

`disable-flashinfer-env` only patches `_can_use_flashinfer()` — a *runtime* check that skips calling the fused kernel. It does nothing to prevent `import flashinfer` itself, which two vLLM modules do unconditionally at module level (guarded only by `try/except ImportError`, which doesn't help since flashinfer *is* installed — it just hangs instead of raising):

- `vllm/compilation/passes/fusion/allreduce_rms_fusion.py` (`if find_spec("flashinfer"): try: import flashinfer.comm ...`)
- `vllm/distributed/device_communicators/flashinfer_all_reduce.py` (`try: import flashinfer.comm ...`)

Both get pulled in transitively the moment any code imports a model file that imports `fused_allreduce_gemma_rms_norm.py` (MiniMax-M3, GLM-5.2's model classes) — which happens during model-class resolution, independent of whether the fused kernel ever gets called. This patch adds the same `IVLLM_DISABLE_FLASHINFER` env-var guard around both import sites, mirroring the "flashinfer not installed" fallback these modules already support (so nothing downstream needed to change — `flashinfer_comm`/`fi_ar_available` degrade exactly as they would if the package were absent).

**Status: tested live, 2026-09-03 — fixed the hang, but exposed a design problem, superseded by `v2` below.** Applied to the real venv and confirmed the `EngineCore` hang is gone (`logs/mmax3q/20260903_161157/` ran to serving). But this v1's guard ties "does this process even attempt the import" to the *same* `IVLLM_DISABLE_FLASHINFER` flag that governs "should Workers use the fused kernel" — so there's no way to get *both* "EngineCore doesn't hang" *and* "Workers use the real flashinfer path" at once. With the flag set (avoids the hang), every Worker is forced onto the plain `all_reduce`+`GemmaRMSNorm` fallback for every layer — and on this same run, generation was syntactically valid but completely incoherent (repeating-token garbage). With the flag unset (lets Workers use flashinfer normally), `EngineCore`'s import hang comes straight back — confirmed by Rob testing exactly that. See `design/active-issues.md` for the follow-up investigation; not yet confirmed whether the fallback path itself is the source of the garbage output (still the leading hypothesis) or something else (e.g. `ivllm-debug-level`'s `VLLM_USE_BREAKABLE_CUDAGRAPH` forcing eager execution).

**Note on `disable-flashinfer-env` itself**: the log evidence shows a v0.26.0-compatible version of that patch (or an equivalent hand-edit) is already active on the cluster's installed venv (`IVLLM_DISABLE_FLASHINFER=1 — skipping flashinfer fused allreduce+RMSNorm...` appears in `logs/mmax3q/*/vllm.0.log`), but only a `v0.25.1` diff is tracked in this directory — no `v0.26.0` version has been committed here. Worth reconciling so the tracked diff matches what's actually deployed.

### `diffs/disable-flashinfer-import.v0.26.0.v2.patch`

**Supersedes `v1` above** — decouples the two questions v1 conflated: "should *this process* ever attempt `import flashinfer`" from "should *Workers* use the fused allreduce+RMSNorm kernel." Rob's framing (2026-09-03): *"if we enable flashinfer we need to be explicit about which path it should take"* — the real problem was never "flashinfer on vs. off," it's that `EngineCore` accidentally imports it for a reason that has nothing to do with the fused kernel decision at all (a multimodal-support check during model-class resolution — `EngineCore` never calls anything else in either patched file, confirmed by tracing every use of `flashinfer_comm` in `allreduce_rms_fusion.py`: they're all inside function bodies or a `if flashinfer_comm is not None:` guard, never touched unless the fused kernel actually runs).

v2 adds a second, independent guard: detect whether this is `EngineCore`'s own process via `setproctitle.getproctitle()` (vLLM itself calls `set_process_title("EngineCore"/"EngineCore_DP<n>")` early in `run_engine_core`, and `set_process_title(name="Worker")` early in worker startup — both calls happen well before either patched file is ever imported, and the resulting process names are exactly what showed up as `Process 48317: VLLM::EngineCore` / `Process 48600: VLLM::Worker_TP0_EP0` in the `py-spy` dumps that diagnosed the original hang, so this isn't speculative — it's the same signal already observed working). `setproctitle` import itself is wrapped in `try/except ImportError`, falling back to "not EngineCore" (i.e. old v1 behavior, gated only by `IVLLM_DISABLE_FLASHINFER`) if the package is ever missing.

Net effect:
- `EngineCore`: **never** imports flashinfer, unconditionally — it has no legitimate use for it, so there's no tradeoff here at all.
- Workers: import flashinfer (and use the real fused kernel) **unless** `IVLLM_DISABLE_FLASHINFER=1` is explicitly set — restoring the flag to its original, correctly-scoped meaning ("this specific job's topology has no working flashinfer allreduce backend," per `knowledge-base.md`'s `multinode`-only scoping), rather than "avoid a hang that has nothing to do with topology."

Practical consequence for the MiniMax-M3 job that surfaced all this: it's single-node (TP=4, one node) — `knowledge-base.md` already documents `IVLLM_DISABLE_FLASHINFER=1` as a *multinode-only* workaround (`trtllm`/`mnnvl` backends are viable node-local). So the recommended next test is: apply `v2` (drop `v1`), **remove `IVLLM_DISABLE_FLASHINFER` from this job's yaml entirely** (or set `VLLM_FLASHINFER_ALLREDUCE_BACKEND=trtllm` explicitly per Rob's "be explicit about which path" instinct, rather than relying on auto-detection), and confirm both that the hang stays fixed and that generation becomes coherent.

**Status: CONFIRMED WORKING LIVE, 2026-09-03.** Applied to the real venv, tested against a full run (`logs/mmax3q/20260903_213700/` and the run after it) — `EngineCore` no longer hangs, and (combined with `minimax-m3-indexer-unfuse.v0.26.0.v1.patch` below) the job serves coherent responses. `v1` should now move to `old/` — repeated `mv` attempts this session were blocked by tool-permission denials; still pending, not a correctness issue.

### `diffs/minimax-m3-indexer-unfuse.v0.26.0.v1.patch`

**Investigated 2026-09-03**: Rob asked whether the `toncao/vllm@minimax-m3-compressed-tensors` fork branch (referenced in `examples/minimax-m3-int4.yaml`'s comments, needed for `cyankiwi/MiniMax-M3-AWQ-INT4`) could become a tracked, versioned patch against `vendor/vllm-0.26.0` instead of a manual "clone this fork and build a separate venv" step. Findings:

- **The fork is python-only and small**: 3 commits on top of upstream `a7fdfeef7` (2026-06-16), touching 9 files, +134/-29 lines total. Genuinely implementable as a diff, not a full alternate install (unlike `examples/solar-open2-250B.yaml`, which needs an entirely separate Upstage fork/venv — a heavier case the same overlayfs idea would also need to handle, see below).
- **The fork's base is 1242 commits behind `v0.26.0`** (`a7fdfeef7` is a genuine ancestor of the `v0.26.0` tag, confirmed via `git merge-base --is-ancestor`) — six weeks of upstream churn. Despite that gap, most of the fork's actual changes turned out to be **already present upstream in `v0.26.0`, independently**: the `gemm1_alpha`/`gemm1_beta`/`gemm1_clamp_limit` plumbing through `make_wna16_moe_quant_config()` and every quant-method's `get_fused_moe_quant_config()` (`fused_moe/config.py`, `fused_moe/oracle/int_wna16.py`, `quantization/moe_wna16.py`, `compressed_tensors_moe_wna16_marlin.py`) is byte-for-byte functionally identical in `v0.26.0` already — the only "conflict" 3-way-merging the fork's diff onto `v0.26.0` surfaced there was keyword-argument *ordering*, a pure no-op. `awq_marlin.py` (which the fork also touches) no longer exists in `v0.26.0` at all — `AWQMarlinMoEMethod` was renamed to `AutoAWQMoEMethod` and moved into `auto_awq.py` during a quantization-code reorg, and **that file already has the identical plumbing too**. None of this needs patching.
- **The one genuinely still-needed piece**: the "indexer un-fuse" change in `vllm/models/minimax_m3/{nvidia,amd}/model.py`. Stock `v0.26.0` uses a single fused GEMM (`MinimaxM3QKVParallelLinearWithIndexer`) for q/k/v *and* the sparse lightning-indexer's `index_q`/`index_k` projections together. Per the fork's own commit message: *"many checkpoints leave the indexer in bf16 while quantizing q/k/v, which a single packed GEMM can't represent."* The fork instead runs q/k/v through a normal (quantizable) `QKVParallelLinear` and the indexer through two standalone unquantized `ReplicatedLinear`s, re-concatenating into the layout the fused attention kernel expects. Reconciling this against `v0.26.0` needed exactly one trivial fix (both files import `get_pp_group` now, which the fork's older base didn't have yet — merged to import both that and the fork's new `get_tensor_model_parallel_rank`); the ~35-line substantive change per file applied without conflict. Packaged as `minimax-m3-indexer-unfuse.v0.26.0.v1.patch`, verified to apply cleanly (`git apply --check`) and round-tripped (apply for real → confirm `ReplicatedLinear` present → `git apply -R` → confirm pristine) against `vendor/vllm-0.26.0`.
- **This is a strong, independent lead on the MiniMax-M3-AWQ-INT4 garbage-output bug** (see `design/active-issues.md`) — quantizing q/k/v to INT4 while leaving the indexer in bf16 is exactly `cyankiwi/MiniMax-M3-AWQ-INT4`'s situation, and a single fused GEMM mishandling that split would corrupt the indexer's weights/output, feeding wrong block-selection scores into the model's custom sparse-attention Triton kernels (`_topk_index_kernel`, `_index_block_score_kernel` — observed being freshly JIT-compiled during this exact model's serving in `logs/mmax3q/20260903_161157/`) — coherent-looking API responses, numerically wrong content. Not yet tested live; that's the next step, and it's independent of the `disable-flashinfer-import` investigation above (different files, different mechanism).
- **The fork's other two commits (defaulting `block_size` to 128 globally) don't need a source patch at all** — `block-size: 128` already works as an ordinary per-job yaml key (`examples/minimax-m3-int4.yaml` already sets it, and it's already in this job's own config), which is strictly better than a global vLLM default change: scoped to the one job that needs it instead of silently changing behavior for every other model sharing the venv.

**Status: CONFIRMED WORKING LIVE, 2026-09-03.** Applied to the real venv alongside `disable-flashinfer-import.v0.26.0.v2.patch` — `logs/mmax3q/20260903_213700/` and the run after it confirm the model loads and produces coherent responses (Rob: "creates sensible responses"). Checkpoint-shard loading was slow on this run but that's diagnosed as transient shared-storage contention, not a patch issue (see `design/active-issues.md`). Follow-up found on this same run: the `minimax_m3` reasoning parser (thinking/`<mm:think>` blocks) isn't parsing correctly, though tool-calling works — a separate issue, not yet root-caused, tracked in `design/active-issues.md`.

### `diffs/solar-open2-support.v0.26.0.v1.patch`

**Investigated and drafted 2026-09-03**, same treatment as the MiniMax-M3 case above: Rob asked whether `examples/solar-open2-250B.yaml`'s requirement for the `UpstageAI/vllm@v0.22.0-solar-open2` fork (previously assumed to need an entirely separate install, since it's a different vLLM *version*, not just a patched file) could instead become a tracked diff against `vendor/vllm-0.26.0`.

- **The raw branch-vs-merge-base diff looks alarming but is almost entirely noise.** The fork branch has 20 commits; `git log` shows 19 of them are ordinary upstream commits (DeepSeek-V4/GDN/ROCm bugfixes, one of which touches a `.cu` CUDA kernel file) that happen to sit on this branch but have nothing to do with solar-open2 — the branch was evidently kept in sync with upstream by cherry-picking/rebasing rather than branching once and staying still. Diffing the whole branch against its merge-base (1967 commits behind `v0.26.0`) gives a misleading 86-file, +9429-line, CUDA-kernel-touching picture.
- **Isolating the single real commit** (`00907fc9b`, `"Support solar-open2"`) gives the true picture: 20 files, but ~2600 lines are tests and ~590 lines are MoE-kernel autotuning JSON tables (runtime tuning data for the model's specific expert-count/hidden-dim shapes — kept in the patch since they're genuine production data, not benchmark scripts). The real code is almost entirely **brand-new, self-contained files** (the model itself, its tool parser, reasoning parser, config class, logits processor) that can't conflict with anything since nothing else imports them, plus 1–4 line registrations in `models/registry.py`, `reasoning/__init__.py`, `tool_parsers/__init__.py`, `transformers_utils/config.py`, `transformers_utils/configs/__init__.py`.
- **One file has a real, non-trivial change to code shared with other models**: `vllm/model_executor/layers/mamba/gdn/kimi_gdn_linear_attn.py` (`KimiGatedDeltaNetAttention`, also used by Kimi-family GDN models). It's purely additive — fills in a previously-empty "V1 profile run" branch with a kernel-warmup call (`_warmup_prefill_kernels`, guarded by `try/except`, explicitly modeled on an existing analogous `QwenGatedDeltaNetAttention._warmup_prefill_kernels`) that force-autotunes the chunked-prefill KDA Triton kernels while GPU memory is plentiful, avoiding a TP-rank desync/collective-abort on first real request under CUDA forward-compat. Same "harmless bolt-on" shape as MiniMax-M3's fix — no existing logic is modified, only a dead branch is filled in.
- **Reconciling against `v0.26.0` needed exactly 2 trivial conflicts**, both "another entry was added to this dict/list since the fork's base, so add ours alongside it": `models/registry.py` (add `"SolarOpen2ForCausalLM": (...)`) and `reasoning/__init__.py` (add the `"solar_open2"` entry next to a `"inkling"` entry that didn't exist yet at the fork's base). Everything else applied clean or as plain new-file adds. Packaged as `solar-open2-support.v0.26.0.v1.patch` (3683 lines, mostly the 5 new files), verified to apply cleanly (`git apply --check`) and round-tripped (apply for real → confirmed `SolarOpen2ForCausalLM`/`solar_open2` registrations landed → `git apply -R` → confirmed pristine) against `vendor/vllm-0.26.0`.
- **The `.cu` CUDA kernel file and other DeepSeek-V4-only files are correctly excluded** — they belong to the unrelated upstream commits riding on the branch, not to solar-open2's own commit, and aren't in this patch at all. Confirms this is genuinely a python-only, no-recompile patch for solar-open2 specifically.

**Status (2026-09-03)**: patch drafted and verified as above. Not yet applied to the real venv or tested against a live job (no live `solar-open2-250B` job has been run yet to confirm against).

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
