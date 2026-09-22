# `patches/` — local patches to installed third-party packages

Patches here modify files inside an already-installed vLLM venv on the HPC
(`$PROJECTDIR/engine/vllm/<version>/lib/python3.12/site-packages/...`), not
anything in this repo. They exist because a fix needs to land before an
upstream PR merges/ships in a tagged release, or because the fix is a local
workaround that shouldn't be a permanent part of `slurm-vllm-setup.sh`. See
`design/active-issues.md` for the full background/investigation behind each
one — this file is the index and the "how to use it" reference.

**Origin**: promoted from `design/prototype/patch/` (its `apply-vllm-patch.sh`
+ `diffs/*.patch`), which is now superseded — the CLI (`ivllm patch`) and this
directory are the production form. The old prototype directory is kept for
its investigation history but should no longer be used for new patches.

## Applying a patch

```bash
ivllm patch <vllm-version> <patch-file>              # apply
ivllm patch <vllm-version> <patch-file> --revert     # revert
```

`<vllm-version>` must already be installed via `ivllm setup <version>`.
`<patch-file>` is a local path to one of the `.patch` files below (or a new
one you've drafted) — the CLI copies it to the login node and runs
`ivllm-patch.sh` (`src/engine/ivllm-patch.sh`) against the installed venv.

**Idempotent in both directions** — applying an already-applied patch, or
reverting one that isn't applied, is a clean no-op rather than an error
(checked via a dry-run in both directions before touching anything). A
manifest of currently-applied patches is kept at
`<vllm_version_dir>/.ivllm-patches-applied` (one basename per line), outside
`site-packages` so it's clearly ivllm's own bookkeeping, not vLLM code.

## Naming convention

**`<descriptive-name>.v<vllm-version>.v<revision>.patch`** — every patch file
is versioned by filename, not edited in place. When a patch's content needs
to change, save it as a new file with the revision number bumped
(`...v1.patch` → `...v2.patch`) rather than overwriting the existing one.
Superseded revisions and fully-retired patches move to an `old/`
subdirectory (kept for reference, not deleted) rather than being removed
outright.

**Never edit an already-applied patch file in place.** This is genuinely
dangerous, not just a style preference: the dry-run idempotency check only
compares the *current* file content against the *current* venv state, with
no way to know an *older* revision of that same file is what's actually
installed. If new content partially overlaps old (some hunks match, some
don't), *both* the forward and reverse dry-runs can fail, and the tool falls
back to reporting "already applied — nothing to do" and exits without
applying anything — silently leaving the venv in whatever incomplete state
the old revision left it in, with no error. `ivllm-patch.sh` cross-checks the
`.ivllm-patches-applied` manifest for *other* revisions of the same base name
and warns if one's still recorded as applied, but reverting the old revision
before applying a new one remains the correct sequence — the warning is a
safety net, not a substitute.

## Producing a patch file

A unified diff, git-style `a/`/`b/` paths, rooted at the vLLM package's
`site-packages` directory (e.g. a path of `vllm/model_executor/layers/foo.py`
inside the diff), applied via `patch -p1`:

```bash
diff -u original.py patched.py \
  | sed 's|^--- original.py|--- a/vllm/path/to/original.py|;
         s|^+++ patched.py|+++ b/vllm/path/to/original.py|'
```

Or, more commonly for this project: check out the relevant vLLM version as a
git worktree from `vendor/vllm` (`git worktree add --detach <dir> v<version>`)
and hand-edit or `git diff` against it, converting the result to the `a/`/`b/`
form above.

**Before trusting a patch against a version it wasn't originally written for**,
verify it with a real dry-run rather than assuming it still applies:

```bash
cd <a throwaway worktree of the target vLLM version>
patch -p1 --dry-run -i /path/to/the.patch
```

Hunks can fail even when nothing about the *logic* changed — vLLM's own code
around the patched lines shifts between versions (new imports added above,
an adjacent function changed) often enough that a patch drafted for one
version needs a small update (not necessarily a rewrite) for the next. See
the `disable-flashinfer-unified` history below for a concrete example of
exactly this happening twice in a row (v0.26.0 → v0.29.0 → v0.29.1rc0).

## Current patches

### `disable-flashinfer-unified.v{0.26.0,0.29.0,0.29.1rc0,0.30.0}.v1.patch`

**What it does**: two independent guards around `import flashinfer.comm` in
`vllm/compilation/passes/fusion/allreduce_rms_fusion.py` and
`vllm/distributed/device_communicators/flashinfer_all_reduce.py`, plus a
runtime early-return in `_can_use_flashinfer()`
(`vllm/model_executor/layers/fused_allreduce_gemma_rms_norm.py`):

1. **Always** skip the flashinfer import in `EngineCore`'s own process
   (detected via `setproctitle.getproctitle()`) — it never legitimately uses
   either patched module, and importing it there triggers flashinfer's own
   eager `torch.cuda.get_device_capability()` probe in a process with no CUDA
   context yet, which can hang indefinitely.
2. Skip the import in Worker processes too, and skip using the fused
   allreduce+RMSNorm kernel entirely, **only when `IVLLM_DISABLE_FLASHINFER=1`**
   is set in the job's yaml `env:` block — opt-in per job, doesn't affect
   other jobs sharing the same venv.

See `skills/generate-vllm-config/SKILL.md`'s "FlashInfer on Isambard" section
for when to actually set that env var (genuine multi-node topologies with no
working flashinfer allreduce backend — not a default "just in case").

**Version history**: drafted and confirmed working live against `v0.26.0`
(2026-09-03, alongside `minimax-m3-indexer-unfuse` below — see
`design/active-issues.md`). Ported to `v0.29.0` and `v0.29.1rc0` (2026-09-14/15)
— each port needed one small hunk update, not a rewrite: v0.29.0 changed the
trailing `except ImportError: pass` in `allreduce_rms_fusion.py` to
`except Exception as e: logger.debug_once(...)` (our patch now leaves that
line as untouched context instead of rewriting it, matching whichever
upstream version is installed); v0.29.1rc0 additionally added a new
`from vllm.platforms import current_platform` import line just above our
insertion point in `fused_allreduce_gemma_rms_norm.py` (AITER/ROCm support,
unrelated to this patch — just needed as extra context). Ported to `v0.30.0`
(2026-09-22, skipping the `0.29.1` final tag which was never cut) — no content
changes needed at all, byte-identical to the `v0.29.1rc0` variant; `patch -p1`
applies clean (only the same benign fuzz/offset already seen on `v0.29.1rc0`).
All four variants verified with a real `patch --dry-run` + full apply against
a fresh worktree of the respective vLLM tag, not just visual inspection.

### `minimax-m3-indexer-unfuse.v0.26.0.v1.patch`

Un-fuses the sparse lightning-indexer's `index_q`/`index_k` projections out
of `MinimaxM3QKVParallelLinearWithIndexer`'s single packed GEMM
(`vllm/models/minimax_m3/{nvidia,amd}/model.py`) — needed because that fused
GEMM can't correctly represent a checkpoint that quantizes q/k/v to INT4
while leaving the indexer in bf16 (`cyankiwi/MiniMax-M3-AWQ-INT4`'s exact
situation). Confirmed working live, 2026-09-03, alongside
`disable-flashinfer-unified` above. See `design/active-issues.md` for the
full investigation (this fixed a real garbage-output bug, not just a
startup crash).

### `solar-open2-support.v0.26.0.v1.patch`

Backports `zai-org`'s "Support solar-open2" upstream commit (model class,
tool/reasoning parsers, config registration — almost entirely new,
self-contained files, so low conflict risk) onto `v0.26.0`, avoiding the need
for an entirely separate fork/venv for `examples/solar-open2-250B.yaml`.
Drafted and verified to apply cleanly (`git apply --check`, round-tripped);
**not yet tested against a live job** as of this writing — no live
`solar-open2-250B` run has happened yet to confirm against. See
`design/active-issues.md` for the full investigation, including why the
raw fork-branch diff looks alarming (86 files, +9429 lines) but the real,
isolated commit is much smaller and safer than that headline number implies.

## Older, prototype-era patches (`design/prototype/patch/diffs/`)

Kept for their investigation history, not part of the production patch flow
above. Includes `hybrid-trtllm.sh` (a one-off script, pre-dating the generic
`patch`/`diffs` approach) and several superseded/archived diffs under its own
`diffs/old/` — see that directory's own `README.md` for details on each.
