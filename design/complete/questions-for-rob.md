# Questions for Rob

**Status: fully answered and processed, 2026-09-03.** Rob's inline `RESPONSE:`
answers below have all been folded into their permanent homes — mainly
`active-issues.md`'s "Remaining answers from Rob's review" entry,
`knowledge-base.md`'s per-flag notes, `backend-contract.md` (abort/warmup),
`design/complete/bench-implementation-plan.md`, and `design/priorities.md`
(the new network-benchmark-harness item). Kept here as the raw Q&A record —
not the place to look for current state, that's the docs above.

Collated during the 2026-09-03 documentation-drift review (post-NCCL-fix). These
are things I found where the *fact* is clear but the *right response* is a
judgment call that isn't mine to make — per instructions, no production code
changes this session, so these are queued rather than acted on. Everything I
was confident about instead, I folded directly into `active-issues.md` or
`knowledge-base.md`.

## Testing

- **`tests/bash/sandboxed/test-wait-report.sh` tests a function that no longer
  exists.** `wait_report()` has been fully replaced by `monitor_node()`
  everywhere it's called (`run-head-vllm.sh`, `run-worker-vllm.sh`) — grepping
  `utils.sh` for `^wait_report()` finds nothing. This test file still calls
  `wait_report "wr-job" "$pid"` directly and will fail outright (function not
  found), not because of a behavior regression but because it's testing
  something that was intentionally renamed away. Its own comments describe a
  real, worth-preserving contract though: "exit-code propagation" — the
  monitored process's death should propagate through as the wrapper's own
  exit status, and an unsolicited *clean* exit (code 0) should still be
  treated as a failure. Does `monitor_node()` (much more involved now — it
  also does trigger-watching, py-spy/cuda-gdb dumps, etc.) still uphold that
  same exit-code contract, or has it changed? If it still holds, this test
  should be rewritten against `monitor_node()`; if the contract genuinely
  changed, the test's assertions need to change too, not just the function
  name it calls. I didn't want to guess at `monitor_node()`'s intended
  contract and rewrite this myself.

RESPONSE: Yes it still waits on and returns the main process exit code. The main
process is going to be ether vllm head itself or the ray head node process, so
there is some uncertainty. Vllm exiting "normally" is treated as an error.

## Config knobs (see the "Which config knobs..." addition to the resolved
GLM-5.2 entry in `active-issues.md` for the full reasoning) — flagging here
too since testing removal is a real action, not just documentation:

- `numa-bind: true`, `no-async-scheduling: true`, `disable-custom-all-reduce:
  true`, and `compilation-config: '{"pass_config": {"fuse_allreduce_rms":
  false}}'` in `glm-5.2-743b-int4.yaml` were all added while chasing the NCCL
  bug (or, for `fuse_allreduce_rms`, appear to target a code path GLM-5.2's
  own `IVLLM_DISABLE_FLASHINFER` handling already covers instead). None were
  ever shown necessary for GLM-5.2 specifically. Worth testing removal as a
  batch against the now-working NCCL 2.30.4 baseline — I didn't do this
  myself since it means running a real job, which is out of scope for a
  documentation-only session.

RESPONSE: please document uncertainty against these flags in the knowledge base.
On the if it isn;t borken don;t fix it philosophy I will likely leave it as in
in the current config.

## Small code smells found in `utils.sh` (documented as Known Issues, not
fixed — see `active-issues.md`'s new "Four small, low-severity bugs..." entry
for full detail) — listed here too since they're small enough you might just
want to fix them directly rather than treat as a backlog item:

- `node_hang_detector()` sleeps twice per loop iteration (looks like an
  accidental duplication).

RESPONSE: this was an intentional hack to work around the issue that it was
occasionally firing during setup. Have noted.

- `report_processes()` references an undeclared `$debug_level` in one log
  message (prints blank).

FIXED

- `semver_sort`/`rev_semver_sort` are named backwards from what they do
  (functionally fine at their one call site, but a trap for future readers).

RESPONSE: Comments amke it clear. Depends on how you interpret the natural order
of versions which in my case is most recent first.

## Urgent — env defaults

- **Was re-enabling `NCCL_GDRCOPY_ENABLE`/`FI_HMEM_CUDA_USE_GDRCOPY` in
  `vllm-env.sh` (commit `244246c`) a deliberate reopening of the "closed,
  wrong tree" GDRCopy investigation, or did it get carried over by copying
  the yaml's old experimental `env:` block into the shared default without
  noticing GDRCopy had since been separately ruled out?** This is the single
  highest-priority item in this whole review — see the new top entry in
  `active-issues.md`'s Known Issues. If deliberate, `knowledge-base.md`'s
  Mandatory table needs its "never" verdict updated and the libfabric
  double-free risk ([ofiwg/libfabric#10041](https://github.com/ofiwg/libfabric/issues/10041))
  re-assessed; if accidental, it should probably be reverted.

RESPONSE: This changed in Isambard containers project to enabled (https://raw.githubusercontent.com/UKGovernmentBEIS/isambard_containers/refs/heads/main/definitions/vllm/vllm.def) and so started experimenting
I then found this: https://raw.githubusercontent.com/HewlettPackard/shs-nccl-env/refs/heads/main/src/shs_nccl_env.cc / https://github.com/HewlettPackard/shs-nccl-env which
validated using GDRCopy. It is unclear to me what the best option is. The comment
"GDRCopy is not needed with vLLM - vllm bypasses NCCL for intranode comms and between nodes is using RDMA over slingshot." Is at odds with the fact that NCCL versions have
such fundamental effects and I dont know its provenance. I think the only
option is to do a network test with these enabled/disabled and see whether it
makes any difference. The slingshot tp reprex prototoype is a basis for doing
this I think as future optimisation.

- **Was `VLLM_COMPILE_CACHE_SAVE_FORMAT` switched from `"unpacked"` to
  `"binary"` (`vllm-env.sh`) specifically to fix the "NCCL version issue," or
  unrelated/accidental?** No comment either way, and it contradicts
  `knowledge-base.md`'s Mandatory-table value.

RESPONSE: This was changed in investigating the cache bloat. Docs state:
- "binary": saves as binary file
     Safe for multiple vllm serve processes accessing the same torch compile cache.
- "unpacked": saves as directory structure (for inspection/debugging)
     NOT multiprocess safe - race conditions may occur with multiple processes.

- **Is the dangling `VLLM_NCCL_SO_PATH` comment block in `common-env.sh`** (describes
  the fix pattern — point at a known-good NCCL `.so` — but never actually sets
  the variable) a placeholder for unfinished work, or a note you decided
  against and should just be deleted? Is `github.com/NVIDIA/nccl/issues/1234`
  the real issue number or a placeholder?

RESPONSE: placeholder for investigation. The mechanism exists in the documentation
with the comment:
Path to the NCCL library file. It is needed because nccl>=2.19 brought
by PyTorch contains a bug: https://github.com/NVIDIA/nccl/issues/1234
maybe worth exploring in the VLLM codebase.

- **Does the new `LD_PRELOAD=".../libcudadebugger.so.1"` (`common-env.sh`)
  actually get honored**, given `active-issues.md` already found `LD_PRELOAD`-ing
  an alternate NCCL library does *not* override PyTorch's own absolute-path
  load? Is `libcudadebugger.so.1` known to behave differently, or untested
  against the same failure mode?

RESPONSE: The cuda-gdb / coredump mecahnism required this not sure how or
whether this got validated. Not broken don't fix.

- **Is `FI_CXI_DEFAULT_RX_SIZE=2048` (new, `vllm-env.sh`)** meant to share the
  same "HPE vendor default" justification as its documented sibling
  `FI_CXI_DEFAULT_TX_SIZE`, or an independent decision needing its own
  `knowledge-base.md` note?

RESPONSE: have removed but added in HP slingshot env defaults for FI_CXI_RDZV_EAGER_SIZE
which we have seen has no particular effect. Again needs a network testing harness

- **The TODO in `common-env.sh` pasting raw `"Could not find: libnccl-env.so"` /
  `"libnccl-profiler.so"` NCCL log lines** — an open, tracked problem, or
  known-safe noise (optional plugins this environment doesn't have) that's
  safe to delete?

RESPONSE: I think these are noise. NCCL profiler woudl be potentially useful,
libnccl-env.so coudl be using HP

## Orchestration / setup scripts

- **Was dropping `#SBATCH --reservation=interactive` from `slurm-vllm-setup.sh`
  deliberate** (e.g. to stop the up-to-3-hour setup job from consuming the
  interactive reservation's limited concurrent-job slot), or incidental to
  some other edit?

RESPONSE: architectural. All partition decisions made by scripts that call
slurm scripts (and not the scripts themselves), so that it can be affected by
CLI flags. Setup as it happens always uses interactive partition when called
through CLI.

- **Confirmed by me, not a question**: moving the NCCL pin in
  `slurm-vllm-setup.sh` to the very end of the file (after DeepGEMM/UCCL-EP/
  NIXL/humming/hpc-ops) was deliberate — I proposed it earlier this session
  specifically to stop those later `uv pip install` steps from re-resolving
  torch's bundled (unpinned) `nvidia-nccl-cu12` and clobbering the pin, and
  it's confirmed working (see the resolved GLM-5.2 entry). Already updated
  `active-issues.md`'s Nemotron entry to match — no action needed here.

RESPONSE: Agreed

- **`ray-setup.sh`'s commented-out `#--include-dashboard=false`** — keep as a
  documented "known lever" (worth a comment saying so), delete as a leftover,
  or actually enable it?

RESPONSE: Tried to disable it but Ray crashed (not sure whether that was incidental)
at some point I woudl like to retest but low priority

- **`src/engine/ivllm-bench.sh` still opens with "PROTOTYPE, not
  production... must be completely rewritten" per `AGENTS.md`, but has been
  living in `src/engine/` (not `design/`) since commit `df67993` and is
  already called directly in production** (`IsambardBareMetalBackend.ts`,
  wired into the real `bench submit|status|results` CLI). Is this file
  considered production-ready as-is, or does the rewrite pass the header
  promises still need to happen? Either way the header is misleading as it
  stands.

RESPONSE: Have update the header. Its been reviewed. It shoudl work, but not yet
end to end tested. It is experimental lifecycle

- **Is `is_startable()` in `utils.sh`** (unused, duplicates `ivllm-serve.sh`'s
  own inline status-guard block almost line-for-line) meant to eventually
  replace that inline block, or was it written for a different, not-yet-built
  caller? The two have already drifted once (see Known Issues).

RESPONSE: Needs refactoring.

- **Resolved by me, not a question**: `IVLLM_PARTITION`'s current default in
  `ivllm-serve.sh` (`--partition=interactive --reservation=interactive`) is
  the correct value to ship, per `active-issues.md`'s own already-documented
  finding that the `interactive` reservation is `Hidden=YES` and
  `--reservation=interactive` alone isn't sufficient for job visibility. The
  stale `# --partition=interactive ... seems unnecessary?` comments left
  behind in `ivllm-get-model.sh` and `ivllm-setup.sh` (predating the flag
  being correctly re-added) are just leftover and safe to delete whenever
  convenient — not a real question, just noting I didn't delete them myself
  since it's a (trivial) production-code change.

RESPONSE: Keeps changing on isambard. Current state is both flags are needed.

- **`request_cancel()`'s `pending`-job path calls `tidy_up` directly from
  whatever process called it** (e.g. the login-node `ivllm-cancel.sh`
  itself), since no compute-side monitor exists yet for a job that never
  started. Is that intentional — i.e. is `tidy_up` allowed to run from the
  login node in this one specific case — or should `architecture.md`'s
  "`tidy_up` is always compute-side" framing (implied by its lifecycle table)
  be corrected to carve out this exception explicitly?

RESPONSE: NOthing in tidy_up's code says it must be run on compute node - only
node specific action is clearing the localdir (which is probably not in the
right place anyway - and probably only works for head node of a cluster)

- **Should `backend-contract.md` §2.5 (`requestCancel`) be updated to include
  the `abort` parameter**, matching `architecture.md`'s `Backend` interface
  and the real `ivllm-cancel.sh -a` behavior? Right now `backend-contract.md`
  only documents a two-parameter signature and doesn't mention `abort` at
  all — either it predates the abort feature and needs updating, or it's
  considered a frozen/legacy doc superseded by `architecture.md` for this
  detail, in which case worth a note saying so.

RESPONSE: Yes abort and warmup

- **Is `design/complete/bench-implementation-plan.md`** (still headed
  "Status: Draft, unimplemented," describing the prototype→production move as
  future work that has already happened) meant to be archived/marked
  complete now, or is there remaining scope in it that's still genuinely
  open?

RESPONSE: E2E testing left

## Testing infrastructure

- **`tests/bash/sandboxed/test-ivllm-bench.sh`'s 5 test cases source
  `/work/prototype/ivllm-bench.sh`**, bound by the sandbox harness to
  `design/prototype/ivllm-bench.sh` — deleted in commit `7ce59a7` after the
  file was promoted to `src/engine/ivllm-bench.sh`. All 5 cases fail
  (verified by running them: `source: ... No such file or directory`). Was
  this noticed and just not yet fixed, or has this test suite silently not
  been run since the promotion?

RESPONSE: Needs updating to reflect promotion

- **Update: fixed, 2026-09-03** — see `active-issues.md`'s "Test suite: fix the stale tests... done" entry for the full list of what changed. 5 stale files fixed and passing; `test-diagnostics.sh`/`test-lockfile.sh` correctly left failing (real regressions); `test-login-handoff.sh` untouched (environment issue). Original findings kept below for reference.
- Confirmed by running the suites this session: **8 of 12 bash sandboxed test
  files currently fail** (`test-wait-report.sh` ×2, `test-ivllm-bench.sh` ×5,
  `test-exit-trap.sh` ×1 [stale string assertion, "SLURM timeout" →
  "SLURM job cancelled"], `test-monitor-head.sh` ×3 [idle-timeout mechanism
  and `run_vllm_warmup`'s new timing both moved since these were written],
  `test-report-memory.sh` ×1 [tests old monolithic `report_memory()` that no
  longer exists in that form], `test-diagnostics.sh` ×1 [real regression, see
  Known Issues], `test-lockfile.sh` ×1 [real regression, see Known Issues]).
  `test-login-handoff.sh`'s `login_force_cancel` fails only because this
  sandbox runs as a non-root user, not a code issue. None of these were
  fixed this session (no test-file changes made, per instructions) — this is
  a full accounting of current test-suite health, in case it's useful before
  deciding what to tackle first.

RESPONSE: Fix stale tests. Leave real regressions failing.

Files/commits referenced throughout: `common-env.sh`, `vllm-env.sh`,
`ray-setup.sh`, `slurm-ray-vllm-serve.sh`, `slurm-vllm-setup.sh`,
`ivllm-serve.sh`, `ivllm-bench.sh`, `ivllm-cancel.sh`, `ivllm-get-model.sh`,
`ivllm-setup.sh`, `utils.sh` — mostly commit `244246c` ("Working glm52q due
to NCCL version issue") and `d4d0700` ("refactor debugging"). Full per-file
detail available in this session's workflow transcript if needed —
ask and I can expand any item above.
