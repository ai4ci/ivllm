# Ivllm environment

This document describes the paths and configuration environment variables that
are set in ivllm and perform functional roles such as compiler paths and are
set either by the brics nccl, libfabric modules themselves or by
[common-env.sh](src/engine/lib/common-env.sh).

They assume the naming conventions in the [resolve functions](src/engine/lib/utils.sh)

e.g. `$nvhpc_dir` in this document is the same as a call to `resolve_nvhpc_dir` in `utils.sh`
and `$IVLLM_PROJECTDIR` is set and is the project root. `$HOME` for user home

Table schema, one row per variable:
* **Variable / flag**
* **Value** — the actual value or expression it resolves to
* **Evidence** — a short tag, elaborated in a free-form note below the table
  only when it needs more than that:
    - proven (known failure without this value)
    - recommendation (Isambard or GH200 specific: Brics, isamabrd containers, doubleword.ai)
    - documentation (official documentation only)
    - recipe (vllm recipes or cookbooks - non GH200 specific)
    - inferred (web searches)

References:
* https://docs.nvidia.com/hpc-sdk/installation-guide/index.html#end-user-environment-settings

Scope note: `common-env.sh` is sourced before the vLLM venv is even known
to exist — everything here should be independent of which vLLM version or
model is being run. Anything that varies per-job (NCCL/CXI tuning, vLLM
runtime flags) lives in `vllm-env.sh` and is documented in
`knowledge-base.md` instead.

## Compiler / library paths

Module loads (`module purge` first, so this is the *complete* module set,
not additive to whatever the login shell already had):

| Module | Evidence |
|--------|----------|
| `brics/default` | recommendation |
| `brics/userenv` | recommendation |
| `brics/nccl` | proven |
| `libfabric` (resolves to `1.22.0`) | proven |
| `gcc-native/13.2` | proven |

Notes:
- `brics/userenv` sets `$LOCALDIR`/`$SCRATCHDIR` — see "Cache setup" below,
  relationship to ivllm's own `$LOCALDIR` override is an open question, not
  yet reconciled (inferred).
- `brics/nccl` loads the NCCL + `aws-ofi-nccl` plugin build onto
  `LD_LIBRARY_PATH`. The Nemotron multi-node hang investigation
  (`active-issues.md`) root-caused a real hang to the NCCL *build*
  mattering, not just the version — this module is how the known-working
  build gets picked up (proven).
- `gcc-native/13.2` is required by DeepEP when compiling against CUDA 12.9,
  per the source comment (proven — a specific, named failure mode).

Compiler selection and flags:

| Variable | Value | Evidence |
|----------|-------|----------|
| `CC` | `$(which gcc)` | proven |
| `CXX` | `$(which g++)` | proven |
| `CFLAGS` | `-mcpu=neoverse-v2 -mtune=neoverse-v2 -O3` | recommendation |
| `CXXFLAGS` | `-mcpu=neoverse-v2 -mtune=neoverse-v2 -O3` | recommendation |
| `TORCH_CUDA_ARCH_LIST` | `9.0a` | documentation |
| `NVCC_APPEND_FLAGS` | `-arch=sm_90a` | documentation |

Notes:
- `CC`/`CXX` resolve *after* the module loads above, so they pick up
  `gcc-native/13.2`.
- `-mcpu=neoverse-v2 -mtune=neoverse-v2 -O3` is cited against a specific
  benchmark: [openbenchmarking.org GH200 result](https://openbenchmarking.org/result/2402098-NE-NVIDIAGH291&sor&sgm=1).
- `9.0a`/`sm_90a` is the documented compute-capability string for
  Hopper/GH200.

Compilation resource limits (GH200-specific tuning, not general vLLM
advice):

| Variable | Value | Evidence |
|----------|-------|----------|
| `OMP_NUM_THREADS` | `16` | recommendation |
| `TORCHINDUCTOR_COMPILE_THREADS` | `4` | recommendation |
| `VLLM_USE_PRECOMPILED` | `1` | documentation |
| `MAX_JOBS` | `8` | recipe |
| `NVCC_THREADS` | `4` | recipe |

Notes:
- `OMP_NUM_THREADS=16`/`TORCHINDUCTOR_COMPILE_THREADS=4`: GH200 has 72
  cores; sized so 4 concurrent compiling workers (one per GPU) don't
  oversubscribe.
- `MAX_JOBS`/`NVCC_THREADS`: source comment flags these as "probably vllm
  compile time only flags and maybe have no runtime effect" — not
  independently confirmed to matter.

CUDA / NVHPC paths (all rooted at `$NVHPC_ROOT`, resolved via
`resolve_nvhpc_root()` in `utils.sh` — which hard-checks for SDK version
`26.3` and fails loudly if missing, not silently falling back):

| Variable | Value | Evidence |
|----------|-------|----------|
| `CUDA_VERSION` | `12.9` | proven |
| `NVHPC_ROOT` | `resolve_nvhpc_root()` → `$nvhpc_dir/Linux_aarch64/26.3` | proven |
| `CUDA_HOME` | `$NVHPC_ROOT/cuda/$CUDA_VERSION` | proven |
| `PATH` | `$CUDA_HOME/bin:$PATH` | proven |
| `CUDA_PATH` | `$CUDA_HOME` | documentation |
| `C_INCLUDE_PATH` | `$CUDA_HOME/include:...` | proven |
| `CPLUS_INCLUDE_PATH` | `$CUDA_HOME/include:...` | proven |
| `CPATH` | `$NVHPC_ROOT/math_libs/$CUDA_VERSION/include:...` | proven |
| `TRITON_PTXAS_PATH` | `$NVHPC_ROOT/cuda/12.9/bin/ptxas` | proven |
| `VLLM_ENABLE_CUDA_COMPATIBILITY` | `1` | documentation |
| `VLLM_CUDA_COMPATIBILITY_PATH` | `$NVHPC_ROOT/cuda/$CUDA_VERSION/compat` | documentation |

Notes:
- `CUDA_VERSION=12.9` is the version the NVHPC 26.3 SDK provides
  forward-compat for; changing it without also changing the SDK install
  would break `CUDA_HOME` resolution.
- `CPATH` (this entry) avoids a specific, named build failure: flashinfer
  JIT kernels include `cublasLt.h`, which lives in `math_libs`, not
  `cuda/include` — it's extended again later for rdma/libfabric, see
  Networking below.
- `TRITON_PTXAS_PATH`: without this, Triton would need to find `ptxas` on
  `PATH` itself (which the `PATH` export above already guarantees) —
  pointing Triton at it directly removes that dependency.
- `VLLM_ENABLE_CUDA_COMPATIBILITY`/`_PATH`: vLLM's own forward-compatibility
  mechanism, pointed at the NVHPC-provided compat libs.

## Networking configuration

NVSHMEM (needed for UCCL-EP / DeepEP-style expert-parallel kernels over
Slingshot):

| Variable | Value | Evidence |
|----------|-------|----------|
| `NVSHMEM_DIR` | `$NVHPC_ROOT/comm_libs/$CUDA_VERSION/nvshmem` | documentation |
| `CMAKE_PREFIX_PATH` | `$NVSHMEM_DIR/lib/cmake:...` | documentation |
| `FI_CXI_OPTIMIZED_MRS` | `false` | documentation |
| `NVSHMEM_REMOTE_TRANSPORT` | `libfabric` | documentation |
| `NVSHMEM_LIBFABRIC_PROVIDER` | `cxi` | documentation |
| `NVSHMEM_DISABLE_CUDA_VMM` | `1` | documentation |

Notes:
- `CMAKE_PREFIX_PATH` must be set *before* anything later that also touches
  `CMAKE_PREFIX_PATH`/`LD_LIBRARY_PATH` — a real v2/v3 ordering bug, caught
  by the `common_env_sources` sandboxed test under `set -u`.
- `FI_CXI_OPTIMIZED_MRS`/`NVSHMEM_REMOTE_TRANSPORT`/`_LIBFABRIC_PROVIDER`/
  `_DISABLE_CUDA_VMM`: cited against [NVSHMEM 2.6.0 release notes](https://docs.nvidia.com/nvshmem/archives/nvshmem-260/pdf/NVSHMEM-Release-Notes.pdf).

Network interface selection (forces every layer — Gloo, NCCL, PyTorch's
internal TensorPipe — onto the same physical Slingshot interface rather
than letting each pick independently):

| Variable | Value | Evidence |
|----------|-------|----------|
| `GLOO_SOCKET_IFNAME` | `hsn0` | recommendation |
| `NCCL_SOCKET_IFNAME` | `hsn` | recommendation |
| `TP_SOCKET_IFNAME` | `hsn0` | recommendation |

TODO: EP_NIC_NAME="cxi0" I've moved into common-env.sh and could do with documenting why this is different

Notes:
- `NCCL_SOCKET_IFNAME=hsn` is a deliberate prefix match (matches
  `hsn0`-`hsn3`, all 4 Cassini NICs) — `GLOO_SOCKET_IFNAME`'s exact-match
  form only needs one, `hsn0`. **Confirmed equivalent to HPE's own vendor
  tooling**: HPE's `shs-nccl-env` plugin sets this explicitly as
  `hsn0,hsn1,hsn2,hsn3` rather than relying on prefix-matching, but per
  NCCL's own documented semantics the two forms resolve identically on this
  platform — see `knowledge-base.md`'s NCCL/Libfabric section for the full
  vendor-tool comparison.
- `TP_SOCKET_IFNAME`: forces PyTorch's internal TensorPipe layer to follow
  Gloo to the exact index.

UCCL / rdma-core (UCCL-EP is this project's DeepEP-equivalent
expert-parallel backend — see `architecture.md`'s Cross-Project Learnings
§4):

| Variable | Value | Evidence |
|----------|-------|----------|
| `RDMA_ROOT` | `resolve_rdma_dir()` → `$IVLLM_PROJECTDIR/engine/rdma` | proven |
| `USE_LIBFABRIC_CXI` | `1` | recipe |
| `USE_DMABUF` | `1` | recipe |
| `UCCL_SOCKET_IFNAME` | `hsn0` | recommendation |
| `LIBFABRIC_INC_DIR` | `/opt/cray/libfabric/1.22.0/include` (hardcoded) | recipe |
| `LIBFABRIC_LIB_DIR` | `/opt/cray/libfabric/1.22.0/lib64` (hardcoded) | recipe |
| `CPATH` (extended) | `+= $RDMA_ROOT/include:$LIBFABRIC_INC_DIR` | proven |
| `CFLAGS`/`CPPFLAGS`/`CXXFLAGS` (extended) | `+= -I$RDMA_ROOT/include -I$LIBFABRIC_INC_DIR` | proven |
| `LDFLAGS` (extended) | `+= -L$RDMA_ROOT/lib64 -L$RDMA_ROOT/lib -L$LIBFABRIC_LIB_DIR` | proven |
| `LD_LIBRARY_PATH` (extended) | `+= $RDMA_ROOT/{lib64,lib}:$LIBFABRIC_LIB_DIR` | proven |
| `LIBRARY_PATH` (extended) | `+= $RDMA_ROOT/{lib64,lib}:$LIBFABRIC_LIB_DIR` | proven |

Notes:
- `USE_LIBFABRIC_CXI`/`USE_DMABUF`: no citation attached at this specific
  point in `common-env.sh` — don't confuse with the *separate* DeepEP docs
  citation for `EP_NIC_NAME`/`EP_DISABLE_GIN` in `vllm-env.sh`
  (`knowledge-base.md`), which is a different pair of variables entirely.
- `UCCL_SOCKET_IFNAME`: inline comment notes it "should be the interface
  that you would use for the `--master_addr` in torchrun" — i.e. must track
  `TP_SOCKET_IFNAME` above.
- `LIBFABRIC_INC_DIR`/`_LIB_DIR`: hardcoded absolute paths, not resolved
  from a module variable — will silently stop matching if the `libfabric`
  module's version ever changes without these being updated too.
- The extended `CPATH`/`CFLAGS`/etc. rows are needed so `nvcc`/`g++` can
  find rdma-core + libfabric headers/libs when building anything that
  touches UCCL.
- Final `LD_LIBRARY_PATH` ordering (brics/nccl first, then rdma/libfabric,
  then NVHPC compat/CUDA/compilers/NCCL/NVSHMEM/math) is called out as
  deliberate in the source comment, but not independently re-verified here
  that reordering it would actually break anything (inferred).

## Cache setup

Model/HuggingFace cache (`common-env.sh`):

| Variable | Value | Evidence |
|----------|-------|----------|
| `HF_HOME` | `resolve_model_dir()` (no args) `+ /hf` → `$IVLLM_PROJECTDIR/model/hf` | proven |

Shared across the whole team by design (see `architecture.md` §7), not
per-user.

JIT compilation cache — this is `utils.sh`, not `common-env.sh`, but "where
compiled kernels live" is squarely a cache-setup question. Per-node
node-local (tmpfs) cache directories, set by `set_jit_caches(job)` and
rooted at `resolve_localdir(job)` (`/local/user/<uid>`):

| Variable | Value |
|----------|-------|
| `VLLM_CACHE_ROOT` | `$localdir/vllm` |
| `EP_JIT_CACHE_DIR` | `$localdir/deep_ep_cache` |
| `DG_JIT_CACHE_DIR` | `$localdir/deep_gemm_cache` |
| `TRITON_CACHE_DIR` | `$localdir/triton` |
| `FLASHINFER_JIT_CACHE_DIR` | `$localdir/flashinfer` |
| `VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR` | `$localdir/flashinfer_auto` |
| `TORCHINDUCTOR_CACHE_DIR` | `$localdir/torchinductor` |
| `VLLM_XLA_CACHE_PATH` | `$localdir/xla` |

Notes:
- `resolve_localdir(job)`: **TODO in source itself** — `brics/userenv`
  (loaded above) also sets `$LOCALDIR`/`$SCRATCHDIR`; the relationship
  between that and this override has not been reconciled, per an explicit
  `# TODO` comment in `utils.sh` right above the function (inferred, not yet
  checked against what `brics/userenv` actually provides).
- `resolve_job_jit_cache(job)`: `$HOME/.cache/ivllm/<job>/jit-cache-<hash>.tar.gz`,
  `<hash>` = md5 of `$IVLLM_PROJECTDIR $vllmVersion $model $dp $tp $pp_node $ep`.
  See `active-issues.md`'s "JIT cache growing without converging" entry for
  a live, unresolved question about whether this hash needs to be
  node-scoped too for `torch.compile`'s own cache specifically.
- `save_cache()`/`restore_cache()`: tar the above to/from `$localdir`, only
  saving on `SLURM_NODEID == 0`. One known race, fixed: tolerates GNU tar
  exit code 1 ("some files changed while reading") as non-fatal (proven,
  `active-issues.md`).

## Debugging flags

Design intent: **`IVLLM_DEBUG_LEVEL` is meant to become a single master
flag** that decides everything that gets included in debug logging — which
third-party env vars get set, at what verbosity — rather than each session
hand-assembling NCCL/libfabric/vLLM `env:` overrides separately every time.
That makes what follows a **design choice ivllm is making**, not a
configuration override the user tunes per job.

Levels 0-2 below are implemented today, in `report_memory()`/`wait_report()`
(`utils.sh`), unchanged by anything below. Levels 3-4 previously had no
implementation at all (`report_memory()` had no `debug_level < 3` branch, so
setting `3` behaved identically to `2`) — **now prototyped, not yet merged**:
`design/prototype/debug-monitor-prototype.sh` implements `set_debugging_env()`
(the function this section describes) plus a stall-triggered flight-recorder
dump wired through modified `monitor_head()`/`wait_report()` functions. It is
syntax-checked but not yet copied into `utils.sh` or wired into the three
node-launch scripts — see that file's own header comment and "WIRING NOTES"
for the exact call sites and an important ordering constraint
(`set_debugging_env()` must run *after* the job's `env:` block is evaluated,
not alongside `set_jit_caches()` near the top of each script, since
`IVLLM_DEBUG_LEVEL` is itself a job-config setting and isn't visible yet at
that earlier point).

### Level 3 (prototyped) — broad diagnostic logging

Sets, in `set_debugging_env()`:
```
VLLM_LOGGING_LEVEL=DEBUG
NCCL_DEBUG=INFO
FI_LOG_LEVEL=info
FI_LOG_PROV=cxi
```
`VLLM_LOGGING_LEVEL` is a real vLLM env var (`vllm/envs.py`, default
`INFO`) controlling vLLM's own Python logger — not to be confused with the
typo `VLMM_LOGGING_LEVEL`, which is silently a no-op (see the typo callout
at the top of this document).

Deliberately **not** `FI_LOG_LEVEL=trace` — counter-intuitive but directly
confirmed on this platform: `info` produced *more* detail than `trace` at
startup, and neither showed anything during an actual live hang (every
relevant log line was from initialization, nothing after warmup completed).
A level design that naively maps "higher number = more verbose" would get
this specific one backwards. `debug` is libfabric's most detailed level but
needs debug support compiled into the build, not confirmed available here.
(Source: `active-issues.md` GLM-5.2 entry.)

**Verified exhausted as of the 2026-08-12 GLM-5.2 run** (`logs/glm52q/20260812_213446/`):
checked every rank's own last `NCCL_DEBUG=INFO` log line against the actual
hang window — all 8 ranks logged a collective launch in the same narrow
window right before the freeze, then **zero further NCCL log lines from any
rank for the entire ~3.5-minute hang**. `NCCL_DEBUG=INFO` only logs at
collective *launch*, not while a rank is blocked waiting inside one, so once
something stalls mid-collective there's structurally nothing further to log
— same conclusion as `FI_LOG_LEVEL`/libfabric above, for the same reason
(proven, this run).

**Candidate addition, not yet in the prototype**: `CUDA_LOG_FILE=stderr` —
turns otherwise-generic CUDA driver API error returns into descriptive
messages, requires CUDA 12.9+ (matches this stack's forward-compat
baseline). Confirmed as a real technique vLLM developers use live to unmask
detailed errors underneath otherwise-opaque NCCL/CUDA-graph failures
([PR #29207](https://github.com/vllm-project/vllm/pull/29207),
[#49826](https://github.com/vllm-project/vllm/issues/49826)). Cheap and
only produces output on an actual error — arguably belongs unconditionally
rather than gated behind level 3 at all, since it costs nothing when
nothing is wrong; not yet a settled design call (documentation for the
technique; inferred for where it belongs in the level design).

### Level 4 (prototyped) — targeted trace, for actively chasing a live hang

Sets, in `set_debugging_env()`:
```
NCCL_DEBUG=TRACE
NCCL_DEBUG_SUBSYS=COLL,PROXY
FI_LOG_SUBSYS=cq,ep_data,mr
TORCH_NCCL_DESYNC_DEBUG=1
TORCH_NCCL_TRACE_BUFFER_SIZE=2000
TORCH_NCCL_DUMP_ON_TIMEOUT=1
TORCH_NCCL_TRACE_CPP_STACK=1
TORCH_NCCL_DEBUG_INFO_TEMP_FILE=<job debug dir>/torch_nccl_trace_node<N>
TORCH_NCCL_DEBUG_INFO_PIPE_FILE=<job debug dir>/torch_nccl_dump_trigger_node<N>
```
High log volume; reserve for a genuinely stuck job, not routine debugging.
Watch the spelling if setting `NCCL_DEBUG_SUBSYS` by hand — not
`NCCL_DEBUF_SUBSYS` (the other typo callout at the top of this document).
The `NCCL_DEBUG`/`FI_LOG_SUBSYS` pair was the original, still-untested part
of this level; the `TORCH_NCCL_*` family (moved here from `knowledge-base.md`
— see below) is the newer and now higher-priority half.

**`TORCH_NCCL_*` (`ProcessGroupNCCL`'s own wrapper layer, sitting *above*
raw NCCL/libfabric) — currently the single most promising unexplored lead
for the GLM-5.2 hang investigation.** None of it has been tried on a real
hang yet.

- **`TORCH_NCCL_DESYNC_DEBUG=1` is a completely different, and arguably far
  better, diagnostic tool than the custom pyspy-based approach this project
  has built by hand.** The months of manual work characterizing this hang —
  the diagnostic patch to label `queue=response[rank=N]`, the
  "0%-GPU-util rank is backwards" finding, reading pyspy stack snapshots to
  guess which rank is actually stuck — is close to what this flag is *built
  by PyTorch specifically to answer*: its own description is *"helpful in
  figuring out the culprit rank of collective desync."* This should be
  tried before any further manual pyspy-based triage (inferred — the
  mechanism directly targets the observed problem, but the actual behavior
  against this specific hang is untested).
- **`TORCH_NCCL_DUMP_ON_TIMEOUT=1` + `TORCH_NCCL_TRACE_BUFFER_SIZE`
  (must be `>0`)**: PyTorch's "flight recorder" — a ring buffer of
  collective start/end events dumped automatically when the watchdog
  detects a timeout or exception. Exactly the kind of structured,
  collective-level history pyspy's Python-stack snapshots can't provide
  (pyspy shows *where* a process is stuck, not *which collective, on which
  rank, relative to every other rank's collective history*). Directly
  relevant to bottoming out `shm_broadcast.py` further, since it would show
  the actual collective sequence numbers each rank reached before the hang
  locked in (inferred, untested against this specific hang).
- **Operational gotcha, confirmed via `vllm-serve-cli.md`, and the reason
  the prototype also includes a manual trigger path**: `DUMP_ON_TIMEOUT`/
  `DESYNC_DEBUG` only fire when PyTorch's own NCCL watchdog detects a
  timed-out collective — which requires waiting out the full
  `--distributed-timeout-seconds` window, **600s by default** ("If None,
  PyTorch's default timeout is used (600s for NCCL)"). GLM-5.2's confirmed
  hang windows so far (e.g. `20260812_213446`, ~3.5 minutes from last
  activity to manual `ivllm cancel`) have been well short of 600s — meaning
  these env vars alone, on the same cancel-when-it-looks-stuck cadence used
  so far, would have produced **nothing**, not because the diagnostic
  doesn't work but because the job gets killed before the watchdog ever
  gets a chance to fire (proven for the default timeout value; inferred for
  why no prior run would have produced flight-recorder output even if these
  vars had been set).
- **`TORCH_NCCL_DEBUG_INFO_PIPE_FILE`: the practical fix for the 600s
  gotcha, and the mechanism the prototype builds around.** Writing anything
  to this pipe manually triggers an immediate flight-recorder dump, without
  waiting the full 600s. See "Stall-triggered dump" below for how the
  prototype wires this to the existing fast (~60s)
  `"No available shared memory broadcast block"` warning instead of relying
  on patience or racing the internal watchdog.

#### Stall-triggered dump (new since the original design — see the prototype)

`design/prototype/debug-monitor-prototype.sh` adds cooperation between two
functions that already exist, rather than a new standalone mechanism:

- **`monitor_head()`** (head node, already tails the aggregated job log
  every tick) gains a second grep alongside its existing crash-pattern
  check, against a new `IVLLM_STALL_INDICATORS` array (currently just
  `"No available shared memory broadcast block found"`). On a new
  occurrence — gated by `IVLLM_STALL_COOLDOWN_SECS` (default 300s) so an
  ongoing hang's repeating message doesn't re-trigger every tick — it writes
  a timestamp into a shared `debug/stall_detected` sentinel file and calls
  a new `trigger_torch_nccl_dump()` helper, which globs
  `debug/torch_nccl_dump_trigger_node*` (one FIFO per node that reached
  level 4) and writes to each with a `timeout`-guarded, repeated write (see
  open questions below for why "repeated"). A stall is a diagnostic event
  only — this does not change `monitor_head()`'s control flow, exit codes,
  or shutdown decisions.
- **`wait_report()`** (runs on every node, head and workers alike) watches
  the same `debug/stall_detected` sentinel and, the first time it sees a
  *new* timestamp there, calls `report_memory()` immediately, once, outside
  its normal polling cadence — giving one synchronized pyspy/GPU snapshot
  from every node at close to the same real moment the stall was noticed,
  rather than relying on independent ~10s ticks to happen to land close
  together by chance. `report_memory()` itself needs no changes at all.

#### Open implementation questions (from the prototype, unresolved)

- `TORCH_NCCL_DEBUG_INFO_PIPE_FILE` is set **per node, not per rank** —
  `set_debugging_env()` runs once per node before vLLM forks its worker
  processes, so individual rank IDs aren't known yet in a Ray-executor
  deployment. All processes on a node (EngineCore + workers, or just
  workers) would share one named pipe. It is **not confirmed** whether
  PyTorch's NCCL watchdog threads across multiple processes can usefully
  share one FIFO, or whether a single write only wakes one reader (the more
  likely FIFO semantics) — `trigger_torch_nccl_dump()` writes several times
  with small gaps as a crude mitigation, unverified against a real run.
- `TORCH_NCCL_DEBUG_INFO_TEMP_FILE`'s exact on-disk naming behavior isn't
  confirmed — whether PyTorch appends its own rank suffix to the given
  path, or overwrites one file per process sharing it, needs checking
  against the actual filenames produced on the first real triggered dump.

#### Other `TORCH_NCCL_*` vars considered, not included in level 4

Real, documented knobs with no ivllm history — untested, but genuinely
future-tuning/diagnostics relevant regardless of how the hang investigation
resolves:

- `TORCH_NCCL_ENABLE_MONITORING`/`TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC`: does
  **not** fix the hang, but would abort a genuinely stuck job after a
  bounded time instead of hanging indefinitely — a real operational
  improvement (freeing tied-up GPU allocation) independent of
  root-causing the hang itself. Needs a heartbeat timeout deliberately set
  longer than the longest legitimate cold-JIT-compile stall already
  observed (`VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800` exists for exactly
  this reason at the vLLM layer) to avoid false-positive aborts (inferred).
- `TORCH_NCCL_USE_COMM_NONBLOCKING`: the torch-layer counterpart to the raw
  `NCCL_COMM_BLOCKING` env var, which `knowledge-base.md`'s Mandatory table
  already **confirms crashes Ray outright** when set to `0`
  (non-blocking). Worth understanding the relationship between these two
  before touching either: it's not yet established whether this
  torch-level flag interacts with, overrides, or is independent of the
  raw-NCCL env var, or whether it would reproduce the same Ray crash
  (documentation for the existing crash; inferred/untested for the
  interaction).
- `TORCH_NCCL_ASYNC_ERROR_HANDLING` (default `3`, teardown without abort),
  `TORCH_NCCL_BLOCKING_WAIT`, `TORCH_NCCL_HIGH_PRIORITY`,
  `TORCH_NCCL_AVOID_RECORD_STREAMS`, `TORCH_NCCL_ENABLE_TIMING` (cheap,
  pairs well with the flight recorder), `TORCH_NCCL_COORD_CHECK_MILSEC`/
  `TORCH_NCCL_WAIT_TIMEOUT_DUMP_MILSEC` (only matter once
  `ENABLE_MONITORING`/`DUMP_ON_TIMEOUT` are in use), `TORCH_NCCL_NAN_CHECK`
  (correctness diagnostic, unrelated to the hang mechanism itself) — none
  tried, no specific hypothesis for any of them yet (documentation).

### Not worth including at any level

- `enable-logging-iteration-details` (a `vllm serve`/`ObservabilityConfig`
  flag, not an env var, so this master function couldn't set it directly
  anyway): tried, confirmed to add **no** additional per-iteration
  scheduler logging in practice. Negative result, not a useful lever
  (`active-issues.md` GLM-5.2 entry).

### Possible future additions — real diagnostic vars, not yet wired into any level

Genuine diagnostic-purpose env vars turned up during this session's review
of `vllm-env.md`/`triton.md`, moved here from `knowledge-base.md` since
they're logging/debug-output toggles rather than job-tunable behavior
(`CUDA_LOG_FILE` is the same category — see the Level 3 section above,
where it's kept alongside the other "broad diagnostic logging" candidates
rather than repeated here). None of these are in `set_debugging_env()`
today:

- `VLLM_GC_DEBUG`: real feature ([PR #24829](https://github.com/vllm-project/vllm/pull/24829)
  "[Core] GC Debug callback", `VLLM_GC_DEBUG`/`VLLM_GC_DEBUG_TOP_COLLECTED_OBJECTS`,
  no-op by default). Used as a genuine diagnostic in
  [#48620](https://github.com/vllm-project/vllm/issues/48620) — but for a
  **shutdown/teardown delay** (`SIGTERM` teardown running two full-heap
  gen-2 GC passes, ~4-7s), not a live-serving hang. No report connects GC
  pauses to a lock/memory-fence race in `shm_broadcast.py` specifically —
  mostly a shutdown-latency diagnostic, not a hang candidate (documentation).
- `VLLM_COMPUTE_NANS_IN_LOGITS`: checks generated logits for NaNs,
  "indicating corrupted output... useful for debugging low level bugs or
  bad hardware." Real, deliberate use elsewhere, including on the same
  model family: [#50435](https://github.com/vllm-project/vllm/issues/50435)
  (GLM-5.2-FP8 P/D accuracy regression — enabled it, no NaN warnings
  appeared, ruled out as the cause), [#47722](https://github.com/vllm-project/vllm/issues/47722),
  [#48221](https://github.com/vllm-project/vllm/issues/48221) (both set it
  as standard debug practice in prefill/decode pod specs),
  [#27301](https://github.com/vllm-project/vllm/issues/27301) (feeds
  `Request.is_output_corrupted`/a `num_corrupted_request` metric). A
  correctness diagnostic, not linked to the hang mechanism itself
  (documentation).
- `TRITON_PRINT_AUTOTUNING`: prints the best autotuning config and total
  time spent per kernel after autotuning completes — a cheap diagnostic
  that could quantify the JIT-compile stalls already observed. One
  incidental GitHub sighting, [#48718](https://github.com/vllm-project/vllm/issues/48718)
  (Qwen3.6 35B-A3B NVFP4 2-3x slower + hangs on B300s) has it set in the
  reporter's env dump, but it isn't analyzed or discussed as relevant to
  that issue — no corroboration either way (documentation).

### Level 0 (default) — implemented

- Per-node cache size / RAM / top-6-process summary line, once per tick.

### Level 1 — implemented

Adds:
- Per-GPU utilisation/memory via `nvidia-smi` — cheap, no process attach.

### Level 2 — implemented

Adds:
- `py-spy dump --nonblocking` stack traces of every vLLM/Ray worker process,
  appended to `debug/pyspy-node<N>.log` under the job directory (persistent
  — survives `clear_localdir()`).

(Levels 3-4 are described above, ahead of 0-2, since they're the
prototyped-but-not-yet-merged part this section exists to capture — 0-2 are
already shipped and just need recording here for completeness.)

### `IVLLM_RUNTIME_DEBUG`

A separate, existing flag — `IVLLM_RUNTIME_DEBUG=1` makes `wait_report()`
call `report_memory()` on every tick regardless of job status, not just
while `initialising`. Independent of `IVLLM_DEBUG_LEVEL` today; the two are
normally set together, and a master-flag design should probably fold this
in too rather than keeping it a second, separately-remembered knob.

