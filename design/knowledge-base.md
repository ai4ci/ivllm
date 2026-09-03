# Isambard VLLM configuration knowledge base

This document must be actively maintained after model debugging or benchmarking.

There are an overwhelming number of possible enviroment flags and vllm serve cli options.

This document tells us what is available, what is relevant, what ivllm is defaulting
to and what has been tested and works / does not work / makes no difference in
the vllm testing and benchmarking we have done so far. For every flag we need a
recommendation.

The flags or options **must be proven to exist in recent versions of the relevant software**
and not be a hallucination, large amounts of web sourced content is unreliable or
outdated.

Referenced:
* The vllm source code - gold standard
* VLLM env vars: https://docs.vllm.ai/en/stable/configuration/env_vars/
* VLLM serve: https://docs.vllm.ai/en/stable/cli/serve/
* NCCL: https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2307/user-guide/docs/env.html
* NCCL (Brics): https://docs.isambard.ac.uk/user-documentation/guides/nccl/
* Modules (Brics): https://docs.isambard.ac.uk/user-documentation/guides/modules/
* VLLM (Brics): https://docs.isambard.ac.uk/user-documentation/tutorials/distributed-inference/
* Libfabric: https://ofiwg.github.io/libfabric/main/man/onepage.html
* Slingshot libfabric: https://ofiwg.github.io/libfabric/main/man/fi_cxi.7.html
* CUDA: https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html
* Torch: https://docs.pytorch.org/docs/2.13/torch_environment_variables.html

The scope of this is limited to flags that may materially affect the stability
or performance, or behaviour of vllm on isambard, and might be set as options in
a `vllm.yaml` configuration, or are defaulted in `vllm-env.sh`. It also lists
flags or variables that can't affect Isambard or are irrelevant and why, so
that we don't waste time investigating dead ends.

Out of scope: purely mechanistic e.g. cache directories, ports, IPs that are
set by ivllm itself, debugging options, hard coded library paths that might be found in
`common-env.sh`. for these see [ivllm-environment](design/ivllm-environment.md).

Table schema — kept deliberately minimal, one row per flag:
* **Variable / flag**
* **Value** — the recommended value (or, for a "Mandatory" table, the only
  sane value)
* **Use** — `always` / `sometimes` / `never`. `always`/`never` entries belong
  in a "Mandatory / must not use" table, not the main tuning tables — if it's
  settled, it's not something we're iterating on. `sometimes` stays in the
  main tables below.

Anything needing more than a half-line of justification goes in a short
free-form **Notes** block under the table, one bullet per flag that needs
it — not crammed into the row. Every note should end with an evidence tag:
`(proven)` reproducible debugging/benchmarking output on this hardware,
`(recommendation)` Isambard/GH200-specific (Brics, isambard_containers,
doubleword.ai), `(documentation)` official docs only, `(recipe)` vLLM
cookbook/non-GH200-specific, `(inferred)` web search / not independently
checked.

**First-pass status (2026-08-12)**: populated from `vllm-env.sh`'s current
defaults plus the GLM-5.2-AWQ-INT4 investigation (`design/active-issues.md`
— by far the richest source of *proven* evidence this project has) and the
`examples/*.yaml` configs. Not done in this pass: a systematic
`vllm serve --help`/env-vars-doc diff against `vllm-env.sh` to catch
anything renamed/removed upstream.

## Environment variables

N.B. only environment variables, not `vllm serve` flags — see the next
top-level section for those.

### Mandatory / must not use

Settled — not up for iteration. If the evidence for one of these looks thin,
that's worth flagging, but the row belongs here either way once the verdict
is `always`/`never`, not in the tuning tables below.

| Variable | Value | Use | Context | Note |
|---|---|---|---|---|
| `FI_PROVIDER` | `cxi` | always | | |
| `FI_CXI_DEFAULT_CQ_SIZE` | `131072` | always | | prevents dropped frames |
| `FI_CXI_DISABLE_HOST_REGISTER` | `1` | always | | avoids queue overflow |
| `FI_MR_CACHE_MONITOR` | `userfaultfd` | always | | avoids Slingshot memhook deadlock |
| `NCCL_CROSS_NIC` | `1` | always | | 4-NIC striping |
| `NCCL_MIN_NCHANNELS` | `4` | always | | matches NIC count |
| `CUDA_DEVICE_MAX_CONNECTIONS` | `1` | always | | avoids transfer races |
| `VLLM_ALLOW_LONG_MAX_MODEL_LEN` | `1` | always | | |
| `VLLM_ENGINE_ITERATION_TIMEOUT_S` | `1800` | always | multi-node | |
| `VLLM_ALLREDUCE_USE_SYMM_MEM` | `0` | always | | disables broken experimental allocator |
| `VLLM_COMPILE_CACHE_SAVE_FORMAT` | `binary` | always | | changed from `unpacked` 2026-09-03; per vLLM's own docs, `binary` is multiprocess-safe, `unpacked` is not (race conditions with concurrent `vllm serve` processes sharing a cache) |
| `DG_JIT_USE_RUNTIME_API` | `1` | always | DeepGEMM | |
| `VLLM_USE_DEEP_GEMM_E8M0` | `0` | always | Hopper/GH200 | E8M0 unsupported |
| `EP_DISABLE_GIN` | `1` | always | DeepEP/UCCL-EP | |
| `FI_CXI_RDZV_THRESHOLD` | `0` | never (expecting "always rendezvous") | | floored to 192 regardless — **matches vendor's own guidance, see note** |
| `FI_CXI_RDZV_GET_MIN` | `0` | never (expecting "always rendezvous") | | see `FI_CXI_RDZV_THRESHOLD` |
| `FI_CXI_RDZV_EAGER_SIZE` | `0` | never (expecting "always rendezvous") | | see `FI_CXI_RDZV_THRESHOLD` |
| `FI_CXI_REQ_BUF_SIZE` | `12MB` (default) | sometimes | software/hybrid RX match mode | untested — see note |
| `FI_CXI_REQ_BUF_MIN_POSTED` | `6` (default) | sometimes | software/hybrid RX match mode | untested — see note |
| `NCCL_GDRCOPY_ENABLE` | `1` | **genuinely unresolved, currently `1`** | see note | conflicting evidence, needs a network benchmark to settle — see note |
| `FI_HMEM_CUDA_USE_GDRCOPY` | `1` | **genuinely unresolved, currently `1`** | see note | same as `NCCL_GDRCOPY_ENABLE` above |

**Update, 2026-09-03 (Rob)**: the "closed, wrong tree" verdict above was this project's own conclusion from reading libfabric/NCCL reference docs — it was never settled by an actual network measurement, and there's real, conflicting evidence on the other side that this earlier pass didn't have. Isambard's own container definition ([UKGovernmentBEIS/isambard_containers](https://raw.githubusercontent.com/UKGovernmentBEIS/isambard_containers/refs/heads/main/definitions/vllm/vllm.def)) changed to enabling GDRCopy, which is what prompted re-enabling it here; separately, HPE's own [`shs-nccl-env`](https://github.com/HewlettPackard/shs-nccl-env) (the vendor-authored Slingshot NCCL tuning tool) validates using GDRCopy rather than treating it as unnecessary. The "GDRCopy is not needed with vLLM — vllm bypasses NCCL for intranode comms and between nodes is using RDMA over slingshot" comment that used to justify leaving it off is left in `vllm-env.sh` (commented out) precisely because its own provenance is unclear and it's now at odds with both of the sources above. **This can only be settled empirically** — with GDRCopy enabled vs disabled, run the same collective workload and compare, ideally with the standalone network benchmark harness described in `design/priorities.md` (an evolution of `design/prototype/slingshot-tp-reprex.sh`) rather than a full vLLM job. Until that test exists and is run, treat this flag's value as "currently enabled, unverified either way" rather than either a proven-good or proven-bad setting.
| `CUDA_MANAGED_FORCE_DEVICE_ALLOC` | `1` | never (leave `0`) | conflicts with `offload-backend: uva` | would force device-only storage, defeating CPU offload — see note |
| `NCCL_COMM_BLOCKING` | `0` | never | Ray executor | crashes Ray outright |

Notes:
- `FI_CXI_DEFAULT_TX_SIZE`/`FI_CXI_RX_MATCH_MODE`: **moved out of this table**
  — no longer a settled "never," see the NCCL/Libfabric tuning section below.
  New vendor evidence (HPE's own `shs-nccl-env` tool) tips this back into
  "worth reconsidering" territory despite the earlier inconclusive test.
- `FI_CXI_RDZV_THRESHOLD`/`_GET_MIN`/`_EAGER_SIZE` set to `0` does **not**
  mean "always rendezvous" as it looks like it should — libfabric silently
  floors the threshold to 192 bytes regardless, confirmed via
  `cxip_env_init()`'s own `<warn>`-level log line (proven). Per `fi_cxi(7)`
  (`design/references/fi_cxi.md`): *"For both 'hybrid' and 'software' modes
  ... care should be taken to minimize the threshold for rendezvous
  processing"* — the config actually tested has `FI_CXI_RX_MATCH_MODE=software`,
  one of the two modes this applies to, so the 192-byte floor happens to
  already satisfy the vendor's own recommendation rather than being an
  unverified risk (documentation, upgrades the earlier "untested" framing).
- `FI_CXI_REQ_BUF_SIZE`/`FI_CXI_REQ_BUF_MIN_POSTED`: per the same manual
  passage, these — not `FI_CXI_OFLOW_BUF_*` — are the buffer-sizing knobs
  that actually govern eager/unmatched-message capacity under software or
  hybrid RX match mode, which is what's tested here (`OFLOW_BUF_*` governs
  hardware-mode overflow buffers instead). Untried in this project; the
  manual flags undersizing at job scale as a cause of flow-control
  activation, which is a plausible-but-unconfirmed contributor to hangs at
  the 12-GPU scale this hang only manifests at (documentation).
- `NCCL_GDRCOPY_ENABLE`/`FI_HMEM_CUDA_USE_GDRCOPY`: **closed out — GDRCopy was
  the wrong tree.** Original motivation: libfabric's manual documents a real
  "CUDA deadlock" failure mode (`cudaMemcpy()` inside libfabric can deadlock
  against a blocked CUDA kernel), naming GDRCopy as one of two mitigations
  (the other, `FI_OPT_CUDA_API_PERMITTED`, see the NCCL/Libfabric section
  below); GLM-5.2's longest-surviving run happened to set both alongside the
  `NCCL_NET_GDR_LEVEL=PHB` change
  above (not isolated from that test, so never independently confirmed).
  Three independent findings now argue against it: (1) per `fi_cxi(7)`
  (`design/references/fi_cxi.md`, confirmed direct quote): *"the CXI provider
  enables ... FI_HMEM_CUDA_USE_DMABUF by default if not specifically set"* —
  DMABUF is a different zero-copy mechanism that also avoids a host-side
  `cudaMemcpy` in the transfer path, so CXI already sidesteps this deadlock
  class by default; `fi_cxi(7)` never mentions GDRCopy. (2) per `nccl.md`
  (`design/references/nccl.md`): `NCCL_DMABUF_ENABLE` defaults to `1`
  (enabled) at the NCCL layer too, independent of CXI — a second, separate
  default pointing at DMABUF, not GDRCopy. (3) `NCCL_GDRCOPY_ENABLE` itself
  does not appear anywhere in that same reference despite it documenting 102
  other `NCCL_*` variables including several added in the last few releases
  — real question mark over whether this flag does anything at all in
  whatever NCCL build ivllm runs, as opposed to being a stale/vendor-specific
  knob. Separately, there is a real, maintainer-confirmed, CXI-specific race
  condition in the GDRCopy cleanup path —
  [ofiwg/libfabric#10041](https://github.com/ofiwg/libfabric/issues/10041), a
  double-free when two threads unmap/unpin the same GDR handle, reported on
  multi-node Slingshot 11 (this project's exact fabric), maintainer quote:
  *"very much tied to the cxi provider"* — looks stale rather than
  deliberately ruled out (last human activity Aug 2024), no workaround
  documented. Net: leave both unset/`0` — every layer of this stack already
  defaults to DMABUF without needing GDRCopy, and explicitly enabling it adds
  a live, unresolved crash risk with no confirmed benefit (documentation for
  the DMABUF defaults; inferred for the net verdict).
- `NCCL_COMM_BLOCKING=0` crashes Ray outright — directly observed, see
  `examples/glm-5.2-743b-int4.yaml`'s own comment (proven).
- `FI_CXI_DEFAULT_CQ_SIZE`/`FI_CXI_DISABLE_HOST_REGISTER`/`FI_MR_CACHE_MONITOR`
  each avoid a named failure mode per `vllm-env.sh`'s own comments (queue
  overflow / "LE resources not recovered" flow-control errors, and Slingshot
  memory-hook deadlocks respectively) (recommendation).
- `EP_DISABLE_GIN` per [deepseek-ai/DeepEP](https://github.com/deepseek-ai/DeepEP) (documentation).
- `VLLM_USE_DEEP_GEMM_E8M0=0`: E8M0 is not supported on Hopper, which GH200 is (documentation).
- `CUDA_MANAGED_FORCE_DEVICE_ALLOC`: per `cuda-env.md`
  (`design/references/cuda-env.md`), a non-zero value "forces the driver to
  use device memory for physical storage" for all Unified Memory in the
  process. ivllm's `offload-backend: uva` (see Offloading, below) relies on
  CUDA Unified Memory being genuinely free to live in CPU *or* GPU memory
  depending on demand — that's the entire mechanism `cpu-offload-gb` uses to
  free GPU memory. Forcing device-only storage would silently defeat that
  offload path (or return `cudaErrorInvalidDevice` if devices aren't
  peer-to-peer compatible). Not currently set anywhere in ivllm — confirmed
  via grep — so this is a "don't ever introduce this" note rather than a bug
  fix (documentation).

### VLLM

Still open / worth iterating on.

| Variable | Value | Use | Context | Note |
|---|---|---|---|---|
| `VLLM_FLASHINFER_ALLREDUCE_BACKEND` | `trtllm` | always | multinode | |
| `VLLM_FLASHINFER_ALLREDUCE_BACKEND` | `mnnvl` | sometimes | single-node | |
| `IVLLM_DISABLE_FLASHINFER` | `1` | always | multinode | |
| `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS` | `1800` | sometimes | | |
| `VLLM_USE_AOT_COMPILE` | untested in isolation | sometimes | | needs research |
| `VLLM_USE_V2_MODEL_RUNNER` | `0` | sometimes | | needs research |

Notes:
- `VLLM_FLASHINFER_ALLREDUCE_BACKEND=trtllm`: correct only when the TP group
  is node-local — it *correctly* refuses (`ValueError`) when the TP group
  itself spans >1 node. The other backend, `mnnvl`, is confirmed
  hardware-incompatible on Slingshot (crashes the CUDA context — no real
  inter-node NVLink/NVSwitch). Genuinely cross-node TP (e.g. TP=8×Ray, no
  DP split) has **no working flashinfer allreduce backend at all** — use
  `IVLLM_DISABLE_FLASHINFER=1` below instead (proven, `active-issues.md`).
- `IVLLM_DISABLE_FLASHINFER=1`: ivllm-specific patch
  (`disable-flashinfer-env.v0.25.1.v1.patch`) — routes to the safe,
  numerically-identical unfused fallback. Needs the patch applied to the
  venv in use; not a stock vLLM flag (proven).
- `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800`: raised after a plain `300`s
  default timeout fired with *no* actual deadlock — pyspy showed every
  worker legitimately busy on a cold Triton JIT compile. Worth raising for
  any large/cold-cache model (proven, `active-issues.md` MiniMax-M3 steps
  15-16).
- `VLLM_USE_AOT_COMPILE`: present in GLM-5.2 runs alongside a JIT-cache
  growth observation (~7GB uncompressed) — plausibly amplifies per-node
  cache-entry size, not isolated from the node-identity cache-key question
  (inferred). **GitHub research**: real, active cache-correctness bug
  history — [#50891](https://github.com/vllm-project/vllm/issues/50891)
  (AOT cache key ignores `limit_mm_per_prompt`, stale artifacts cause
  crashes), [#35766](https://github.com/vllm-project/vllm/issues/35766)
  (alters aotautograd cache keys unintentionally),
  [#31536](https://github.com/vllm-project/vllm/pull/31536) (fix for
  partition-wrapper bug when loading AOT-cached functions). None are
  hang-specific, but they corroborate treating this stack's JIT-cache
  behavior with suspicion generally, consistent with the node-identity
  cache-key question already open here (documentation).
- `VLLM_USE_V2_MODEL_RUNNER=0`: tried per
  [vllm-project/vllm#43420](https://github.com/vllm-project/vllm/issues/43420)'s
  workaround suggestion for a related `RayExecutorV2`/`shm_broadcast` hang —
  kept set through several runs where the hang still occurred; not confirmed
  to have changed anything (inferred). **Re-examined via GitHub research**:
  #43420's actual root cause is that `RayExecutorV2` inherits
  `MultiprocExecutor`'s `shm_broadcast`, which is single-host only — so
  specifically **data-parallel** workers crossing node boundaries silently
  fall back to Gloo-over-TCP, which then times out. GLM-5.2's config runs
  `tensor-parallel-size: 8` with no DP split, so this exact precondition
  likely never applied — which would explain why the workaround didn't
  change anything: it may have been the wrong fix for the wrong bug, not a
  refutation of V2 being involved at all. Worth being suspicious the flag
  even took effect: [#44697](https://github.com/vllm-project/vllm/issues/44697)
  separately reports V2 getting force-re-enabled regardless of this override
  in some flag combinations (MTP+PP>1) — this project has hit silently-
  ignored-flag bugs before (`enable-flashinfer-autotune`, the
  `VLMM_LOGGING_LEVEL` typo), so confirming from logs that GLM-5.2's runs
  actually executed on the legacy V1 runner (not just that the env var was
  set) would be worth doing before ruling V2 out. Two more `shm_broadcast`-
  exhaustion reports use an exact, greppable phrase worth checking our own
  hang logs for: [#51593](https://github.com/vllm-project/vllm/issues/51593)
  ("No available shared memory broadcast block found in 60 seconds",
  deterministic at batch=3, unresolved) and
  [#45198](https://github.com/vllm-project/vllm/issues/45198) ("no available
  shared memory broadcast blocks for 60+ seconds" — see the new "Triton
  JIT-compile hang" entry in `active-issues.md`) (documentation for the root
  cause and phrase matches; inferred for relevance to GLM-5.2 specifically).

### Distributed / Ray executor

*(From `design/references/vllm-env.md`/`vllm-serve-cli.md` — vLLM's own
distributed-execution, Ray-executor, and process-lifecycle vars/flags. None
currently set anywhere in ivllm unless noted otherwise.)*

| Variable | Value | Use | Context | Note |
|---|---|---|---|---|
| `VLLM_USE_RAY_V2_EXECUTOR_BACKEND` | `True` (default) | sometimes | Ray executor version | **confirmed default in our installed 0.26.0** — see note |
| `VLLM_DISTRIBUTED_USE_SPLIT_GROUP` | untested | sometimes | subgroup construction (`split_group` vs `new_group`) | **real external hang mechanism found** — see note |
| `VLLM_SKIP_P2P_CHECK` | `1` (default, skip) | sometimes | custom-allreduce P2P verification | source comment ties `=0` to a hang workaround — weakly corroborated, see note |
| `VLLM_DISABLE_PYNCCL` | `False` (default) | sometimes | pynccl vs `torch.distributed` NCCL backend | **open question — affects whether `TORCH_NCCL_*` vars apply at all**, see note |
| `VLLM_WORKER_MULTIPROC_METHOD` | `fork` (default) | never (already safe) | fork-after-CUDA-init risk | confirmed auto-overridden to `spawn` in Ray actors — see note |
| `VLLM_ENGINE_READY_TIMEOUT_S` | `600` (default) | sometimes | startup timeout | untested override, real precedent in a close analog issue |
| `VLLM_KEEP_ALIVE_ON_ENGINE_DEATH` | `False` (default) | sometimes | keep API server up after engine death | untested, no GitHub precedent |
| `VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS` | `5` (default) | sometimes | engine/worker shutdown timeout | ties to `ivllm-cancel.sh`'s `tidy_up()` work — see note |
| `VLLM_USE_SPINLOOP_EXT` | `0` (default, off) | sometimes | busy-poll optimization in `shm_broadcast.py` | see `active-issues.md`'s `#28053` correction |
| `VLLM_GPU_NIC_PCIE_MAPPING` / `VLLM_NIC_SELECTION_VARS` | untested | sometimes | vLLM-native GPU↔NIC RDMA affinity | **real, working feature, worth trying** — see note |
| `VLLM_ENABLE_STARTUP_PLAN` | `0` (default, off) | sometimes | node-keyed memory-profiling cache | same node-identity-cache-key concern as the JIT cache — see note |
| `--distributed-timeout-seconds` / `--cpu-distributed-timeout-seconds` | untested | sometimes | `torch.distributed.init_process_group` / Gloo timeouts | untested, real adjacent hang reports |
| `--enable-eplb` / `--eplb-config` | untested | sometimes | expert-parallel load balancing | untested, real OOM report under DP=16+redundant-experts |
| `--enable-dbo` / `--ubatch-size` | untested | sometimes | dual-batch overlap, overlaps EP all-to-all with compute | untested, touches our exact EP all-to-all path |
| `--max-parallel-loading-workers` | untested | sometimes | caps parallel weight-loading workers | avoids RAM OOM under TP + large models |

Notes:
- `VLLM_USE_RAY_V2_EXECUTOR_BACKEND`: confirmed via direct read of our
  vendored `vllm-0.26.0` source (`vendor/vllm-0.26.0/vllm/envs.py`) to
  default to `True` — GLM-5.2's Ray-executor deployment already runs on
  `RayExecutorV2`, the exact component
  [#43420](https://github.com/vllm-project/vllm/issues/43420)'s hang lives
  in. A vLLM/Ray-team RFC ([#35848](https://github.com/vllm-project/vllm/issues/35848))
  states the *old* backend (`RayDistributedExecutor`/Compiled Graph) had
  "unresolved stability issues including NCCL hangs and out-of-order
  delivery across multiple Ray releases," which is why `RayExecutorV2` was
  built — both backends have documented hang histories, so there's no clean
  "known-good fallback" here. Forcing `=0` is a diagnostic worth trying, not
  a confident fix (documentation for the default and RFC; inferred for
  relevance).
- `VLLM_DISTRIBUTED_USE_SPLIT_GROUP`: no vLLM-repo hits, but a real,
  independently-discovered mechanism —
  [pytorch/pytorch#145376](https://github.com/pytorch/pytorch/issues/145376)
  reports `torch.distributed.new_group` with the Gloo backend hanging when
  `split_group` was called first and not all ranks belong to the resulting
  group. This flag is exactly the toggle between vLLM's `split_group`-based
  subgroup construction (mixed `cpu:gloo,cuda:nccl` backend + eager
  `device_id` binding, per its own source comment) and the legacy
  `new_group` path. Wouldn't surface in vLLM's own issue tracker since it's
  a PyTorch-level bug — worth testing directly (documentation for the
  PyTorch bug; inferred for relevance here).
- `VLLM_SKIP_P2P_CHECK`: source comment states *"if the program hangs when
  using custom allreduce, potentially caused by a bug in the driver (535
  series), it might be helpful to set `VLLM_SKIP_P2P_CHECK=0`"* — but
  GitHub research found this claim **only in the source comment itself**,
  not corroborated by any actual issue/PR thread describing this fix
  working. Several driver-535/P2P-hang issues were checked directly and
  none mention this variable. Still cheap to try given `disable-custom-all-reduce`
  is already being tested for GLM-5.2, but temper expectations — this is
  weaker evidence than most other candidates in this table (documentation
  for the claim; inferred/unconfirmed for whether it actually helps).
- `VLLM_DISABLE_PYNCCL`: **open question, not yet resolved**. Determines
  whether vLLM's TP/EP collectives route through its own lightweight
  `pynccl` wrapper (default) or `torch.distributed`'s `ProcessGroupNCCL`.
  This matters directly for last session's `TORCH_NCCL_*` recommendation
  (flight recorder, `TORCH_NCCL_DESYNC_DEBUG`) — those env vars are consumed
  by `ProcessGroupNCCL`'s C++ implementation, so if vLLM's actual collective
  path for TP/EP goes through `pynccl` instead, they may not apply to the
  code path that matters here at all. No GitHub research or source read has
  resolved this yet — worth checking `vllm/distributed/device_communicators/`
  directly before relying on the `TORCH_NCCL_*` recommendation (documentation
  for the mechanism; open/unconfirmed for which path GLM-5.2 actually uses).
- `VLLM_WORKER_MULTIPROC_METHOD`: defaults to `fork`, and vLLM's own
  maintainers identified fork+Ray-actor as undefined behavior
  ([PR #14705](https://github.com/vllm-project/vllm/pull/14705)) — but
  `vendor/vllm-0.26.0/vllm/utils/system_utils.py`'s `_maybe_force_spawn()`
  explicitly detects `is_in_ray_actor()` and forces `spawn` regardless of
  the default, confirmed present in our installed version. Nothing to
  change here (proven, direct source read).
- `VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS=5` (default): per
  [#31252](https://github.com/vllm-project/vllm/issues/31252), a feature
  request describing exactly this pain point — "the shutdown function...
  only provides a fixed 5-second timeout for the EngineCore process,"
  insufficient for connector cleanup. Relevant to this project's own
  `ivllm-cancel.sh`/`tidy_up()` review earlier this session — worth knowing
  this timeout exists on vLLM's own side too, separate from ivllm's own
  force-cancel logic (documentation).
- `VLLM_GPU_NIC_PCIE_MAPPING`/`VLLM_NIC_SELECTION_VARS`: confirmed real via
  [PR #42083](https://github.com/vllm-project/vllm/pull/42083) ("Add support
  for per GPU worker RDMA NIC selection") — comma-separated `GPU_BDF=NIC_BDF`
  pairs plus a list of env vars to set per-worker from that mapping (e.g.
  `NCCL_IB_HCA`), library-agnostic (UCX/NVSHMEM/NCCL/UCCL). Explicitly
  targets "flat PCIe topology" multi-NIC nodes where automatic GPU-NIC
  discovery fails — the same problem class as Slingshot-11's 4-NIC/4-GPU
  layout. Reports a 4.83x transfer-time improvement in testing; merged, no
  reported production issues. This is a stronger, vLLM-native candidate for
  GPU-NIC affinity than the earlier `CUDA_DEVICE_ORDER` speculation, which
  turned out to address a different (heterogeneous-GPU) problem entirely
  (documentation, untested on this platform).
- `VLLM_ENABLE_STARTUP_PLAN`: no GitHub hits either way. Persists a
  per-node memory-profiling result "keyed by a hardware+config fingerprint"
  under `VLLM_CACHE_ROOT/startup_plan/`, auto-applied on later boots if the
  fingerprint matches. Same shape of concern as the already-open JIT-cache
  node-identity question in `active-issues.md` — not currently enabled, so
  not an active risk, but worth remembering if it's ever turned on
  (inferred).

| Variable | Value | Use | Context | Note |
|---|---|---|---|---|
| `NCCL_NET_GDR_LEVEL` | `PHB` | always (as of 2026-09-03) | multi-node | promoted to `vllm-env.sh`'s shared default; NCCL's own library default is `SYS` |
| `FI_OPT_CUDA_API_PERMITTED` | n/a | never (not settable via env var) | alternative deadlock avoidance to GDRCopy | `fi_setopt()` API, not an env var — see note |
| `NCCL_CUMEM_ENABLE` | `0` | sometimes | possibly-stale NCCL-version workaround | **re-test candidate** — see note |
| `NCCL_IB_PCI_RELAXED_ORDERING` | `1` | sometimes | | unproven value |
| `NCCL_P2P_NET_CHUNKSIZE` | `262144` | sometimes | | untested for hang, matches vendor's own next-power-of-2 example |
| `FI_CXI_RDZV_PROTO` | `alt_read` | sometimes | | longest stable GLM-5.2 run — **independently confirmed as HPE's own Slingshot-11 default**, see note |
| `FI_CXI_DEFAULT_TX_SIZE` | `1024` | sometimes (reconsider `2048`) | tested vs GLM-5.2 hang, no measurable difference | **HPE's own Slingshot-11 default is `2048`** — see note |
| `FI_CXI_RX_MATCH_MODE` | `software` | sometimes (reconsider `hybrid`) | tested vs GLM-5.2 hang, no measurable difference | **HPE's own Slingshot-11 default is `hybrid`** — see note |
| `NCCL_P2P_LEVEL` | untested | sometimes | intra-node GPU↔GPU analogue of `NCCL_NET_GDR_LEVEL` | **unvalidated, worth testing** — see note |
| `NCCL_OOB_NET_ENABLE` | untested | sometimes | routes comm-init bootstrap over the already-proven OFI/CXI path instead of sockets | **unvalidated** — init-time only, weak link to the live-serving hang |
| `NCCL_BUFFSIZE` | untested | sometimes | GPU↔GPU per-channel buffer size, default 4MiB | future perf/memory-pressure tuning, not a hang candidate |
| `NCCL_NTHREADS` | untested | sometimes | CUDA threads per collective kernel block | future perf tuning only, not a hang candidate |
| `NCCL_MULTI_RANK_GPU_ENABLE` | not applicable today | sometimes | multi-rank-per-GPU only (experimental, since 2.30) | not used — ivllm runs 1 rank/GPU — but see note |
| `VLLM_USE_RAY_COMPILED_DAG` | untested | sometimes | Ray executor internals | **new candidate from GitHub research** — see note |
| `VLLM_USE_RAY_WRAPPED_PP_COMM` | untested | sometimes | Ray executor internals | **new candidate from GitHub research** — see note |
| `VLLM_ENABLE_PCIE_ALLREDUCE` | untested | sometimes | allreduce path alternative | **new candidate from GitHub research** — see note |

Notes:
- `NCCL_NET_GDR_LEVEL=PHB`: cleanest isolated comparison in this project's
  history — `PHB` sustained ~8m19s of real GLM-5.2 serving before a hang vs
  the default `SYS`'s ~2m49s, otherwise identical config (~3x). Doesn't fix
  the hang, delays it (proven). **Open decision — now resolved in favor of
  promoting `PHB`.** HPE's own `shs-nccl-env` plugin
  ([HewlettPackard/shs-nccl-env](https://github.com/HewlettPackard/shs-nccl-env),
  `src/shs_nccl_env.cc`) — an official Slingshot-11 NCCL auto-config tool
  that sets defaults only if not already present — defaults `NCCL_NET_GDR_LEVEL`
  to `PHB` for *all* Slingshot-11 NCCL workloads generally, not anything
  GLM-5.2/vLLM-specific. `vllm-env.sh`'s shared default is still `SYS`, with
  `PHB` only applied per-job so far — given this is now vendor consensus, not
  just our own single-model finding, worth promoting `PHB` to the shared
  default rather than continuing to treat it as a job-level override
  (proven for the delay effect; documentation/recommendation for promoting
  it, from an independent vendor source). **GitHub research finding**:
  [vllm-project/vllm#26318](https://github.com/vllm-project/vllm/issues/26318)
  is a near-identical failure signature to ours — Slurm + Slingshot
  multi-node pipeline-parallel hang, closed unresolved ("not planned") —
  whose mitigation attempt bundled `NCCL_NET_GDR_LEVEL` alongside
  `NCCL_CROSS_NIC`, `FI_CXI_RDZV_THRESHOLD`/`_GET_MIN`,
  `FI_CXI_OFLOW_BUF_SIZE`/`_COUNT`, `NCCL_GRAPH_MIXING_SUPPORT=0`, and
  disabling `VLLM_USE_RAY_COMPILED_DAG`/`VLLM_USE_RAY_WRAPPED_PP_COMM` — see
  those two new candidates below. Worth reading in full (documentation).
- `FI_OPT_CUDA_API_PERMITTED`: the *other* documented mitigation for the
  same CUDA deadlock — restricts an endpoint from making CUDA API calls at
  all. **Not usable today**: this is an `fi_setopt(FI_OPT_ENDPOINT, ...)`
  call in application code (confirmed via `fi_endpoint(3)`), not an
  environment variable — ivllm has no mechanism to set it, and it isn't
  confirmed whether `aws-ofi-nccl` (the actual libfabric caller in this
  stack) exposes any equivalent knob. Documented here so the option isn't
  lost, not as an actionable recommendation (documentation).
- `NCCL_CUMEM_ENABLE=0`: **not actually "unproven" — this traces to a real,
  now-possibly-stale historical NCCL bug.** Per GitHub research:
  [vllm-project/vllm#5091](https://github.com/vllm-project/vllm/pull/5091)
  documents NCCL 2.19 auto-enabling cuMem allocation, which broke against
  CUDA graphs (~3 months of debugging per the PR author); vLLM shipped the
  `NCCL_CUMEM_ENABLE=0` override as the fix, which is where `vllm-env.sh`'s
  default comes from. But
  [vllm-project/vllm#24141](https://github.com/vllm-project/vllm/pull/24141)
  later **reverted** that override once NCCL 2.22.3 fixed the underlying bug
  upstream — the PR states leaving it enabled restores real perf
  optimizations that forcing `=0` gives up. `active-issues.md`'s
  `nccl-probe.sh` work already confirmed this platform runs **NCCL 2.30.4**
  (upgraded past 2.22.3) — meaning the original reason for `=0` may no
  longer apply here, and this could be a stale workaround costing
  performance rather than a needed fix. Separately,
  [vllm-project/vllm#28901](https://github.com/vllm-project/vllm/issues/28901)
  shows `NCCL_CUMEM_ENABLE=1` combined with NCCL symmetric memory breaking on
  NCCL 2.28+ (CUDA error during graph capture) — plausibly the same
  interaction behind `VLLM_ALLREDUCE_USE_SYMM_MEM=0` already being forced off
  in the Mandatory table above; worth understanding whether these two
  settings are addressing the same underlying issue before changing either
  independently. **Re-test candidate**: try `NCCL_CUMEM_ENABLE` unset/`1`
  now that NCCL is at 2.30.4, watching specifically for the symm-mem/graph-
  capture interaction from #28901 (documentation for the historical bug/fix;
  inferred for whether it's stale on this platform now).
- `NCCL_IB_PCI_RELAXED_ORDERING`: `vllm-env.sh`'s own inline comment is
  **"N.B. unproven value"** — carried over as-is, not independently
  re-verified. No corroborating vLLM issue/PR found in GitHub research
  either (inferred).
- `NCCL_P2P_NET_CHUNKSIZE=262144`: present in the latest GLM-5.2 config, not
  otherwise discussed or isolated in this project's investigation — an
  untested addition. Per `nccl.md` (`design/references/nccl.md`), the
  manual's own worked example for "the next power-of-2 step up from the
  131072 default" is literally `262144` — whoever set this picked the
  vendor's textbook next value, not an arbitrary number, though this still
  doesn't confirm any effect on the hang (documentation for the value's
  provenance; inferred for hang relevance).
- `FI_CXI_RDZV_PROTO=alt_read`: requires the paired `--network=disable_rdzv_get`
  `sbatch` flag to take effect at all (see `ivllm-environment.md`/
  `ivllm-serve.sh` — already promoted to production for both mp and ray
  paths). Missing that flag for a long stretch meant `alt_read` was silently
  half-applied; adding it produced the single longest/most stable GLM-5.2
  run recorded (proven, per [HPE's rendezvous-protocol docs](https://support.hpe.com/hpesc/public/docDisplay?docId=dp00005991en_us&page=user/rendezvous_protocol_configuration.html)).
  **Independently corroborated**: HPE's own `shs-nccl-env` plugin (see
  `NCCL_NET_GDR_LEVEL` note above) also defaults `FI_CXI_RDZV_PROTO` to
  `alt_read` for all Slingshot-11 NCCL workloads — this is genuinely
  reassuring confirmation that this project's single most concrete empirical
  finding lines up with vendor consensus, not something fragile or
  GLM-5.2-specific (documentation, independent vendor corroboration).
- `FI_CXI_DEFAULT_TX_SIZE`/`FI_CXI_RX_MATCH_MODE`: previously filed as
  settled "never" (Mandatory table) on the strength of one negative test —
  `2048`/`hybrid` together (per
  [uccl-project/uccl#956](https://github.com/uccl-project/uccl/issues/956)'s
  corruption-avoidance discussion) showed no measurable change against the
  GLM-5.2 hang either direction. **Reopened**: HPE's own `shs-nccl-env`
  plugin defaults to exactly this pair — `FI_CXI_DEFAULT_TX_SIZE=2048`,
  `FI_CXI_RX_MATCH_MODE=hybrid` — for the entire Slingshot-11 NCCL ecosystem,
  a far broader validation base than one vLLM-specific test. "No measurable
  difference" in that narrow test doesn't rule out real value under
  different traffic patterns — `hybrid`'s specific purpose is avoiding
  match-resource starvation under heavy unexpected-message load, which is
  plausible for an 8-GPU MoE model's dispatch/combine pattern and wasn't
  specifically stress-tested. Since the one test run found no regression
  either, promoting both to match vendor defaults is low-risk and worth
  doing rather than continuing to diverge from vendor consensus without a
  specific reason (proven for the earlier negative result; recommendation
  for promoting, from an independent vendor source).
- `NCCL_SOCKET_IFNAME=hsn` (set in `common-env.sh`, documented in
  `ivllm-environment.md`'s Networking configuration table): HPE's
  `shs-nccl-env` plugin instead sets this explicitly to
  `hsn0,hsn1,hsn2,hsn3`. **Confirmed not a functional gap** — per `nccl.md`'s
  own documented semantics, `NCCL_SOCKET_IFNAME` is a *prefix* filter by
  default (`eth` matches `eth0, eth1, …`), so the bare `hsn` prefix already
  matches all four Cassini NICs identically to HPE's explicit list. HPE's
  version is likely just more defensive/portable for a generic tool that
  can't assume a fixed NIC count on every Slingshot deployment it runs on.
  No change needed here (documentation, confirmed via vendor comparison).
- `NCCL_P2P_LEVEL`: per `nccl.md`, this is the exact intra-node (GPU↔GPU)
  counterpart to `NCCL_NET_GDR_LEVEL` (NIC↔GPU) — same "maximum topological
  distance before falling back" idea, `LOC`/`NVL`/`PIX`/`PXB`/`PHB`/`SYS`.
  Auto-selected if unset, and current logs already show real P2P/IPC hops
  happening (not stuck at `LOC`), so intra-node NVLink looks healthy today.
  Worth testing anyway by direct analogy: `NCCL_NET_GDR_LEVEL=PHB` is this
  project's cleanest proven result (~3x delay-not-fix on the hang), so
  forcing `NCCL_P2P_LEVEL=PHB`/`PXB` would test whether the same
  delay-not-fix pattern shows up intra-node too — cheap to try, adds real
  evidence either way (inferred, unvalidated). **GitHub research**: real
  vLLM precedent exists but is mixed —
  [#33041](https://github.com/vllm-project/vllm/issues/33041) tried
  `NCCL_P2P_LEVEL=SYS` on a Blackwell NCCL-init hang, "made no difference";
  [#40608](https://github.com/vllm-project/vllm/issues/40608) (Kimi
  K2.5/K2.6 Blackwell) lists `NCCL_P2P_LEVEL=SYS` +
  `VLLM_ENABLE_PCIE_ALLREDUCE=1` together as a reported working config
  (`VLLM_ENABLE_PCIE_ALLREDUCE` is a new candidate, not previously tracked
  here — see below); [#45094](https://github.com/vllm-project/vllm/issues/45094)
  (TP=2 PP=2 decode deadlock at 0 tok/s after an earlier OOM) has
  `NCCL_P2P_LEVEL=NVL` in its env dump but doesn't implicate it directly
  (documentation).
- `VLLM_USE_RAY_COMPILED_DAG` / `VLLM_USE_RAY_WRAPPED_PP_COMM`: surfaced via
  [#26318](https://github.com/vllm-project/vllm/issues/26318) (see
  `NCCL_NET_GDR_LEVEL` note above) — disabled alongside a bundle of
  NCCL/libfabric tweaks in an unresolved Slurm+Slingshot multi-node PP hang
  report with a very similar shape to ours. Not investigated at all in this
  project yet. Given ivllm's `distributed-backend-executor: ray`, these are
  Ray-executor-internal flags genuinely worth trying, independent of
  whichever NCCL/libfabric env vars are set (documentation, untested here).
- `VLLM_ENABLE_PCIE_ALLREDUCE`: not previously tracked in this doc at all —
  surfaced via [#40608](https://github.com/vllm-project/vllm/issues/40608)
  as part of a reported-working Blackwell multi-node config alongside
  `NCCL_P2P_LEVEL=SYS`. An alternative allreduce path; untested on this
  platform (documentation).
- `NCCL_OOB_NET_ENABLE`: since 2.23, routes the out-of-band bootstrap
  allgather (communicator init) over NCCL net — i.e. over the already-proven
  OFI/CXI path — instead of the default socket-based OOB. Architecturally
  interesting but the hang signature here happens mid-serving
  (`EngineCore` blocked on `shm_broadcast` during live traffic), not at
  communicator init, so this doesn't target the observed failure directly.
  Worth knowing about for init-time robustness/future tuning, low priority
  for the current hang (documentation, unvalidated).
- `NCCL_BUFFSIZE` / `NCCL_NTHREADS`: real performance-tuning levers
  (per-channel GPU↔GPU buffer size, default 4MiB; CUDA threads per
  collective kernel block, default 512) with no plausible mechanism linking
  either to a permanent one-sided wait — pure future optimization/memory-
  pressure candidates, not hang candidates. Given this stack is already
  memory-tight (INT4 GLM-5.2, `cpu-offload-gb`, KV-cache pressure), raising
  `NCCL_BUFFSIZE` specifically could worsen OOM-adjacent instability rather
  than help — test with that in mind if ever revisited (documentation).
- `NCCL_MULTI_RANK_GPU_ENABLE`: experimental (2.30+), off by default, requires
  the *application* to deliberately co-locate multiple ranks per GPU device —
  not something ivllm does today (standard 1 rank/GPU for TP/DP). Not
  applicable now, but worth remembering: per `nccl.md`, "failing to [size
  resources correctly] will result in a hang in NCCL" — a literal, named,
  self-documented hang mechanism. Relevant if any future Wide-EP/multi-rank
  disaggregation scheme (see `feature-development.md`'s "Multinode
  disaggregation strategies") ever puts more than one rank on a GPU
  (documentation).

### Compilers (e.g. CUDA, Torch etc)

*(vLLM/Torch/DeepGEMM/FlashInfer-specific flags from `vllm-env.sh` only —
CUDA/NVHPC paths and compiler selection live in `common-env.sh`, see
`ivllm-environment.md`.)*

| Variable | Value | Use | Context | Note |
|---|---|---|---|---|
| `FLASHINFER_HEAD_DIMS` | `128` | sometimes | | source: "unvalidated" |
| `FLASHINFER_POS_ENCODING_MODES` | `0` | sometimes | | source: "unvalidated" |
| `VLLM_ENABLE_INDUCTOR_MAX_AUTOTUNE` | `0` | sometimes | | maybe safe to re-enable, untested |
| `TORCHINDUCTOR_AUTOTUNE_REMOTE_CACHE` | `0` | sometimes | | same permissions workaround |
| `TORCHINDUCTOR_FX_GRAPH_REMOTE_CACHE` | `0` | sometimes | | same permissions workaround |
| `TORCHINDUCTOR_AUTOGRAD_REMOTE_CACHE` | `0` | sometimes | | same permissions workaround |

Notes:
- `FLASHINFER_HEAD_DIMS`/`FLASHINFER_POS_ENCODING_MODES`: `vllm-env.sh`'s
  own comment flags these as "limit runtime combinatorics ... unvalidated"
  (recipe).
- The three `TORCHINDUCTOR_*_REMOTE_CACHE` vars + `VLLM_ENABLE_INDUCTOR_MAX_AUTOTUNE`
  were set to work around a permissions issue from shared autotuning caches
  defaulting to a location owned by the original cacher. Now on per-user
  caches, so `vllm-env.sh`'s own comment notes these "may not be strictly
  needed" any more — possibly safe to re-enable, not re-tested (inferred).

### Backends (e.g. DeepGEMM)

Nothing currently open here — `EP_DISABLE_GIN` are settled,
see Mandatory above.

### CUDA driver

*(Raw CUDA driver/runtime environment variables per NVIDIA's own reference,
`design/references/cuda-env.md` — independent of vLLM/Torch/NCCL. None of
these are currently set anywhere in ivllm — confirmed via grep.)*

| Variable | Value | Use | Context | Note |
|---|---|---|---|---|
| `CUDA_FORCE_PRELOAD_LIBRARIES` | `1` | sometimes | JIT/NVVM library preload, threading | **unvalidated, strong candidate** — see note |
| `CUDA_MODULE_LOADING` | `EAGER` | sometimes | vs default `LAZY` kernel/data loading | **unvalidated candidate**, JIT-stall theme — see note |
| `CUDA_CACHE_DISABLE` / `CUDA_CACHE_PATH` / `CUDA_CACHE_MAXSIZE` | unset today | sometimes | driver-level PTX→CUBIN cache, relevant under CUDA forward-compat | **real gap, unmanaged today** — see note |
| `CUDA_DEVICE_ORDER` | `PCI_BUS_ID` | sometimes | GPU enumeration order vs GPU-NIC affinity | untested, future optimization |
| `CUDA_SCALE_LAUNCH_QUEUES` | untested | sometimes | launch-queue depth scaling | future perf tuning only, untested |
| `CUDA_GRAPHS_USE_NODE_PRIORITY` | untested | sometimes | CUDA graph node scheduling priority | vLLM already uses CUDA graphs heavily (`cudagraph-metrics: true`) — future perf tuning |
| `CUDA_BINARY_LOADER_THREAD_COUNT` | untested | sometimes | parallel device-binary loading, default 1 thread | pairs with `CUDA_MODULE_LOADING=EAGER` if that's ever tried |

Notes:
- `CUDA_FORCE_PRELOAD_LIBRARIES=1`: per `cuda-env.md`, verbatim: *"Setting
  this environment variable is necessary to avoid certain deadlock
  situations involving multiple threads."* This is a third, independent,
  NVIDIA-documented deadlock mechanism turned up this session — distinct
  from the libfabric CUDA-deadlock/GDRCopy story above and from
  `FI_CXI_ENABLE_TRIG_OP_LIMIT` (libfabric's own triggered-op deadlock
  guard) — this one is at the raw CUDA driver/JIT-preload layer, not the
  network layer. Disabled by default (`0`). Given this project's whole hang
  signature is a multi-threaded, multi-process stall inside a heavily
  JIT-compiling stack (Triton/DeepGEMM kernels compiling on many worker
  processes concurrently), this is a genuine, concrete, cheap-to-try
  candidate — costs only a slightly larger memory footprint/init time
  (documentation for the mechanism; inferred for relevance to this
  project's hang). **GitHub research**: zero mentions found anywhere in
  vLLM's issue/PR history, either way — neither confirmed nor refuted by
  vLLM's own community, genuinely unexplored territory there.
- `CUDA_MODULE_LOADING=EAGER`: default is `LAZY` — kernel/module code loads
  on first invocation, meaning the *first* request that hits a rarely-used
  code path (an under-utilized MoE expert, a rarely-hit fused-kernel
  variant) pays a load-time cost mid-serving, not at startup. This is a
  different mechanism than the already-observed "Triton kernel JIT
  compilation during inference" warning (that's Triton's own JIT, not CUDA
  driver module loading) — but it's the same *shape* of failure (a surprise
  stall triggered by first-use of code deep into a live serving run) via a
  different subsystem. `EAGER` trades higher startup time/memory for
  eliminating this specific first-use variance entirely — cheap to test in
  isolation as a way to rule this specific mechanism in or out (inferred,
  unvalidated).
- `CUDA_CACHE_DISABLE`/`CUDA_CACHE_PATH`/`CUDA_CACHE_MAXSIZE`: this is the
  CUDA **driver's own** on-disk PTX→CUBIN JIT cache (default
  `~/.nv/ComputeCache`) — a completely different, lower-level cache than any
  of the 8 application-level JIT caches ivllm already manages carefully in
  `ivllm-environment.md`'s Cache setup section (Triton/DeepGEMM/FlashInfer/
  TorchInductor/etc, all redirected to job-scoped storage with a documented
  tar-race fix). None of `CUDA_CACHE_DISABLE`/`_PATH`/`_MAXSIZE` are set
  anywhere in ivllm — confirmed via grep. Given this project explicitly
  relies on CUDA forward-compatibility (NVHPC/CUDA 12.9 forward-compat, see
  `ivllm-environment.md`) — precisely the scenario where the driver may need
  to JIT-recompile embedded PTX for the actual GPU architecture — this cache
  is genuinely in play, defaults to an un-job-scoped, un-redirected location
  in the user's home directory, and could in principle hit the same
  concurrent-write race class already fixed for the other 8 caches. Worth a
  deliberate decision (redirect `CUDA_CACHE_PATH` into the job-scoped cache
  dir like everything else, or confirm it's low-volume enough not to matter)
  rather than leaving it as an unexamined gap (documentation for the
  mechanism; inferred for whether the gap is actually load-bearing).
- `CUDA_DEVICE_ORDER=PCI_BUS_ID`: default is `FASTEST_FIRST` (heuristic-based).
  On a 4-identical-GPU GH200 node with known GPU-NIC affinity mattering for
  Slingshot/CXI performance (4 NICs, 4 GPUs, NUMA-local pairing), a
  heuristic enumeration order isn't guaranteed to match physical PCI
  topology the way `PCI_BUS_ID` deterministically would — relevant to the
  NUMA/GPU-affinity binding work already touched on elsewhere in this
  project. Untested, future optimization candidate rather than a bug fix
  (inferred). **GitHub research finding — downgrade the framing above**:
  vLLM's own use case for this var
  ([#19741](https://github.com/vllm-project/vllm/issues/19741),
  [#7472](https://github.com/vllm-project/vllm/issues/7472)) is specifically
  *heterogeneous*-GPU device-ordering (wrong-compute-capability GPU picked as
  device 0) — not applicable to this platform's homogeneous GH200 fleet. The
  GPU-NIC-affinity reasoning above is this project's own hypothesis, not
  something vLLM's community has validated for that purpose (documentation
  for vLLM's actual use case; inferred/unconfirmed for the affinity angle).
- `CUDA_LOG_FILE`: moved to `ivllm-environment.md`'s Debugging flags
  section (Level 3) — it's a diagnostic-output toggle, not a job-tunable
  behavior override, so it belongs with the rest of the `IVLLM_DEBUG_LEVEL`
  design rather than here.
- `CUDA_SCALE_LAUNCH_QUEUES`/`CUDA_GRAPHS_USE_NODE_PRIORITY`/`CUDA_BINARY_LOADER_THREAD_COUNT`:
  legitimate performance-tuning levers for vLLM's CUDA-graph-heavy,
  deeply-pipelined execution model, but untested and with no plausible
  mechanism linking any of them to the hang's synchronization signature —
  future optimization backlog, not hang candidates (documentation).

### PyTorch CUDA runtime

*(Raw `PYTORCH_*`/`TORCH_*` (non-NCCL)/cuBLAS/cuDNN vars, per
`design/references/torch-cuda.md` — independent of the `TORCH_NCCL_*` family
above. None of these are currently set anywhere in ivllm — confirmed via
grep.)*

| Variable | Value | Use | Context | Note |
|---|---|---|---|---|
| `PYTORCH_ALLOC_CONF` | untested | sometimes | CUDA caching-allocator config (`expandable_segments`, `max_split_size_mb`, etc.) | worth investigating for memory pressure — **but see `expandable_segments` warning in note** |
| `PYTORCH_NVML_BASED_CUDA_CHECK` | `1` | sometimes | avoids CUDA-init errors in forked processes, uses NVML instead of CUDA runtime | untested, plausible for Ray/MP-executor startup robustness |
| `PYTORCH_NO_CUDA_MEMORY_CACHING` | `1` | never in production | disables the CUDA caching allocator entirely | debug-only, severe perf cost — only for ruling out allocator-related hypotheses |
| `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE` | `1` | sometimes | forces TF32 for cuBLAS regardless of `set_float32_matmul_precision` | **confirmed working fix elsewhere in vLLM** — see note |
| `NVIDIA_TF32_OVERRIDE` | untested | sometimes | global TF32 disable across all kernels | same numerical-precision family as above, untested |
| `CUBLAS_WORKSPACE_CONFIG` | untested | sometimes | cuBLAS per-allocation workspace sizing | future perf/memory tuning — also a determinism knob, see note |
| `CUBLASLT_WORKSPACE_SIZE` | untested | sometimes | cuBLASLt workspace sizing — more relevant than raw cuBLAS for quantized GEMM kernels | untested |
| `TORCH_CUDNN_V8_API_LRU_CACHE_LIMIT` / `_DISABLED` / `_DEBUG` | n/a | sometimes | cuDNN v8 API cache/behavior | **relevance unconfirmed** — this stack's attention/GEMM kernels (FlashAttention/FlashInfer/marlin/deep_gemm) may not exercise cuDNN's conv-focused API path at all |
| `CUDNN_CONV_WSCAP_DBG` / `CUDNN_ERRATA_JSON_FILE` | n/a | sometimes | cuDNN workspace/errata debug | same cuDNN-relevance caveat as above |

Notes:
- `PYTORCH_ALLOC_CONF` (formerly `PYTORCH_CUDA_ALLOC_CONF`): the standard
  PyTorch CUDA caching-allocator tuning knob (`expandable_segments`,
  `max_split_size_mb`, `garbage_collection_threshold`, etc.) — genuinely
  worth investigating given this stack's memory-pressure profile (INT4
  GLM-5.2, `cpu-offload-gb`, ~1M-context KV-cache sizing, the ~7GB
  uncompressed JIT-cache growth already observed). Not yet touched at all in
  this project — a real gap for future memory-pressure/OOM-adjacent work,
  independent of the hang investigation (documentation for the mechanism;
  inferred for relevance here). **GitHub research — important warning, not
  just a data point**:
  [#29544](https://github.com/vllm-project/vllm/issues/29544) "expandable_segments:
  True causes vLLM EngineCore initialization to fail" — on multi-node TP4,
  setting `expandable_segments:True` causes `RuntimeError: cancelled` thrown
  from **`shm_broadcast.acquire_read`** during EngineCore's KV-cache init,
  → SIGABRT (closed not-planned, `expandable_segments:False` works fine).
  This directly touches the same `shm_broadcast` mechanism this project's
  own hang lives in — **do not enable `expandable_segments` on this
  platform without first checking whether it reproduces this crash on
  GLM-5.2's exact multi-node TP config**, even though it isn't currently set
  anywhere in ivllm. Separately, [#40291](https://github.com/vllm-project/vllm/issues/40291)
  and [#29360](https://github.com/vllm-project/vllm/issues/29360) show
  `expandable_segments:True` already set and OOM still occurring (FP8/warmup
  fragmentation) — i.e. it isn't even a reliable OOM fix elsewhere either.
  Any future investigation of `PYTORCH_ALLOC_CONF` here should start with
  `max_split_size_mb`/`garbage_collection_threshold` and treat
  `expandable_segments` as a known risk, not the obvious first setting to
  try (documentation — directly relevant, not just adjacent).
- `PYTORCH_NVML_BASED_CUDA_CHECK`: per `torch-cuda.md`, explicitly
  "helpful if forked processes fail with a CUDA initialization error." Given
  Ray's process-per-worker model and this project's separate history of
  hard CUDA-context crashes (the `mnnvl` allreduce-backend crash, see VLLM
  section above), this is a plausible startup-robustness lever — untested,
  not linked to the live-serving hang specifically (inferred).
- cuDNN-family vars (`TORCH_CUDNN_V8_API_*`, `CUDNN_CONV_WSCAP_DBG`,
  `CUDNN_ERRATA_JSON_FILE`): flagged with a relevance caveat rather than
  filed as irrelevant outright — this project hasn't confirmed whether
  cuDNN's conv-focused API is exercised anywhere in this MoE LLM inference
  stack at all (attention/GEMM here mostly route through FlashAttention/
  FlashInfer/marlin/deep_gemm, not cuDNN convolutions). Worth a quick check
  before investing further here (inferred, unconfirmed). No GitHub hits for
  any of these five vars either way.
- `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1`: **confirmed working fix elsewhere in
  vLLM** — [#31579](https://github.com/vllm-project/vllm/issues/31579)
  "`VLLM_FLOAT32_MATMUL_PRECISION=tf32` does not set cublas tf32 matmul": a
  reporter confirms this env var eliminates both a warning and a legacy/new
  TF32-API conflict error that vLLM's own flag fails to trigger correctly.
  Numerical-precision/performance relevance, not hang-related (documentation).
- `CUBLAS_WORKSPACE_CONFIG`: minor GitHub precedent —
  [#17166](https://github.com/vllm-project/vllm/issues/17166)
  (non-reproducible outputs with a fixed seed) suggests
  `CUBLAS_WORKSPACE_CONFIG=:4096:8` for deterministic cuBLAS. Determinism
  matters less for inference serving than training, but worth having on
  record (documentation).

### Triton

*(Per `design/references/triton.md`, Triton's own README — mostly a
build-from-source/development guide, not a production deployment doc. ivllm
consumes a pre-built Triton via vLLM's own pip dependency, so most of these
knobs — `TRITON_BUILD_WITH_CLANG_LLD`/`_CCACHE`, `MAX_JOBS`,
`LLVM_BUILD_DIR`, and the MLIR/LLVM IR-dump/debug family
(`MLIR_ENABLE_DUMP`, `LLVM_IR_ENABLE_DUMP`, `TRITON_REPRODUCER_PATH`,
`TRITON_ENABLE_LLVM_DEBUG`, `TRITON_LLVM_DEBUG_ONLY`, `TRITON_ENABLE_ASAN`
(AMD-only anyway), `TRITON_KERNEL_DUMP`/`_DUMP_DIR`/`_OVERRIDE`/`_OVERRIDE_DIR`,
`TRITON_INTERPRET`, `TRITON_FRONT_END_DEBUGGING`, `TRITON_DISABLE_LINE_INFO`,
`USE_IR_LOC`, `LLVM_EXTRACT_DI_LOCAL_VARIABLES`, `MLIR_ENABLE_TIMING`/
`LLVM_ENABLE_TIMING`, `MLIR_ENABLE_DIAGNOSTICS`) — are either build-time-only
(not applicable to a pip-installed deployment) or deep compiler-internals
debugging tools with no plausible production/tuning role. Listed here as a
single group rather than individually since none warrant their own row.*

| Variable | Value | Use | Context | Note |
|---|---|---|---|---|
| `PTXAS_OPTIONS` | untested | sometimes | passes flags directly to the `ptxas` assembler | **directly relevant to the existing ptxas-subprocess-stall hypothesis** — see note |
| `TRITON_ALWAYS_COMPILE` | `1` | never in production | forces recompilation regardless of cache hit | debug-only — would defeat all of ivllm's JIT-cache management, massive overhead |
| `TRITON_HOME` | n/a | sometimes | root of the `.triton` directory (cache + build-time downloads) | **needs disambiguation from `TRITON_CACHE_DIR`** — see note |
| `TRITON_DEFAULT_FP_FUSION` / `TRITON_F32_DEFAULT` | untested | sometimes | numerical fusion/precision defaults for `tl.dot` | future perf/numerics tuning, untested |

Notes:
- `PTXAS_OPTIONS`: this project's own JIT-compile-proximity hypothesis
  (`active-issues.md`, MiniMax-M3 investigation) already established that "a
  JIT compile shells out to `ptxas`/a C compiler as a real subprocess... and
  can genuinely stall the compiling rank for seconds." `PTXAS_OPTIONS` is the
  one documented way to influence that subprocess's own behavior (e.g.
  optimization level) directly — worth investigating whether any `ptxas`-side
  tuning (or just confirming what flags are already implicitly in effect)
  sheds light on why that stall duration varies, rather than treating the
  subprocess as an opaque black box (inferred — directly connects to an
  already-established mechanism, but untested as a lever). No GitHub hits
  found for this exact var in vLLM's history (checked against two known
  `ptxas`-failure issues, #4207 and #27542 — neither mentions it).
- `TRITON_PRINT_AUTOTUNING`: moved to `ivllm-environment.md`'s Debugging
  flags section ("Possible future additions") — it's a diagnostic-output
  toggle, not a job-tunable behavior override.
- `TRITON_HOME` vs `TRITON_CACHE_DIR`: `ivllm-environment.md` already
  redirects `TRITON_CACHE_DIR` to the job-scoped `$localdir/triton`
  (confirmed, `set_jit_caches()`). `TRITON_HOME` per `triton.md` is described
  as "the location of the `.triton` directory where Triton's cache is
  located **and downloads are stored during the build**" — worded primarily
  as a build-time concept (default: user's home directory). Not established
  whether `TRITON_CACHE_DIR` is a strict subdirectory override of
  `TRITON_HOME` at runtime, or whether an unset `TRITON_HOME` could still
  cause some Triton runtime path to fall back to `~/.triton` alongside the
  already-redirected cache — worth a quick source check rather than assuming
  the existing redirect is complete (documentation for `TRITON_CACHE_DIR`;
  inferred/open question for `TRITON_HOME`'s runtime role).
- `TRITON_ALWAYS_COMPILE=1`: genuinely useful as a one-off diagnostic to
  confirm cache-hit assumptions, but must never be left set — it would force
  every kernel invocation to recompile, defeating the entire JIT-cache
  infrastructure ivllm already carefully manages (documentation).

### Irrelevant to Isambard

| Variable | Why irrelevant |
|---|---|
| `NCCL_ALGO=NVLSTree` (or any NVLink-SHARP/NVLS forcing) | No real NVSwitch multicast hardware on this platform — confirmed via `0 nvls channels` at NCCL init, every run. Likely reproduces the same crash seen forcing `mnnvl` (untested directly) |
| `NCCL_PXN_DISABLE` | PXN exists for topologies with fewer NICs than GPUs; this platform has one Cassini NIC per GPU (confirmed: "found 4 nics" per 4-GPU node in NCCL logs) — nothing for it to proxy regardless of setting |
| `NCCL_MNNVL_ENABLE` (any node count, single or multi) | Requires an NVSwitch fabric spanning *physical node* boundaries with a configured IMEX domain (per `nccl.md`: "MNNVL requires a fully configured and operational IMEX domain for all the nodes that form the NVLink domain"). Isambard's GH200 nodes connect to each other over Slingshot-11/Cassini (CXI), not an inter-node NVSwitch fabric — there is no IMEX domain here, at any scale. Architecturally inert regardless of value; leave at `0`/unset |
| `NCCL_IB_DISABLE` | Disables NCCL's internal IB-verbs transport, which was never a candidate here — Cassini NICs speak libfabric, not verbs (confirmed: `NET/OFI Selected Provider is cxi` in NCCL logs, `active-issues.md`). Pure no-op |
| `NCCL_SOCKET_NTHREADS` / `NCCL_NSOCKS_PERTHREAD` | Tune NCCL's internal Socket *data* transport, which also isn't in use — the GPU data path goes through the OFI/CXI plugin, not raw sockets. Sockets are only used for low-volume bootstrap/rank rendezvous, which these throughput knobs don't affect |
| `NCCL_NET` / `NCCL_NET_PLUGIN` | Already resolve correctly via auto-detection (confirmed `cxi` selected, `aws-ofi-nccl`'s `libnccl-net.so` found without help) — only useful if auto-detection were picking the wrong plugin, which it isn't |
| `CUDA_LAUNCH_BLOCKING` | Forces fully synchronous kernel launches — a legitimate interactive debugging tool (pinpoints the exact API call behind an error) but not a production lever; the perf cost is severe. Belongs in a manual debugging session, not `env:` blocks |
| `CUDA_DEVICE_WAITS_ON_EXCEPTION` | Requires an attached `cuda-gdb` session to be useful at all — interactive debugging only, not applicable to batch SLURM jobs |
| `CUDA_DISABLE_PERF_BOOST` | Power-saving knob (reduces GPU pstate boosting) — this platform's goal is throughput/latency, not power savings; actively counter to the use case |
| `CUDA_AUTO_BOOST` | Deprecated by NVIDIA itself — superseded by `nvidia-smi --applications-clocks` |
| `CUDA_DEVICE_DEFAULT_PERSISTING_L2_CACHE_PERCENTAGE_LIMIT` | Only takes effect under CUDA MPS (`nvidia-cuda-mps-control -d`) — ivllm doesn't use CUDA MPS anywhere |

## Vllm serve configuration

N.B. only `vllm serve` flags, not environment variables — see the previous
top-level section for those.

### Mandatory / must not use

| Flag | Value | Use | Context | Note |
|---|---|---|---|---|
| `enable-flashinfer-autotune` | `false` | never | bare yaml key | silently ignored, use `no-enable-flashinfer-autotune` |
| `numa-bind` | `true` | never | `distributed-backend-executor: ray` | structurally no-op under Ray |

Notes:
- `enable-flashinfer-autotune: false` as a bare top-level yaml key is
  **silently ignored** — confirmed upstream vLLM bug (`active-issues.md`
  "Known Issues"): vLLM's `--config` yaml loader drops any bare `key: false`
  for a tri-state (`bool | None`) flag, and this one resolves `None` → `True`
  via the active `--optimization-level` preset, so the bare-`false` form
  keeps flashinfer's autotune warmup running regardless — root cause of a
  reproducible startup OOM chased across multiple MiniMax-M3 sessions.
  **Use instead**: `no-enable-flashinfer-autotune: true`, or wrap as
  `kernel-config: '{"enable_flashinfer_autotune": false}'` (proven, reported
  upstream to `vllm-project/vllm`).
- `numa-bind: true` is **confirmed structurally ineffective** under
  `RayExecutorV2` — `configure_subprocess()` (`vllm/utils/numa_utils.py`) is
  only wired into `MultiprocExecutor`'s init path; `RayExecutorV2` builds
  its worker actors independently and never calls it. Dead weight for any
  `distributed-backend-executor: ray` job on vLLM 0.26.0 (proven,
  `active-issues.md` GLM-5.2 entry; see `architecture.md`'s MP vs Ray
  section).

### Compilation

| Flag | Value | Use | Context | Note |
|---|---|---|---|---|
| `compilation-config` | `'{"pass_config": {"fuse_allreduce_rms": false}}'` | sometimes | Qwen3.5/DeepSeek-V3.2-style compile-pass allreduce | see `IVLLM_DISABLE_FLASHINFER` |

Notes:
- Disables the fused-allreduce-RMSNorm compile pass — the correct lever for
  models whose allreduce is routed through vLLM's compile pass
  (`AllreduceFusionPass` — Qwen3.5, DeepSeek-V3.2), as opposed to
  `IVLLM_DISABLE_FLASHINFER` (env-var table above) for models like
  MiniMax-M3/GLM-5.2 that call the fused op directly, bypassing the compile
  pass entirely. Which one applies is model-dependent (proven,
  `active-issues.md` MiniMax-M3 entry).
- **Decision (Rob), 2026-09-03**: by this entry's own analysis, GLM-5.2
  belongs to the `IVLLM_DISABLE_FLASHINFER` group, not the compile-pass
  group — `glm-5.2-743b-int4.yaml` sets `fuse_allreduce_rms: false` anyway
  (labelled "Multi-node compilation fix" in the yaml, added 2026-08-12
  during the NCCL hang investigation, not tied to a specific proven need in
  this doc). Genuinely uncertain whether it does anything for GLM-5.2 —
  leaving it in place per "if it isn't broken, don't fix it" rather than
  testing removal right now. Documented as an open question, not a
  recommendation either way.

### Scheduling

| Flag | Value | Use | Context | Note |
|---|---|---|---|---|
| `no-async-scheduling` | `true` | sometimes | multi-node `shm_broadcast` hang | changes failure signature, doesn't fix it |

Notes:
- One of two changes confirmed to measurably help the GLM-5.2
  `shm_broadcast` hang — changes *which* failure signature you hit (fast
  hang when async is on, vs. many-minutes-of-real-serving before a
  different signature when off) — does not fix the underlying hang. Worth
  trying on any similarly-affected multi-node job (proven, `active-issues.md`
  GLM-5.2 entry).
- **Update, 2026-09-03**: the GLM-5.2 hang itself is now resolved (was a
  silently-reverted NCCL pin — `nvidia-nccl-cu12` was actually 2.28.9, not
  the 2.30.4 fix already proven for the near-identical Nemotron-3-Ultra hang;
  see `active-issues.md`'s resolved GLM-5.2 entry). This flag was only ever
  shown to change the hang's *symptom*, never proven necessary for GLM-5.2
  once the real cause is fixed.
- **Decision (Rob), 2026-09-03**: leaving this set as-is in
  `glm-5.2-743b-int4.yaml` for now — "if it isn't broken, don't fix it."
  Genuinely uncertain whether it's still needed, not verified either way;
  documented here so a future reader knows this is a live open question, not
  a settled recommendation, should GLM-5.2 need retuning for throughput
  later.

### Backends

| Flag | Value | Use | Context | Note |
|---|---|---|---|---|
| `distributed-backend-executor` | `ray` or `mp` | sometimes | multi-node | see `architecture.md` MP vs Ray |
| `enable-expert-parallel` | model-dependent | sometimes | MoE models | ruled out as MiniMax-M3 crash cause |
| `moe-backend` | model/quant-dependent | sometimes | | see note below for observed values |
| `all2all-backend` | model-dependent | sometimes | | `deepep_v2` confirmed broken, see note |
| `disable-custom-all-reduce` | `true` | sometimes | GLM-5.2, added chasing the NCCL hang | tested and ruled out as the hang's cause — see note |

Notes:
- `disable-custom-all-reduce`: explicitly tested against the GLM-5.2 hang
  (2026-08-12, `active-issues.md`) and shown to have zero effect — the hang's
  actual cause (NCCL version, see the resolved GLM-5.2 entry) was unrelated.
  **Decision (Rob), 2026-09-03**: leaving it set in
  `glm-5.2-743b-int4.yaml` anyway, per "if it isn't broken, don't fix it" —
  not proven necessary, not proven harmful, just not being touched right now.
- `distributed-backend-executor`: full comparison lives in `architecture.md`'s
  dedicated "MP vs Ray" section, not duplicated here — short version, both
  hit the same `shm_broadcast` multi-node hang, and `ray` additionally loses
  `numa-bind` support entirely (proven).
- `enable-expert-parallel`: ruled out as the cause of the GLM-5.2/MiniMax-M3
  illegal-memory-access crash family — disabling it did not stop the crash
  (proven, `active-issues.md` MiniMax-M3 step 20).
- `moe-backend`, observed values across `examples/*.yaml`: `marlin`
  (AWQ-INT4 — GLM-5.2-INT4, Nemotron-3-Ultra), `deep_gemm` (GLM-5.2-FP8,
  Solar-Open2), `triton` (Qwen3.6/Qwen3-Coder long-context), `humming`
  (DeepSeek-V4-Pro — config marked `lifecycle: failing`, not actually
  runnable, so untested in practice), `flashinfer_b12x` (Qwen3.6-35B-A3B-FP8).
  No single recommended default — pick per quantization scheme, following
  whichever existing example config matches (recipe/proven mixed, per
  model).
- `all2all-backend`, observed values: `allgather_reducescatter` (safe
  default, supports all quantization types — used after `deepep_v2` failed,
  see below), `deepep_low_latency` (GLM-5.2-FP8, Solar-Open2,
  DeepSeek-V4-Pro). **`deepep_v2` is confirmed broken**: the installed
  `deep_ep` is UCCL-EP's compatibility wrapper, which genuinely lacks the v2
  `ElasticBuffer` API — `AssertionError: DeepEP v2 (ElasticBuffer) not
  available` (proven, `active-issues.md` MiniMax-M3 step 1).
  `flashinfer_nvlink_two_sided` appears commented-out/untested in one
  example config — no evidence either way.

### Offloading

| Flag | Value | Use | Context | Note |
|---|---|---|---|---|
| `cpu-offload-gb` | per-GPU, not per-node/total | sometimes | untested on real hardware | see note — vLLM source-confirmed semantics |
| `offload-backend` | `uva` | sometimes | GH200 NVLink-C2C | untested on real hardware |
| `enable-prefix-caching` | `true` | sometimes | generally recommended | tested + ruled out for the GLM-5.2 hang specifically |

Notes:
- `cpu-offload-gb` is **per GPU** — confirmed directly from the vLLM source
  docstring (`vllm/config/offload.py`): "The space in GiB to offload to CPU,
  **per GPU**". Not yet validated on real Isambard hardware — the one
  example using a real number (`cpu-offload-gb: 128`, DeepSeek-V4-Pro) is
  marked `lifecycle: failing`/"NOT VIABLE" for an unrelated node-count
  reason, so that figure is a paper value from `design/references/`'s
  offloading research notes, not an empirical one (documentation for the
  per-GPU semantics; inferred for whether `128` itself is a good value).
  Tracked as an open item in `priorities.md` ("Testing cpu-offloading and
  kv-value-offloading").
- `offload-backend: uva`: Unified Virtual Addressing over GH200's
  NVLink-C2C — same untested-on-real-hardware caveat as above.
- `enable-prefix-caching`: standard, generally-recommended vLLM feature.
  **Specifically tested and ruled out** as a factor in the GLM-5.2 hang —
  disabling it made no measurable difference to hang frequency/timing
  (proven, negative result, for the hang investigation only — not a general
  recommendation against the feature).

### Irrelevant to Isambard

Nothing identified yet for `vllm serve` flags specifically. Revisit once
more configs/models have been tried.
