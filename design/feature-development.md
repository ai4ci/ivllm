# Feature roadmap — isambard-vllm (ivllm) v3

This document maps out
future directions derived from the architecture document (`architecture.md`)
and the Architecture Decision Records (`adr.md`).

When features are complete move to `./scope.md`

### Model performance benchmarking — `ivllm bench`

**Done — moved to `design/scope.md` §1.** Implemented as
`ivllm bench submit|status|results`, backed by a standalone login-node
orchestrator (`ivllm-bench.sh`) rather than the `IVLLM_BENCH_MODE`
compute-side hook this section originally proposed — see `design/adr.md`
ADR-118's "Implementation update" for exactly what changed and why during
build, and `scope.md` for the current, implemented behaviour. E2E test
coverage: `tests/bash/sandboxed/test-ivllm-bench.sh` sources `ivllm-bench.sh`
directly (no SLURM) to unit-test `write_status_summary()`/the model-prefetch
loop against real fixtures — a genuine end-to-end run (real `sbatch`
submission, `srun --overlap` against a live job) hasn't been exercised yet
and would need real cluster time, not just the sandboxed harness.

### Improved diagnostics

**Update, 2026-09-03**: `wait_report()` no longer exists — renamed to
`monitor_node()` project-wide, and much of what this section originally
planned as future work (`IVLLM_DEBUG_LEVEL` levels 3-4, `set_debugging_env()`,
trigger-driven `report_cuda`/`report_torch`/`report_gpu`/`report_processes`
captures) is now actually implemented, not just prototyped — see
`design/active-issues.md`'s GLM-5.2 entry and CUDA-coredump-catch-22 entry for
what's actually true about it today. This section is kept for historical
context on the original design intent rather than rewritten to match.

**What existed at the time of writing** (`report_memory()`/`wait_report()`,
`utils.sh`): `IVLLM_DEBUG_LEVEL` is a numeric knob checked once per
`wait_report()` tick (every `IVLLM_CHECK_INTERVAL_SECS`, while
`initialising`, or always if `IVLLM_RUNTIME_DEBUG=1`):

| Level | Adds |
|-------|------|
| 0 (default) | Per-node cache size / RAM / top-6-process summary line |
| 1 | + per-GPU utilisation/memory (`nvidia-smi`, cheap, no process attach) |
| 2 | + `py-spy dump --nonblocking` stack traces of every vLLM/Ray worker process, appended to `debug/pyspy-node<N>.log` under the job directory (persistent — survives `clear_localdir()`, unlike `$localdir` itself) |

Levels above 2 (e.g. `IVLLM_DEBUG_LEVEL: 3`, used in some example configs —
see `examples/glm-5.2-743b-int4.yaml`) currently have **no additional
effect** — there's no `debug_level < 3` branch in `report_memory()`, so `3`
behaves identically to `2` today. Worth either documenting that explicitly
where the flag is set, or actually using `3` for something (the GLM-5.2
investigation's pyspy traces were the single most decisive diagnostic tool
across this whole project — see `design/active-issues.md` — so a genuine
level 3, e.g. NCCL/libfabric trace-level logging turned on automatically
rather than needing separate manual env vars, would have real value).

**The one-stop idea** (a single `IVLLM_DEBUG_LEVEL` that also manages
`NCCL_DEBUG`/`NCCL_DEBUG_SUBSYS`, `FI_LOG_LEVEL`/`FI_LOG_SUBSYS`, and
Triton/torch verbosity together) is not built — every job that has needed
this level of visibility so far (GLM-5.2-INT4, most recently) has set these
env vars individually and by hand in its `vllm.yaml` `env:` block (see
`knowledge-base.md`, in progress, for exactly which ones and what they
showed). Two real lessons from that investigation worth folding into any
future one-stop design: (1) libfabric's own log-level ordering is
non-monotonic on this platform — `FI_LOG_LEVEL=info` produced *more* output
than `trace`, and neither showed anything during an actual live hang, only
at connection setup — so a naive "debug=more verbose=better" mapping would
be actively misleading here; (2) at least two env vars set by hand in that
same config turned out to be silently-wrong due to typos
(`VLMM_LOGGING_LEVEL`, `NCCL_DEBUF_SUBSYS` — see `knowledge-base.md`) with
no error or warning from anything in the stack — a real, motivating example
for why a single validated flag (that fails loudly on a typo) would be
worth building, rather than free-form `env:` blocks.

### Advanced scheduling

**Architecture: "Scheduling vLLM starts with SLURM"** section

Expose SLURM native scheduling features: deferred start, job dependencies,
and job arrays for batch evaluation.

| Step | Description | Dependencies |
|------|-------------|--------------|
| 1. | `--begin <datetime>` for deferred start | — |
| 2. | `--dependency=afterok:<job-id>` for job chaining | — |
| 3. | `--script=eval.py` pattern: start vLLM, wait for health, run script | — |
| 4. | Job arrays: batch start multiple models in parallel | — |

The engine bash backend supports additional parameters to the sbatch jobs.
Could be used to implement this feature. Question is what other than scheduling
might this be used for. A small extension to `ivllm connect` for scheduling a
job in the future and disconnecting from ssh is an option.

### Model routing server

**Architecture: "Model routing server"** section

A local HTTP proxy that listens on `localhost:11434` and routes OpenAI-API
requests to multiple inference backends (multiple Isambard jobs, local Ollama,
etc.) based on model name.

| Step | Description | Dependencies |
|------|-------------|--------------|
| 1. | Build a lightweight router that reads `status.json` files from all backends | Backend abstraction |
| 2. | Auto-connect stopped models on first request (lazy startup) | `connect()` working |
| 3. | Short idle timeouts for infrequently-used models | Idle timeout (Phase 0) |
| 4. | Stateless router: restart rediscovers models from lockfiles | Lockfile protocol |

**Rationale:** Single entry point for agent harnesses. Manages port allocation,
multi-model concurrency, and lazy startup transparently.

### Prototype: Plugins and patches

On the original nemotron release there was a plugin specified for the reasoning
parser. Solar open needs a specifc vllm fork to run as it includes parsers:
https://github.com/vllm-project/vllm/compare/main...UpstageAI:vllm:v0.22.0-solar-open2
It would be useful to be able to apply these as plugins or patches to stock vllm at
specific model runtime. We had a mechanism in v2 for doing this by linking
plugins into the job directory, it might be useful to have model specific monkey
patches.

We have implemented prototypes for doing this but this requires manual
intervention. There is no existing mechanism for applying patches to a job on
a per job basis - as defined by configuration.

Concept - set up mirror vllm version directory using something like bubblewrap
on the backend. patch that copy of the backend on a per job basis. run bwrapped
version per job. Original vllm install stays untouched.

## Future Directions

The v3 architecture is designed to support these evolutions without structural
changes. Each maps to one or more ADRs.

### Multinode disggregation strategies

* Update current vllm config skill.
* Current multinode is partly data parallel friendly in that it will allow Wide EP
and calculates appropriate data-parallel configuration across multiple nodes
* on multinode we have 4 slingshot x 4 NVlink x GH200
* Need to shard experts in a way that miniminses internode traffic but in which
shared layers don't fill up whole 96Gb free of GPU. Hybrid parallelisation with
experts is necessary where shared layers are significant size - TP4 in node to shard
shared layers and DP or PP between nodes. Potentially more performant is Wide EP
where TP=1 and DP for all the nodes. Only will work for models where shared
layers are small
* https://docs.vllm.ai/en/stable/serving/data_parallel_deployment/
* https://docs.vllm.ai/en/stable/serving/expert_parallel_deployment/
* when is DP better that TP. Wide EP strategies.
* Prefill decode disggregation maybe another route for sharding this is currently
not supported.
* With the UCCL-P2P and EP we should have NIXL support:
* https://docs.vllm.ai/en/stable/features/nixl_connector_usage/

### Multi-backend support

**ADR-110** (Backend-agnostic lockfile) · **ADR-111** (Backend interface)

Add the ability to run vLLM on backends other than Isambard bare-metal —
for example, Ollama (local), a different HPC, or container-based deployment.

| Step | Description | Dependencies |
|------|-------------|--------------|
| 1. | Add `backend` and `backendConfig` fields to `LockfileV3` | — |
| 2. | Extend `Credentials` to support multiple named backends | — |
| 3. | Register additional backends in `backendRegistry` | `Backend` abstract class exists |
| 4. | CLI: `ivllm connect <job> --backend <name>` | — |
| 5. | Each backend implements its own runtime (bash framework, local exec, container) | — |

**Rationale:** Keeps the lockfile protocol shared so a future model router can
discover and dispatch to any backend.


### Container-based installation

**ADR-114** (Dual installation path)

Support pre-built Apptainer images alongside the current bare-metal path.
Users choose via an environment variable or config flag.

| Step | Description | Dependencies |
|------|-------------|--------------|
| 1. | Add `CONTAINER` env var to SLURM scripts — if set, use `singularity exec` | Bash framework |
| 2. | Test container path with existing `isambard_containers` images | — |
| 3. | Recipe database (Phase 2) can specify preferred runtime per model | — |
| 4. | Maintain both paths through testing | — |

**Trade-off:** Container path is easier to deploy (pre-built, newer CUDA 13)
but harder to debug (inside container). Bare-metal is the default.

### Maybe: Multiple models per node

**ADR-113** (Each model is independent job)

Run multiple vLLM instances on a single node (e.g. Qwen3.6 + Gemma4 sharing
4 GPUs) with independent lockfiles, ports, and idle timeouts.

| Step | Description | Dependencies |
|------|-------------|--------------|
| 1. | Add GPU affinity (`CUDA_VISIBLE_DEVICES`) to SLURM scripts | — |
| 2. | Add `resources` block to `backendConfig` in lockfile | ADR-110 |
| 3. | Monitor must be GPU-aware: shutdown of one model doesn't affect others | Monitor triad |
| 4. | Integrate with router (Phase 4) for multi-model discovery | — |

### Maybe: Model recipe database

**ADR-115**

Ship a built-in `models.yaml` with 100+ model configurations (inspired by
`model_recipes.yaml` in `isambard_containers`). Users type `ivllm connect
Qwen/Qwen3.6-35B-A3B-FP8` and the tool auto-configures vLLM args, parallelism,
and environment variables — no custom YAML needed.

| Step | Description | Dependencies |
|------|-------------|--------------|
| 1. | Design recipe schema with inheritance (`_base` configs) | — |
| 2. | Ship `models.yaml` bundled with ivllm, installed by `ivllm setup` | — |
| 3. | CLI: auto-resolve model ID to recipe when `--config` is omitted | — |
| 4. | Allow local override recipes via `~/.config/ivllm/models.yaml` | — |

**Rationale:** Eliminates the need for per-job config files for popular models;
encodes hard-won tuning in shared recipes.

