# Feature roadmap — isambard-vllm (ivllm) v3

This document maps out
future directions derived from the architecture document (`architecture.md`)
and the Architecture Decision Records (`adr.md`).

When features are complete move to `./scope.md`

### Model performance benchmarking — `ivllm compare`

TODO: Is this still consistent with the implementation?
TODO: What is best way to E2E test this?

**ADR-118** (revised 2026-08-05 — persistent-path reuse against a shadow
project directory; supersedes the original ephemeral-job design in place)

Let an agent submit several candidate vLLM configs, get throughput/latency
numbers for each, and compare them. **Revised decision**: reuse the real,
unmodified `ivllm-serve.sh`/lockfile/monitor-triad path exactly as production
uses it — pointed at a separate `$BENCH_PROJECTDIR` whose expensive
subdirectories (`engine/vllm`, `engine/nvhpc`, `engine/rdma`, `model`) are
symlinked back to the real project directory, so nothing is re-downloaded or
recompiled, but `engine/jobs`/`engine/diagnostics` stay independent so
benchmark runs never collide with real job state. This replaces the original
"ephemeral, lockfile-free job" design — see `design/adr.md` ADR-118 for the
full rationale (numbers should describe what actually gets deployed, not a
parallel launch path that can silently drift from it).

| Step | Description | Dependencies |
|------|-------------|--------------|
| 1. | One-time admin setup: create `$BENCH_PROJECTDIR` with `engine/vllm`, `engine/nvhpc`, `engine/rdma`, `model` symlinked to the real project dir's equivalents; `engine/jobs`/`engine/diagnostics` left as real, independent directories | — |
| 2. | `run_vllm_bench()` (new, small function in `utils.sh`): activate the vLLM venv (`resolve_vllm_version_dir` + `source bin/activate` — `monitor_head`'s shell doesn't have this on `PATH` by default, unlike `run_head_vllm.sh`), then run `vllm bench serve --host localhost --port "$server_port" --dataset-name random`, save `bench.json` | — |
| 3. | Add an `IVLLM_BENCH_MODE` env-var-gated branch in `monitor_head()`, right after `update_status_running "$job"` (`utils.sh:1061` — the same point that already confirms healthy *and* warmed-up): call `run_vllm_bench()` then `request_cancel "$job"`, driving the *existing* graceful-shutdown path — no new shutdown logic needed | Step 2 |
| 4. | CLI: new `benchmarkProjectDir` config field (`ivllm config --benchmark-project-dir <path>`) so `compare` never risks reusing/clobbering the regular `projectDir` setting | — |
| 5. | CLI: `ivllm compare <comparisonName> --submit <config1.yaml> <config2.yaml> ...` — for each config, a real `Backend.requestStart()` call (`ivllm-serve.sh -b`, non-interactive batch partition — already skips the interactive reservation, no new code needed for this) with `IVLLM_BENCH_MODE=1` injected into that job's env exports; writes `comparison.json` manifest (configName → slurmJobId, submittedAt, status) | Steps 3-4 |
| 6. | CLI: `ivllm compare <comparisonName> --analyse` — one-shot status check against the manifest, rsyncs down newly-completed configs' diagnostics, prints a status table with metrics inline; no verdict-picking | Step 5 |
| 7. | Diagnostics stored at `$BENCH_PROJECTDIR/engine/diagnostics/<comparisonName>/<configName>/{vllm.yaml,slurm.sh,vllm.log,bench.json}` — reuses the existing `capture_job_diagnostics`/`tidy_up` machinery automatically, since these are ordinary jobs on the real path | — |
| 8. | Default `--time` of 2 hours (large/multi-node model load + warmup can itself take 45–60+ min), overridable per comparison run | — |

**No longer needed, removed from scope by the revision:** extracting
`IVLLM_ARGS`-building logic out of `run_head_vllm.sh`/`run_worker_vllm.sh`
into a lockfile-free shared function, and a separate ephemeral-job
multi-node coordination design — both were needed only to support a
lockfile-free launch path. Since Option 3 reuses the real, lockfile-backed
path unmodified, multi-node benchmarking works automatically with zero new
coordination code.

**Rationale:** the thing being benchmarked is now *exactly* the thing that
gets deployed (same scripts, same lockfile/monitor triad, same JIT-cache
handling, same multi-node coordination) rather than a parallel
implementation that has to be kept in sync by hand — more honest numbers,
and less new code to maintain than the original design, despite reusing
more. Still sits outside any new `Backend` lifecycle method — benchmark jobs
are ordinary `Backend.requestStart()` calls with `IVLLM_BENCH_MODE=1` set
and a different project dir, not a new state machine.

### Improved diagnostics

TODO: elaborate we have IVLLM_DEBUG and associated level.
Be useful if this was a one stop integrated flag to manage all other flags in
e.g. NCCL, libfabric, ?Torch, ?Others.

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

