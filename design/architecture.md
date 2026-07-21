# Architecture — isambard-vllm (ivllm) v3

## Overview

The system manages vLLM inference jobs on Isambard AI HPC. A thin TypeScript CLI
on the user's local machine coordinates with a bash framework on the HPC to
start, monitor, connect to, and shut down inference jobs.

The key architectural shift from v2 is:

> **Lifecycle ownership moves from the LOCAL client to the COMPUTE node.**

In v2, the LOCAL client owned the entire lifecycle via a "session-owner" pattern:
the client created a lockfile, submitted a SLURM job, monitored it, established
a tunnel, kept it alive via heartbeat, and killed everything on disconnect. This
meant one client = one model, and no possibility of detach/reattach.

In v3, the COMPUTE node runs a self-sufficient bash framework that manages its
own lifecycle. The LOCAL client is a thin orchestrator that can come and go
freely — any project member can connect to a running instance.

---

## The three layers

```
┌──────────────────────────────────────────────────────────────────┐
│  LOCAL (TypeScript CLI)                                          │
│                                                                  │
│  ivllm connect <job>     Start or attach to a running job        │
│  ivllm cancel <job>      Request graceful shutdown               │
│  ivllm list              List all jobs with status                │
│  ivllm config            Show/set connection details              │
│  ivllm setup <version>   Install vLLM on HPC (one-off)           │
│  ivllm agent             Launch AI assistant connected to vLLM    │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ SSH/SCP layer (preserved from v2)                           │ │
│  │ • Multiplexed SSH (ControlMaster) for fast remote commands  │ │
│  │ • SCP file upload/copy                                      │ │
│  │ • SSH forward tunnel: localhost:PORT → COMPUTE:PORT         │ │
│  │ • Dry-run mode for testing without HPC access               │ │
│  └─────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
                           │
                           │ SSH
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│  LOGIN NODE (HPC login)                                          │
│                                                                  │
│  • Accepts CLI commands (sbatch, scancel, file ops)              │
│  • Model downloads via srun on interactive partition             │
│  • Status.json files visible on parallel filesystem              │
│  • No long-lived processes (scripts forward to COMPUTE)          │
└──────────────────────────────────────────────────────────────────┘
                           │
                           │ Parallel filesystem + SLURM
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│  COMPUTE NODE (HPC job allocation)                               │
│                                                                  │
│  Bash Framework ($PROJECTDIR/engine/lib/)                        │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ utils.sh      - Lockfile management, cache, monitors,       │ │
│  │                  shutdown, diagnostics                       │ │
│  ├─────────────────────────────────────────────────────────────┤ │
│  │ vllm-env.sh   - NVHPC/NCCL/Slingshot environment (tuning)   │ │
│  ├─────────────────────────────────────────────────────────────┤ │
│  │ hf.sh         - Model download (shared HF cache)            │ │
│  ├─────────────────────────────────────────────────────────────┤ │
│  │ vllm-setup.sh      - vLLM installation (versioned venvs)         │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Job Directory ($PROJECTDIR/engine/jobs/<job>/):                 │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ status.json       - Lockfile (lifecycle state machine)      │ │
│  │ jit-cache.tar.gz  - Compiled JIT kernels (shared storage)   │ │
│  │ vllm.<N>.log      - vLLM output logs (one per node)        │ │
│  │ vllm.yaml         - Model configuration                     │ │
│  │ slurm.sh          - SLURM batch script                      │ │
│  └─────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
```

---

## Lockfile Protocol

The lockfile (`status.json`) is the single source of truth for a job's state.
It lives on the parallel filesystem, visible to all nodes. All parties
(LOCAL, LOGIN, COMPUTE) communicate through it.

### States and transitions

```
  ┌──────────┐
  │ PENDING  │  ← CLI creates this when starting a job
  └────┬─────┘
       │ (SLURM allocates resources, vLLM starts)
       ▼
  ┌──────────────┐
  │ INITIALISING │  ← SLURM script writes compute hostname, PID
  └──────┬───────┘
         │ (vLLM /health responds)
         ▼
  ┌───────────┐
  │  RUNNING  │  ← vLLM is serving requests
  └─────┬─────┘
         │ (a trigger condition arises)
         ▼
  ┌───────────┐      ┌──────────┐
  │  STOPPED  │      │  FAILED  │  ← terminal states
  └───────────┘      └──────────┘
       ▲                   ▲
       │                   │
  ┌────────┐               │
  │ CANCEL │  ← user writes this to request graceful shutdown
  └────────┘               │
                           
  STOPPED = clean shutdown (user cancel, idle timeout, SLURM timeout)
  FAILED  = unclean shutdown (vLLM crash, startup failure)
  CANCEL  = request state (not terminal — the monitor reads it and acts)
```

### Lockfile schema

```json
{
  "status": "pending | initialising | running | failed | stopped | cancel",
  "jobName": "qwen36",
  "model": "Qwen/Qwen3.6-35B-A3B-FP8",
  "serverPort": 49153,
  "requestedTime": "2026-07-14T12:00:00+00:00",
  "idleTimeout": 30,
  "slurmJobId": "123456",
  "computeHostname": "nid12345",
  "startTime": "2026-07-14T12:05:00+00:00",
  "stopTime": "2026-07-14T13:05:00+00:00",
  "vllmPid": 12345,
  "reason": "idle timeout",
  "exitCode": 0
}
```

Notes:


### Lockfile lifecycle rules

| Action | Status change | Who | How |
|--------|--------------|-----|-----|
| Start job | (none) → `pending` | CLI | `jq -n '{status: "pending", ...}' > status.json` with `set -C` (atomic create) |
| SLURM allocated | `pending` → `initialising` | COMPUTE (head node) | `jq '.status = "initialising" | .slurmJobId = ... | .computeHostname = ...'` |
| vLLM healthy | `initialising` → `running` | COMPUTE (head node) | `jq '.status = "running"'` after warmup |
| User cancels | (any) → `cancel` | CLI or any user | `jq '.status = "cancel"'` (request, not terminal) |
| Clean shutdown | `cancel` → `stopped` | COMPUTE (monitor) | `tidy_up` exit trap |
| Idle timeout | `running` → `stopped` | COMPUTE (monitor_head) | Log monitoring + SIGUSR2 → `tidy_up` |
| SLURM timeout | (any) → `stopped` | COMPUTE (SIGUSR1 trap) | SLURM sends SIGUSR1 → `tidy_up` |
| vLLM crash | `initialising`/`running` → `failed` | COMPUTE (exit trap) | `tidy_up` detects non-zero exit |
| Force cancel | (any) → `stopped` | CLI | `scancel <id>` + `jq '.status = "stopped"'` |

---

## The Monitor Triad

The bash framework runs three monitoring processes on the compute allocation:

### 1. `monitor_startup` (foreground, head node only)

Blocks until vLLM becomes healthy. Then:

1. Polls `/health` every 10s
2. Once healthy: saves JIT cache
3. Sends a warmup request to `/v1/chat/completions` (triggers JIT compilation)
4. Saves cache again after warmup
5. Transitions status to `running`
6. Detaches (returns)

### 2. `monitor_head` (background, head node only)

Runs for the entire job lifetime. Checks every 10s:

1. Lockfile exists? If deleted → panic shutdown
2. Lockfile status is `cancel`? → clean shutdown with reason "user cancel"
3. vLLM process alive? If dead → shutdown with reason "lost contact with vllm process"
4. **Idle timeout** (backend-specific — see ADR-105):
   - For the Isambard backend: incrementally scan the vLLM access log for
     "real" API requests (not `/health` probes). If none within the idle
     window → shutdown with reason "idle timeout".
   - Other backends implement their own idle detection.
5. Worker threads check head status: `monitor_worker` reads lockfile; if not running → shutdown

### 3. `monitor_worker` (background, worker nodes)

For multi-node jobs, each worker node runs this:

1. Polls lockfile via parallel filesystem
2. If lockfile disappears or status is not `pending`/`initialising`/`running` → SIGTERM local vLLM process
3. Reports memory/JIT cache usage during startup

---

## Withdrawal / Connect Cycle

The CLI can safely disconnect and reconnect at any point:

```
Disconnect:
  Ctrl+C or terminal close
  → tunnel dies
  → CLI exits
  → Job keeps running (COMPUTE monitors keep it alive)
  → Idle timeout will eventually shut it down if unused

Reconnect:
  ivllm connect <job>
  → Reads status.json
  → If "running": establish SSH tunnel immediately
  → If "stopped" or "failed": restart the job
  → Monitor lockfile, tail logs during startup
```

---

## What we preserve from v2

| Component | v2 file | Status |
|-----------|---------|--------|
| CLI entry & command routing | `src/index.ts` | Preserved — add new commands |
| Config management | `src/config.ts` | Preserved as-is |
| YAML config parsing | `src/vllm-config.ts` | Preserved — add `idleTimeout` field, metadata block, and `targetVllmVersion` replacing `minVllmVersion` |
| SSH/SCP operations | `src/remote-ops.ts` | Preserved — refactor interface slightly |
| Local port/health ops | `src/local-ops.ts` | Preserved |
| Version matching | `src/semver.ts` | Preserved |
| Assistant launcher | `src/assistant.ts`, `src/commands/agent.ts` | Preserved |
| NVHPC/NCCL tuning env vars | `src/templates/inference.ts` → `renderNVHPCPreamble()` | Move to `vllm-env.sh` |
| JIT cache management | `src/templates/inference.ts` → `renderWorkDirSetup()` | Move to `utils.sh` |
| Dry-run mock infrastructure | `src/remote-ops.ts` | Preserved |
| Test framework | `tests/*.ts` | Preserved — add tests for new commands |

| Component | v2 file | Status |
|-----------|---------|--------|
| Session lifecycle pipeline | `src/session-helper.ts` | **Replace** — logic moves to bash |
| Monitoring loop | `src/monitors.ts` | **Replace** — monitoring moves to bash `monitor_*` |
| Bash templates | `src/templates/inference.ts` (1,176 lines) | **Replace** — extract to `lib/*.sh` |
| `ivllm start` / `ivllm interactive` | `src/commands/start.ts`, `interactive.ts` | **Replace** with `ivllm connect` |
| `ivllm stop` | `src/commands/stop.ts` | **Replace** with `ivllm cancel` |
| Lockfile schema | `job_details.json` | **Replace** with `status.json` |
| Old design docs | `design/` → `design/old/` | **Archived** |

---

## Key design properties

1. **No single point of failure**: If LOCAL disconnects, the compute node keeps running. If a COMPUTE monitor dies, the exit trap fires and cleans up.

2. **Idempotent operations**: `ivllm connect` can be run repeatedly. If the job is already running, it just attaches the tunnel. If stopped, it restarts.

3. **Multi-user safe**: Lockfiles and caches use `umask 0002` and `chmod g+w`. Any project member can connect to any running job.

4. **Graceful degradation**: No heartbeat needed from LOCAL. The compute node `monitor_head` is the authority on whether to keep running.

5. **Testable components**: The bash framework has a standalone test harness (`test-vllm.sh`) that mocks vLLM and SLURM. The CLI has `--dry-run` and `--mock` modes.

---

## Future Directions

The v3 architecture is designed to accommodate several planned evolutions
without structural changes. These are not commitments — they inform design
choices about abstraction boundaries and data formats.

### 1. Multi-backend support

The current architecture is Isambard-specific (SLURM, NVHPC, Slingshot 11),
but the lockfile protocol and CLI interface are backend-agnostic by design.

To add a new backend (e.g. local Ollama, a different HPC, containers):

- **New backend module** in `src/backends/<name>/` implementing a standard
  interface:
  ```typescript
  interface Backend {
    name: string;
    connect(job: JobConfig): Promise<ConnectResult>;
    cancel(jobName: string): Promise<void>;
    status(jobName: string): Promise<LockfileV3 | null>;
    list(): Promise<LockfileV3[]>;
    setup?(version?: string): Promise<void>;
  }
  ```
- **Each backend owns its runtime** — the compute-side management is
  entirely the backend's responsibility. The Isambard backend uses bash
  framework; an Ollama backend would just `exec ollama run` locally.
- **Lockfile protocol is shared** — `status.json` format is backend-agnostic.
  Backend-specific metadata goes in a `backend` namespace:
  ```json
  {
    "status": "running",
    "jobName": "qwen36",
    "model": "Qwen/Qwen3.6-35B-A3B-FP8",
    "backend": "isambard-vllm",
    "backendConfig": {
      "slurmJobId": "123456",
      "computeHostname": "nid12345",
      "gpuCount": 4,
      "vllmVersion": "0.22.0"
    }
  }
  ```
- **Multiple SSH connections** are already supported at the SSH layer
  (`remote-ops.ts` takes a `Credentials` object per connection). Future
  CLIs would configure multiple backends with different credentials.
- **The CLI default (`ivllm connect`) targets a default backend**;
  `ivllm connect <job> --backend <name>` selects an alternative.

### 2. Model routing server

A local HTTP server that acts as an OpenAI API-compatible proxy, managing
multiple inference backends simultaneously.

```
Agent → http://localhost:11434/v1  →  Router  →  isambard: model A
                    │                        ├─ isambard: model B
                    │                        ├─ ollama: model C (local)
                    │                        └─ ...
```

**How it builds on v3**:

| v3 feature | Router use |
|-----------|-----------|
| Lockfile protocol | Router reads `status.json` to know which models are running |
| `ivllm connect` | Router calls `backend.connect()` internally |
| Idle timeout | Router sets short idle timeout for infrequently-used models |
| `backend` namespace in lockfile | Router dispatches requests to the correct backend type |
| Port pool (`generateRandomHighPort`) | Router manages a fixed port pool (e.g. 11435–11534) |

**Lazy startup flow**:
1. Agent sends `POST /v1/chat/completions` with model `Qwen/Qwen3.6-35B-A3B`
2. Router checks model registry → model is `stopped`
3. Router calls `backend.connect('qwen36')` → lockfile transitions
   `stopped → pending → initialising → running`
4. Router waits for `running`, establishes tunnel, proxies request
5. Subsequent requests go straight through (no startup delay)
6. After idle timeout, model shuts down, lockfile goes back to `stopped`

**Router state is stateless** — it reads all state from lockfiles. If the
router process dies, restarting it rediscovers all running models from
`$PROJECTDIR/engine/jobs/`.

### 3. Multiple models on a single node

Running e.g. Qwen3.6 and Gemma4 simultaneously on one Isambard node:

- **Each model is an independent job** with its own `status.json`, port,
  vLLM process, and GPU allocation
- **GPU partitioning** via vLLM's `--num-gpu-blocks` or `CUDA_VISIBLE_DEVICES`
  to assign specific GPUs per model instance
- **The prototype already supports this**: per-job directories under
  `$PROJECTDIR/engine/jobs/`, independent lockfiles, independent monitors
- **The monitor triad runs per-model**: each vLLM instance gets its own
  monitor_head, monitor_worker (for multi-node only)
- **Resource constraints** encoded in the lockfile:
  ```json
  {
    "status": "running",
    "jobName": "gemma4",
    "model": "google/gemma-4-4b-it",
    "resources": {
      "gpus": [0, 1],        // specific GPU indices
      "memoryGb": 40,         // per-GPU memory limit
      "cpuCores": 16
    }
  }
  ```

### 4. Interactive reservation clarification

The v2 codebase maintained a distinction between `ivllm start` (sbatch)
and `ivllm interactive` (srun), based on the assumption that the Isambard
interactive reservation only accepted `srun` jobs. This is incorrect —
**the interactive reservation accepts both `sbatch` and `srun`**.

[Isambard interactive reservation](https://docs.isambard.ac.uk/user-documentation/information/job-scheduling/#interactive-reservation-isambard-ai-phase-2):
- Dedicated pool of nodes for interactive work
- 8-hour limit (vs standard partitions)
- 50% premium billing (1.5 NHR per NHR used)
- Per-user limits (typically 1 node)

**Implication for `ivllm connect`**: The `--interactive` flag is still
useful as a shorthand for `--partition=interactive --reservation=interactive`,
but it should use `sbatch` by default, not `srun`. The `srun` TTY-binding
mode is deprecated. Tailing the log file over ssh is the only way to follow the
progress of the .
The default behaviour (`sbatch` to interactive partition) is cleaner.

```bash
# These are now equivalent:
ivllm connect qwen2 --partition=interactive --reservation=interactive
ivllm connect qwen2 --interactive  # shorthand for the above, uses sbatch
```

### 5. Scheduling vLLM starts with SLURM

SLURM has native support for deferred and dependent job submission that
`ivllm connect` could expose:

**Deferred start** (`--begin`):
```bash
# Start vLLM at a specific time (e.g. overnight)
ivllm connect qwen2 --begin "2026-07-16T02:00:00"
# Submits: sbatch --begin=2026-07-16T02:00:00 slurm.sh
```

**Job dependencies** (`--dependency`):
```bash
# Start model B after model A is healthy
ivllm connect model-a
ivllm connect model-b --dependency=afterok:<job-id-of-a>
```

**Scheduled evaluation pipeline**:
```bash
# 1. Download/evaluate data overnight
sbatch prepare-data.sh
# 2. Start model after data is ready
ivllm connect qwen36 --dependency=afterok:$(sbatch_prepare_data)
# 3. Run evaluation script against model (same job, after vLLM is healthy)
ivllm connect qwen36 --script=eval.py --dependency=afterok:$(sbatch_prepare_data)
```

The last form uses the `submit_job` pattern from `isambard_containers`:
start vLLM, wait for health, then run a script against it in the same job.
This is more efficient than separate jobs because it avoids the queue wait
between model startup and script execution.

**Job arrays for batch evaluation**:
```bash
# Evaluate multiple models/configs in parallel
for model in qwen3.6 gemma4 llama4; do
  ivllm connect $model --begin "+30 minutes" --time-limit "4:00:00"
done
```

### Interaction between the three directions

```
                    ┌────────────────────────┐
                    │    Model Router         │
                    │  (local HTTP proxy)     │
                    └────┬───────────────────┘
                         │
         ┌───────────────┼───────────────┐
         │               │               │
         ▼               ▼               ▼
  ┌────────────┐  ┌────────────┐  ┌────────────┐
  │ Backend A  │  │ Backend B  │  │ Backend C  │
  │ Isambard   │  │ Other HPC  │  │ Ollama     │
  │ vLLM       │  │ Container  │  │ (local)    │
  └────────────┘  └────────────┘  └────────────┘
         │
    ┌────┴────┐
    │ Node 1  │  ← Multiple models per node
    │ Qwen3.6 │
    │ Gemma4  │
    └─────────┘
```

The **lockfile protocol** is the unifying abstraction: every backend writes
`status.json`, every client reads it. The router dispatches based on the
`backend` field. Multiple models on one node are just multiple `status.json`
files in the same `jobs/` directory.

---

## Cross-Project Learnings

Review of [UKGovernmentBEIS/isambard_containers](https://github.com/UKGovernmentBEIS/isambard_containers)
revealed several patterns that inform our design.

### 1. Container-based vLLM deployment

The `isambard_containers` project ships pre-built Apptainer images with vLLM
(and PyTorch, TRL) for Isambard GH200. Containers include CUDA 13.0.2,
Slingshot/NCCL networking configured, and the `brics/apptainer-multi-node`
module for multi-node support.

**Implication for our design**: The v2 bare-metal pip install approach
(ADR-011/013) works but requires ongoing maintenance for each vLLM version.
An Apptainer-based installation path (as described in the now-on-hold ADR-010)
would be simpler to maintain and could reuse the existing build infrastructure.
The `isambard_containers` project already maintains vLLM containers we could
consume directly. Our Phase M1 bash framework's `vllm-setup.sh` could optionally
pull a pre-built container instead of running `pip install`.

#### Build comparison: our bare-metal vs their container

**Key insight: our approaches are equivalent in what they produce, but
different in how they package it.**

Both approaches ultimately run `vllm serve` with CUDA 12+ and PyTorch.
The differences are in the build and deployment chain:

| Layer | Our bare-metal | Their container |
|-------|---------------|----------------|
| Base OS | HPC host (RHEL) | Ubuntu 24.04 (container) |
| CUDA | NVHPC SDK 26.3 → CUDA 12.9 compat libs | `nvidia/cuda:13.0.2-cudnn-devel` base image |
| PyTorch | `pip install` (cu129 wheel) | `pip install` inside container (cu130 wheel) |
| vLLM | `pip install` (pre-compiled wheel) | **Build from source** (aarch64 has no pre-compiled wheels) |
| NCCL | `module load brics/nccl` (host module) | `libnccl-dev` apt package + `brics/apptainer-multi-node` |
| FlashInfer | `pip install` (wheel, cached) | `pip install` from flashinfer.ai (pre-compiled cubins) |
| aws-ofi-nccl | Host module provides it | Built inside container |
| DeepGEMM | `pip install` | Built from source via `tools/install_deepgemm.sh` |
| DeepEP | Not installed | Built from source (requires NVSHMEM from NVHPC) |
| JIT caches | On `$LOCALDIR` tmpfs, tar.gz to shared storage | On `/tmp` inside container (ephemeral) |
| Multi-node | Ray bootstrap (currently) | `--distributed-executor-backend mp` |

**Fundamental differences**:

1. **vLLM build from source**: Their container builds vLLM from source
   because there are no pre-compiled aarch64 wheels (`vllm-project/vllm#23350`).
   Our pip install downloads pre-compiled wheels from `wheels.vllm.ai`.
   Building from source is slower (~30 min vs ~2 min) but allows patches.

2. **NVHPC vs container CUDA**: We use NVHPC SDK for CUDA forward compat
   (driver 565 can run CUDA 12.9). Their container ships native CUDA 13.0.2
   — the container doesn't need forward compat because the CUDA libs are
   bundled. NVHPC *could* be configured for CUDA 13, but the vLLM wheels
   default to cu129. Their approach is simpler: one container, one CUDA.

3. **NCCL source**: We load `brics/nccl` as a host module. Their container
   installs `libnccl-dev` and `aws-ofi-nccl` from source, plus uses
   `brics/apptainer-multi-node` for host NCCL bindings. The result is
   functionally equivalent.

4. **DeepEP**: Their container includes DeepEP (DeepSeek expert parallelism
   kernels) which requires NVSHMEM from the NVHPC SDK. We don't install this.
   This is a genuine advantage of the container path for DeepSeek models.

**Verdict**: No fundamentally different components. Both approaches produce
functionally equivalent vLLM runtimes. The container path:
- Is **harder to build** (source compile, more deps, Apptainer def file complexity)
- But **easier to deploy** (single `.sif` file, no venv, no module dependencies)
- Offers **newer CUDA** (13.0.2 vs 12.9 compat)
- Includes **DeepEP** for DeepSeek models

See ADR-114 (dual installation path) and ADR-115 (model recipe database).

### 2. Multiprocessing (MP) vs Ray for multi-node

The `isambard_containers` project uses `--distributed-executor-backend mp`
for multi-node vLLM — the vLLM-native multiprocessing backend. This is
what `--nnodes` defaults to when no backend is explicitly specified.

Our own codebase already has **both approaches**:

| Template | Backend | Notes |
|----------|---------|-------|
| Interactive multi-node | MP (default) | Sets `--nnodes`, `--node-rank`, `--master-addr` only. Dead Ray env vars present but unused. |
| Batch multi-node | Ray (explicit) | Uses `--distributed-executor-backend ray` with full Ray startup/teardown. |

Both approaches have trade-offs:

| Concern | MP | Ray |
|---------|----|-----|
| Complexity | Simple (no extra service) | Complex (GCS, object store, ray start/stop) |
| Failure handling | Process dies → job fails | Ray can restart workers, detect failures |
| Startup time | Faster (no Ray bootstrap) | Slower (Ray cluster init) |
| Slingshot performance | Unknown — NCCL is same either way | Proven with `brics/nccl` |
| Log noise | Minimal | Ray creates 900+ log files per node |
| Node churn | Harder (no node discovery) | Ray handles node add/remove |

**Performance on Slingshot**: Both MP and Ray use NCCL for GPU communication
under the hood. Ray adds an extra layer for process management but does not
change the NCCL data path. However, MP may have different timing, connection
setup, or collective algorithm selection that interacts differently with
Slingshot's CXI fabric. This is an empirical question that needs benchmarking.

**Implication for our design**: Our batch template currently uses Ray while
the interactive template (which is used more often) uses MP. We should:
1. Clean up the interactive template (remove dead Ray env vars)
2. Keep both paths — MP for interactive/simple multi-node, Ray for
   production batch jobs where failure handling matters
3. Benchmark MP vs Ray on Slingshot before making a final decision
4. The `isambard_containers` project uses MP exclusively and has validated
   it for multi-node — this is an option we can adopt

### 3. Model recipe database

The `isambard_containers` project has a YAML recipe file
(`model_recipes.yaml`) with 100+ model configurations. Each recipe specifies
vLLM args, parallelism settings, env vars, and even container version
requirements. Recipes inherit from base configs (e.g. `deepseek-ai/DeepSeek-R1`
extends `_deepseek_base` which sets `enable_expert_parallel: true`).

**Implication for our design**: Our per-job `vllm.yaml` approach requires
the user to figure out correct parallelism settings themselves. A shared
recipe database like `model_recipes.yaml` would let `ivllm connect`
auto-configure any supported model. Users would only need to type:
`ivllm connect deepseek-ai/DeepSeek-R1` without any config file.

### 4. Static SLURM template with --export

Their SLURM template (`serve_vllm_mp.slurm`) is a static file, not a
generated template. All dynamic values are passed via `sbatch --export`
as environment variables. The Python CLI builds the `--export` string from
a dict of key-value pairs.

**Implication for our design**: This aligns with our Phase M3 goal of
replacing the generated TypeScript templates with static files that source
shared libs. Passing job parameters via `--export` instead of inline
substitution is cleaner and matches typical SLURM patterns.

### 5. Clean worker coordination pattern

The SLURM template coordinates head and worker nodes using marker files
in a shared tmp directory:
- Head writes `$JOB_TMP/markers/done` with exit code when user script completes
- Workers poll for this file and shut down gracefully
- Workers write `$JOB_TMP/markers/worker_done_<rank>` to confirm TCPStore
disconnection before head exits

This avoids race conditions in PyTorch distributed teardown and is simpler
than our current Ray-based approach.

### 6. Debug mode with nvidia-smi monitoring

`DEBUG_MODE=1` enables NCCL tracing, per-node GPU monitoring (nvidia-smi
polling every 10s), and flight-recorder dumps for NCCL timeouts. This is
valuable for diagnosing multi-node issues.

**Implication for our design**: Our bash framework's `monitor_head` and
`renderMonitor()` should include a similar debug mode. The prototype already
has `report_memory()` but could be extended to include GPU metrics.

### 7. Gimlet secure tunneling

The project provides `gimlet-agent` for secure HTTPS tunnels (OAuth2
+ mutual TLS). While we don't need this level of security for our use case,
it demonstrates the pattern of **tunneling as a layer on top of the basic
SSH tunnel**, not part of the core lifecycle.

### What we could adopt directly

| Pattern | From | Benefit |
|---------|------|--------|
| DeepEP installation | `vllm.def` | Support for DeepEP |
| MP backend (not Ray) | `serve_vllm_mp.slurm` | Simpler multi-node, fewer bugs |
| Model recipe YAML | `model_recipes.yaml` | Auto-config for 100+ models |
| Static SLURM templates | `serve_vllm_mp.slurm` | Cleaner than generated templates |
| `--export` for job params | `serve.py` | Standard SLURM pattern |
| Worker coordination markers | `serve_vllm_mp.slurm` | Clean multi-node teardown |
| Pre-built containers | `vllm.def` + sifter registry | Avoids pip install maintenance |
| Debug mode | `serve_vllm_mp.slurm` | Easier multi-node diagnostics |

**Full source**: [github.com/UKGovernmentBEIS/isambard_containers](https://github.com/UKGovernmentBEIS/isambard_containers)
