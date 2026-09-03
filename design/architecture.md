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

The orchestration is expected to be handled from client side but login node
bash framework is sufficient to perform most model related from login node.

The `jobName` is unique. No 2 jobs with the same name can be running at the
same time. Jobs are shared between users within the same project, up to the
limits of slurm permissions.

---

## The three layers

```
┌──────────────────────────────────────────────────────────────────┐
│  LOCAL (TypeScript CLI)                                          │
│                                                                  │
│  ivllm connect <job>     Start or attach to a running job        │
│  ivllm cancel <job>      Request graceful shutdown               │
│  ivllm status [job]      List all jobs with status (or single job)               │
│  ivllm config            Show/set connection details             │
│  ivllm setup <version>   Install vLLM on HPC (one-off)           │
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
│  Bash Framework ($PROJECTDIR/engine/ivllm-*.sh                   │
│  • Handles lifecycle of running jobs and sbatch submission       │
│  • Wrapper scripts: ivllm-{serve,status,setup,cancel,...}.sh       │
│  • Model downloads (ivllm-get-model.sh) via srun, login-node-    │
│    side — not a compute-node lib/ script                        │
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
│  │                  shutdown, diagnostics, file locations      │ │
│  ├─────────────────────────────────────────────────────────────┤ │
│  │ common-env.sh   - NVHPC/NCCL/Slingshot environment (tuning) │ │
│  │ vllm-env.sh     - vllm specific variables                   │ │
│  ├─────────────────────────────────────────────────────────────┤ │
│  │ slurm-vllm-setup.sh   - Setup vllm version including plugins| │
│  │ slurm-vllm-serve.sh   - Run a model, mp executor path        │ │
│  │ slurm-ray-vllm-serve.sh - Run a model, ray executor path     │ │
│  │   (+ ray-setup.sh, ray-run-vllm.sh, run-head-vllm.sh,        │ │
│  │      run-worker-vllm.sh — per-node launch scripts)           │ │
│  │ cuda-postprocess.sh   - Decodes a piped GPU coredump on-node │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Job Directory ($PROJECTDIR/engine/jobs/<job>/):                 │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ status.json       - Lockfile (lifecycle state machine)      │ │
│  │ jit-cache.tar.gz  - Compiled JIT kernels (shared storage)   │ │
│  │ vllm.<N>.log      - vLLM output logs (one per node)         │ │
│  │ vllm.yaml         - Model configuration inc. metadata       │ │
│  │ vllm.yaml.clean   - Vllm serve cli specific options         │ │
│  │ slurm.sh          - SLURM batch script                      │ │
│  └─────────────────────────────────────────────────────────────┘ │
│  Cache Directory ($HOME/.cache/ivllm/<job>/):                    │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │ jit-cache.tar.gz  - Compiled JIT kernels (user specific)    │ │
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
  ┌────────┐          ┌─────────┐
  │ CANCEL │          │  ABORT  │
  └────────┘          └─────────┘
  (either request state can be written from any non-terminal status)

  STOPPED = clean shutdown (user cancel, idle timeout, SLURM timeout)
  FAILED  = unclean shutdown (vLLM crash, startup failure, user abort)
  CANCEL  = request state (not terminal — the monitor reads it, exit code 201 → STOPPED)
  ABORT   = request state (not terminal — the monitor reads it, exit code 254 → FAILED,
            capture_job_diagnostics() runs unconditionally — see below)
```

`ABORT` exists for debugging hangs where the job would otherwise sit there
consuming an allocation indefinitely (see `design/active-issues.md`'s GLM-5.2
investigation) — it's mechanically identical to `CANCEL` (a status string
written to the lockfile, detected by `monitor_head`/`ivllm-cancel.sh`) except
for two things: it always resolves to `FAILED`/`"user abort"` rather than
`STOPPED`/`"user cancel"`, and it unconditionally runs
`capture_job_diagnostics()` (`tidy_up()`'s exit code 254 vs 201 — `utils.sh`)
so a deliberately-killed hung job still gets its logs/pyspy dumps archived,
which a normal cancel does not bother with. `ivllm cancel --force --abort`
takes the same distinction down the force-cancel path.

### Lockfile schema

```json
{
  "status": "pending | initialising | warmup | running | failed | stopped | cancel | abort",
  "jobName": "qwen36",
  "model": "Qwen/Qwen3.6-35B-A3B-FP8",
  "user": "testuser",
  "serverPort": 49153,
  "requestedTime": "2026-07-14T12:00:00+00:00",
  "idleTimeout": 30,
  "slurmJobId": "123456",
  "computeHostname": "nid12345",
  "startTime": "2026-07-14T12:05:00+00:00",
  "stopTime": "2026-07-14T13:05:00+00:00",
  "reason": "idle timeout",
  "exitCode": 0
}
```

Notes:
serverPort, computeHostname, startTime, stopTime, slurmJobId, reason
and exitCode are responsibility of the server to populate. There is no
per-vLLM-process PID in the lockfile (a `vllmPid` field existed briefly and
was dropped) — process lifecycle is owned by a single orchestrator process
per job on the SLURM step host, which tracks every node's `srun` client PID
directly rather than publishing one through the lockfile.

### Lockfile lifecycle rules

| Action | Status change | Who | How |
|--------|--------------|-----|-----|
| Start job | (none) → `pending` | CLI | `jq -n '{status: "pending", ...}' > status.json` with `set -C` (atomic create) |
| SLURM allocated | `pending` → `initialising` | COMPUTE (head node) | `jq '.status = "initialising" | .slurmJobId = ... | .computeHostname = ...'` |
| vLLM healthy | `initialising` → `warmup` (2026-09-03) | COMPUTE (head node) | `/health` responds; `jq '.status = "warmup"'`, then warmup requests are sent |
| Warmup completes | `warmup` → `running` | COMPUTE (head node) | `jq '.status = "running"'` after warmup requests succeed |
| User cancels | (any) → `cancel` | CLI or any user | `jq '.status = "cancel"'` (request, not terminal) |
| User aborts | (any) → `abort` | CLI (`ivllm cancel --abort`) | `jq '.status = "abort"'` (request, not terminal) |
| Clean shutdown | `cancel` → `stopped` | COMPUTE (monitor) | `tidy_up "$job" 201` exit trap, reason `"user cancel"` |
| Unclean shutdown | `abort` → `failed` | COMPUTE (monitor) | `tidy_up "$job" 254` exit trap, reason `"user abort"`, `capture_job_diagnostics()` runs |
| Idle timeout | `running` → `stopped` | COMPUTE (monitor_head) | Log monitoring + SIGUSR2 → `tidy_up` |
| SLURM timeout | (any) → `stopped` | COMPUTE (SIGUSR1 trap) | SLURM sends SIGUSR1 → `tidy_up` |
| vLLM crash | `initialising`/`running` → `failed` | COMPUTE (exit trap) | `tidy_up` detects non-zero exit |
| Force cancel | (any) → `stopped`/`failed` | CLI (`ivllm cancel --force[--abort]`) | `scancel <id>` + `tidy_up "$job" 201` or `254` depending on `--abort` |

---

## Process Orchestration and Monitoring

Process orchestration is centralized: a single **orchestrator process** runs
on the SLURM step host (the top-level `slurm-vllm-serve.sh` batch script),
and `srun`-launches vLLM on the head node plus each worker node (for
multi-node jobs) as background job steps of that one script. The
orchestrator holds every node's local `srun` client PID directly — there is
no separate per-worker monitor process, and no PID published through the
lockfile. Two monitors run alongside the orchestrator, both on the step
host:

> **Update, 2026-09-03**: `monitor_startup` no longer exists as a separate
> function — it was merged into `monitor_head` (confirmed via
> `tests/bash/sandboxed/test-monitor-head.sh`'s own comment: "`monitor_startup()`
> was merged into `monitor_head()`"). The step-by-step breakdown below (health
> poll → JIT save → warmup → JIT save → transition to `running`) is still
> materially accurate as a description of what happens, and now happens
> inside `monitor_head` itself (via `run_vllm_warmup()`) rather than as a
> separate detached process — but there is no longer a `monitor_startup`
> function to point to in the source. `monitor_head` also now transitions the
> lockfile through an intermediate `warmup` status (between `initialising`
> and `running`) that this document doesn't mention anywhere yet — see
> `design/backend-contract.md`.

### `monitor_head` (background, step host)

Runs for the entire job lifetime. Checks every 10s:

1. Lockfile exists? If deleted → panic shutdown
2. Lockfile status is `cancel`? → clean shutdown with reason "user cancel"
3. Orchestrator process alive? If it has exited (e.g. vLLM crashed, taking
   down its `srun` step and, transitively, the orchestrator's `wait`) →
   stop monitoring
4. **Idle timeout** (backend-specific — see ADR-105):
   - For the Isambard backend: **update, 2026-09-03** — no longer a log
     scan. `monitor_head` now maintains a heartbeat file, touched whenever a
     real API request is seen, and compares its mtime against the current
     time (`utils.sh`, "Running — check idle timeout using heartbeat file"):
     `idle_seconds = now - heartbeat_mtime`; shuts down once
     `idle_seconds >= idle_timeout * 60`. `idle_timeout` is in minutes
     (confirmed against `design/backend-contract.md:92`). This is a real
     behavior change from the old log-grep heuristic, not just a rewrite —
     the old approach could trigger near-instantly against an empty/rotated
     log regardless of the configured timeout; the new one requires the full
     real elapsed time every time, which is more correct but means any test
     or tooling assuming near-instant idle-shutdown on a short configured
     timeout needs updating (see `active-issues.md`'s 2026-09-03 test-suite
     findings).
   - Other backends implement their own idle detection.

On any shutdown condition, `monitor_head` signals the orchestrator
(`SIGUSR2`), which runs the exit-trap logic (`tidy_up`) that kills every
tracked `srun` PID — head and workers alike — and updates the lockfile.
**Update, 2026-09-03**: this used to be accurate but no longer is —
`run_worker_vllm.sh` now calls `monitor_node()` (`utils.sh`), the same
trigger-watching monitor `run_head_vllm.sh` uses (`wait_report()`, the
passive memory-reporter this paragraph originally described, was fully
replaced project-wide). Worker nodes now watch the same diagnostics-trigger
pipe files as the head node and independently fire `report_memory`/
`report_gpu`/`report_processes`/`report_cuda`/`report_torch` captures when
one is written — they're not just idle memory-reporters waiting to be killed
any more. Shutdown is still initiated centrally by the orchestrator, which
still reaches workers by killing their local `srun` client PID (killing the
client tears down the corresponding remote step) — that part is unchanged.

The orchestrator itself is a background subshell within `slurm-vllm-serve.sh`
(the top-level SLURM batch script); the exit-trap logic (`tidy_up`) and its
signal handlers are registered on that subshell, not the top-level script.
SLURM's own pre-timeout warning (`--signal=B:SIGUSR1@120`, sent 120s before
the job's time limit) is delivered only to the top-level batch script
process — which has its own small trap whose only job is to forward that
signal into the orchestrator subshell, so the same `tidy_up` exit-trap logic
handles it, recording `"SLURM timeout"` and shutting down cleanly before the
hard kill.

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
  interface. The current `Backend` abstract class defines:
  ```typescript
  abstract class Backend {
    abstract bootstrap(): Promise<void>;
    abstract setup(version: string, force?: boolean): Promise<void>;
    abstract connect(job: string, localPort: number): Promise<CloseableEventEmitter>;
    abstract requestCancel(job: string, force: boolean, abort: boolean): Promise<void>;
    abstract requestStart(job: string, maxTime: string, batch: boolean, config?: string): Promise<void>;
    abstract getAllJobStatus(): Promise<LockfileV3[]>;
    abstract watchLog(job: string, node?: string, start?: boolean): Promise<CloseableEventEmitter>;
    abstract fetchDiagnostics(job: string, localDest?: string): Promise<string>;
    getJobStatus(job: string): Promise<LockfileV3>;
    isCancelling(job: string): Promise<boolean>;
    isRunning(job: string): Promise<boolean>;
    isStopped(job: string): Promise<boolean>;
    isStartable(job: string): Promise<boolean>;
    isStarting(job: string): Promise<boolean>;
    // Non-abstract, default-throws — only implemented by backends that
    // enable benchmarking (see `ivllm bench`, scope.md):
    requestBenchmark(comparison: string, configs: string[], time?: string): Promise<void>;
    getBenchmarkStatus(comparison: string): Promise<BenchmarkStatus>;
    fetchBenchmarkResults(comparison: string, localDest: string): Promise<
      | { ready: true; path: string }
      | { ready: false; status: BenchmarkStatus }
    >;
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
  (`src/ops/SshRemoteOps.ts` takes a `Credentials` object per connection). Future
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

Running e.g. Qwen3.6 and Gemma4 simultaneously on one Isambard node. Cheaper
to run on a single node:

- **Each model is an independent job** with its own `status.json`, port,
  vLLM process, and GPU allocation
- **GPU partitioning** via vLLM's `--num-gpu-blocks` or `CUDA_VISIBLE_DEVICES`
  to assign specific GPUs per model instance
- **The prototype already supports this**: per-job directories under
  `$PROJECTDIR/engine/jobs/`, independent lockfiles, independent monitors
- **Monitoring runs per-model**: each vLLM instance gets its own orchestrator
  process and `monitor_head`/`monitor_startup` pair
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
- Per-user limits - only 1 sbatch job allowed

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
   kernels) which requires NVSHMEM from the NVHPC SDK. However there is no
   evidence that this is actually functional in the containers as it requires
   Infinband networking rather that the Slingshot libfabric that is actually
   available on Isambard. We instead have installed UCCL-EP which is functionally
   equivalent to DeepEP and has first class support for libfabric.

**Verdict**: No fundamentally different components. Both approaches produce
functionally equivalent vLLM runtimes. The container path:
- Is **harder to build** (source compile, more deps, Apptainer def file complexity)
- But **easier to deploy** (single `.sif` file, no venv, no module dependencies)
- Offers **newer CUDA** (13.0.2 vs 12.9 compat)
- Includes **DeepEP** for DeepSeek models which may not work on slingshot

See ADR-114 (dual installation path) and ADR-115 (model recipe database).

### 2. Multiprocessing (MP) vs Ray for multi-node

**Status: both are implemented and actively used, selected per-job via
`distributed-backend-executor: ray` in `vllm.yaml`** (default when unset is
`mp`) — `ivllm-serve.sh` reads this key and dispatches to
`slurm-vllm-serve.sh` (mp) or `slurm-ray-vllm-serve.sh` (ray) accordingly.
This is no longer a design question, but the empirical picture from real
multi-node debugging (Nemotron-3-Ultra, GLM-5.2-INT4 — see
`design/active-issues.md`) is worth recording:

| Concern | MP | Ray |
|---------|----|-----|
| Complexity | Simple (no extra service) | Complex (GCS, object store, ray start/stop) |
| Failure handling | Process dies → job fails | Ray can restart workers, detect failures |
| Startup time | Faster (no Ray bootstrap) | Slower (Ray cluster init) |
| Log noise | Minimal | Ray creates 900+ log files per node |
| Node churn | Harder (no node discovery) | Ray handles node add/remove |
| GH200/Slingshot hangs | Confirmed reproducible (see below) | Confirmed reproducible (same signature) |
| `numa-bind` support | Works — `vllm/utils/numa_utils.py`'s `configure_subprocess()` wraps worker launches in `numactl` | **Does not work** — `RayExecutorV2` builds its worker actors independently of `MultiprocExecutor`'s init path, which is the only one wired to `configure_subprocess()`. Confirmed structurally in vLLM 0.26.0 source, not just by absent log output. `numa-bind: true` is a no-op for any Ray-executor job on this vLLM version. |

**Performance on Slingshot**: still not benchmarked head-to-head — both use
NCCL for GPU communication under the hood, so the data path itself is
identical; whether MP's different process-management/connection-setup
timing interacts any differently with Slingshot's CXI fabric than Ray's does
remains an open, empirical question (`ivllm bench` — see `scope.md` — is the
tool that could now actually answer this, unlike when this section was
first written).

**The multi-node hang investigated at length in `design/active-issues.md`
reproduces under both executors, with the identical `shm_broadcast`
signature** (`EngineCore` waiting forever for a worker response that never
arrives) — this rules out either executor's process-management layer as the
root cause; the leading theory implicates `vllm/distributed/device_communicators/shm_broadcast.py`
itself, a mechanism both `MultiprocExecutor` and `RayExecutorV2` share
(`RayExecutorV2` extends `MultiprocExecutor` and reuses its response-queue
plumbing). Separately, the "4-node SLURM job only gets 2 nodes to come up"
issue (also in `active-issues.md`) was confirmed present on the mp path
before the CPU-allocation fix that was found via the ray path — i.e. neither
issue is executor-specific.

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

However specific per model tested configuration has a benefit to control the
models we support.

### 4. Clean worker coordination pattern

The SLURM template coordinates head and worker nodes using marker files
in a shared tmp directory:
- Head writes `$JOB_TMP/markers/done` with exit code when user script completes
- Workers poll for this file and shut down gracefully
- Workers write `$JOB_TMP/markers/worker_done_<rank>` to confirm TCPStore
disconnection before head exits

This is similar to our lockfile based approach but simpler.

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
| Model recipe YAML | `model_recipes.yaml` | Auto-config for 100+ models |
| Worker coordination markers | `serve_vllm_mp.slurm` | Clean multi-node teardown |
| Pre-built containers | `vllm.def` + sifter registry | Avoids pip install maintenance |
| Debug mode | `serve_vllm_mp.slurm` | Easier multi-node diagnostics |

**Full source**: [github.com/UKGovernmentBEIS/isambard_containers](https://github.com/UKGovernmentBEIS/isambard_containers)
