# Architecture Decision Records — isambard-vllm v3

This document records architectural decisions for the v3 redesign.
Older decisions from v1-v2 are archived in `design/old/adr.md`.

---

## ADR-101: Lifecycle ownership on COMPUTE node

**Status**: Accepted

**Context**: In v2, the LOCAL client owned the full inference lifecycle
(session-owner pattern, ADR-002). This meant the vLLM instance was
tightly coupled to the client's SSH session — if the client disconnected,
the job was cancelled. This prevented detach/reattach, multi-user access,
and made the system fragile.

**Decision**: Move lifecycle ownership from LOCAL to COMPUTE. The bash
framework on the compute node manages its own lifecycle: startup monitoring,
health checks, idle timeout detection, and graceful shutdown. The LOCAL
client is a thin coordinator that can connect or disconnect freely.

**Rationale**:
- Enables detach/reattach: client comes and goes, job persists
- Enables multi-user: any project member can connect to a running job
- Eliminates SSH as a single point of failure for job lifetime
- Compute node is the natural authority for whether to keep running
- Simplifies LOCAL code: no more heartbeat, no more session ownership

**Consequences**:
- Bash framework must be robust (tested with mock harness)
- Exit traps are critical — must handle all exit paths
- Lockfile becomes the single source of truth for job state
- v2's `shutdown()` function in `session-helper.ts` is replaced by bash `tidy_up`

---

## ADR-102: Lockfile-driven communication protocol

**Status**: Accepted

**Context**: v2 used `job_details.json` as a simple status channel (4 states:
pending → initialising → running → failed/timeout). It was written by the
SLURM script and polled by LOCAL. There was no mechanism for the user to
request shutdown through the lockfile, and no idle timeout tracking.

**Decision**: Replace `job_details.json` with `status.json` using a richer
schema and a 6-state model:

States: `pending → initialising → running → (failed | stopped)`
Request state: `cancel` (written by user, consumed by monitor)

Full schema in `design/architecture.md`. Key additions:
- `cancel` as a non-terminal request state
- `idleTimeout` for automatic shutdown after inactivity
- `vllmPid`, `requestedTime`, `startTime`, `stopTime`, `reason`, `exitCode`
- `jobName`, `model`, `serverPort` for discovery

**Rationale**:
- `cancel` allows graceful shutdown without `scancel` — the job cleans up
- Enables idle timeout: monitor reads timeout config from lockfile
- Enriches diagnostics: `reason` and `exitCode` help debug failures
- Discovery: `ivllm list` reads lockfiles to show all jobs

**Consequences**:
- Lockfile must be atomically created (`set -C`) and safely updated (`tmp + mv`)
- `cancel` state is not terminal — monitors must detect it and transition to `stopped`
- Bash `jq` is the only dependency (already assumed in v2)
- Old `job_details.json` format must be migrated

---

## ADR-103: Bash framework as the HPC runtime

**Status**: Accepted

**Context**: v2 embedded SLURM/bash scripts inside TypeScript template
strings (`src/templates/inference.ts`, 1,176 lines). This made the bash
code unlintable, untestable, and hard to evolve. Every change required
TypeScript recompilation.

**Decision**: Extract all HPC-side logic into standalone bash scripts
under `$PROJECTDIR/engine/lib/`:

| File | Responsibility |
|------|---------------|
| `src/engine/lib/utils.sh` | Lockfile management, cache save/restore, monitor triad, exit trap, diagnostics |
| `src/engine/lib/common-env.sh` | NVHPC/NCCL/Slingshot environment setup |
| `src/engine/lib/vllm-env.sh` | vLLM-specific environment variables |
| `src/engine/lib/slurm-vllm-serve.sh` | Run a model on one or more nodes |
| `src/engine/lib/slurm-vllm-setup.sh` | vLLM installation |
| `src/engine/lib/slurm-hf-download.sh` | Model download via `srun` |
| `src/engine/lib/run_head_vllm.sh` | Head node vLLM launcher |
| `src/engine/lib/run_worker_vllm.sh` | Worker node vLLM launcher |
| `src/engine/ivllm-*.sh` | Login-node wrapper scripts (serve, status, setup, cancel, show-log, get-model) |

SLURM job scripts become thin wrappers that `source` these libraries.

**Rationale**:
- Bash code is testable independently (`test-vllm.sh` mock framework)
- No TypeScript recompilation needed for HPC-side changes
- Scripts can be edited directly on the HPC for rapid iteration
- Separation of concerns: TypeScript handles user interaction and SSH;
  bash handles compute-side orchestration
- The v2 `renderNVHPCPreamble()` function contains years of trial-and-error
  tuning — moving it to `vllm-env.sh` preserves it in a directly usable form

**Consequences**:
- Must maintain a `test-vllm.sh` harness that mocks `srun`, `scancel`, and `vllm`
- Scripts must be idempotent and safe to source multiple times
- Must handle `bash -u` / `set -euo pipefail` for robustness
- Scripts are installed by `ivllm setup` alongside the vLLM venv

---

## ADR-104: Detach/Reattach model

**Status**: Accepted

**Context**: v2's session-owner pattern meant the LOCAL process had to stay
alive for the entire inference session. This was fragile (network issues,
laptop sleep, accidental Ctrl+C) and single-user.

**Decision**: The LOCAL client can safely disconnect at any point. The
compute job continues to run. Reconnection is a simple `ivllm connect <job>`:

1. Read `status.json`
2. If `running` → establish SSH tunnel, print endpoint, exit (or stay in monitor mode)
3. If `stopped` or `failed` → re-run `slurm.sh`
4. If `cancel` → warn that shutdown is in progress

No reconnection state is needed on the compute side — the lockfile is
the only coordination point.

**Rationale**:
- Drop-dead simple: no state machines, no session tokens, no PID files
- Lockfile already contains everything needed (compute hostname, port, model)
- SSH tunnel is stateless — creating a new one on reconnect is trivial
- Works seamlessly with multi-user: any user can `connect` to any running job

**Consequences**:
- The `ivllm connect` command must handle all lockfile states gracefully
- Idle timeout is essential to prevent abandoned jobs consuming GPU hours
- No heartbeat from LOCAL — `monitor_head` on COMPUTE is the only authority
- COMPUTE must manage shutdowns and write state changes on crashes and slurm timeouts
- Force cancel (using slurm scancel) and clean up from the client in case of inconsistent state.

---

## ADR-105: Idle timeout monitoring (backend-specific)

**Status**: Accepted (design intent — open issue)

**Context**: Without idle timeout, a job started and abandoned by a
disconnected client would run until its SLURM time limit (default 4h,
potentially 8h+). This wastes expensive GPU hours on Isambard.

However, idle timeout detection is inherently **backend-specific**. The
Isambard backend uses vLLM access logs with known timestamp formats, but
other backends (containers, Ollama, different HPCs) will have different
logging, different process models, and different ways to detect idleness.

Approaches considered (and their limitations):

| Approach | Problem |
|----------|--------|
| **Parse access log timestamps** | Backend-specific log format. Fragile across vLLM versions. |
| **Check log file mtime** | Health check heartbeats and `/health` probes also write to the log, giving false positives. A filter could exclude `/health` lines but this adds complexity. |
| **Poll vLLM `/metrics` endpoint** | Requires the backend to expose Prometheus metrics (not universal). Adds extra request load. |
| **Sidecar that tracks requests per model** | Another process to manage. Adds deployment complexity. |
| **Process-level monitoring (/proc/net)** | OS-specific. Doesn't work inside containers. |
| **inotify on log file** | Lustre (parallel filesystem) doesn't support inotify. Linux-specific. | **clients periodically touch lockfile & check logfile mtime** | places responsibility on clients to keep connection alive but could be backend independent

**Decision**: Idle timeout is a per-backend responsibility. The lockfile
carries `idleTimeout` (in minutes, or `-1` for never) as a cross-backend
contract. Each backend's monitoring process enforces it however it sees fit.

For the **Isambard vLLM backend** (the bash framework's `monitor_head`):

Parse the vLLM access log incrementally:
1. Track the last byte offset read from the log file
2. Each cycle, read only new bytes (`tail -c +$LAST_OFFSET`)
3. If new lines contain "real" API endpoints (`/v1/chat/completions`,
   `/v1/models`, `/v1/completions`, not `/health`), record the current time
4. If time since last real request > idleTimeout → shutdown

Baseline VLLM log line format does not include dates unless configured - requires vllm_logs.json configuration file and `export VLLM_LOGGING_CONFIG_PATH="vllm_logs.json"` to be present on the HPC. With this in place we get logs like (confirmed from live vLLM run):

```
(APIServer pid=34633) [2026-07-14 22:37:50,765] INFO:     10.242.0.28:38194 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=34633) [2026-07-14 22:37:50,935] INFO:     10.242.0.28:45178 - "GET /health HTTP/1.1" 200 OK
```

**Rationale**:
- Decouples idle detection from the generic lockfile protocol
- Each backend uses the most natural approach for its environment
- The Isambard backend's log parsing is simple, proven, and doesn't require
  changes to vLLM
- Incremental reading (byte offset) is efficient even on Lustre

**Consequences**:
- The `idleTimeout` lockfile field is a contract, not an implementation
- Backend implementations document their idle detection strategy
- For Isambard: `VLLM_LOGGING_CONFIG_PATH` + `vllm_logs.json` are required
- `idleTimeout: -1` means "never timeout" (explicit choice)
- This remains an open design area — better approaches may emerge

---

## ADR-106: Multi-user access via shared filesystem

**Status**: Accepted

**Context**: v2 supported multi-user vLLM installation (ADR-013) but the
session-owner pattern meant only one user could interact with a running job
at a time.

**Decision**: All artifacts in `$PROJECTDIR/engine/` use group-writable
permissions (`umask 0002`, `chmod g+w`). Any project member can:

- Run `ivllm list` to see all jobs
- Run `ivllm connect <job>` to attach to a running instance
- Run `ivllm cancel <job>` to request shutdown
- View logs and diagnostics for any job

The only restriction is that SLURM `scancel` requires job ownership
(or `--uid` with appropriate permissions).

**Rationale**:
- Maximises utilisation: a single 4-node job can serve the whole team
- Eliminates HuggingFace rate limits: one download serves everyone
- Simplifies onboarding: new team members just configure their CLI

**Consequences**:
- Lockfiles must be created with group-writable permissions
- JIT caches must be group-readable (already handled by tar `--mode='g+rwX'`)
- SLURM timeouts from one user's session affect all connected users
- Must document the multi-user workflow clearly

---

## ADR-107: Unified CLI — `connect` replaces `start`/`interactive`

**Status**: Accepted

**Context**: v2 had two separate commands for starting a job:
`ivllm start` (sbatch, background) and `ivllm interactive` (srun, TTY-bound).
They shared most of the same code path but had different submission
mechanisms and different user experiences. The distinction confused users.

**Decision**: Replace both with a single `ivllm connect` command that:

1. If the job doesn't exist or is in a terminal state (`stopped`/`failed`):
   - Submit via `sbatch` (background) by default
   - Offer `--batch` flag for submission to standard non-interactive queue
   - tail remote logfiles over SSH to monitor startup progress.
2. When the job is `running`:
   - Establish SSH tunnel immediately
   - Print endpoint URL
   - Stay in foreground with optional monitoring mode
3. If the job is `initialising` or `pending`:
   - Tail logs and wait for `running`
   - Then establish tunnel

**Rationale**:
- Single mental model: "I want to connect to my model"
- default is interactive reservation (can use slurm )
- Handles all job states (new, running, stopped, failed)
- No confusion about which command to use

**Consequences**:
- `ivllm start` and `ivllm interactive` are deprecated and removed
- `ivllm stop` is replaced by `ivllm cancel` (with `--force` for hard kill)
- Documentation must be updated to use `connect` everywhere

---

## ADR-108: Preserve v2 TypeScript patterns that work

**Status**: Accepted

**Context**: The v2 codebase has several well-tested, well-designed
TypeScript modules that don't need rewriting.

**Decision**: Preserve the following components with minimal changes:

| Module | Justification |
|--------|--------------|
| `src/config.ts` | Clean interface, file-based persistence, CLI flag parsing |
| `src/ops/RemoteOps.ts` + `src/ops/SshRemoteOps.ts` | Multiplexed SSH, SCP, tunnel spawning — all battle-tested |
| `src/local-ops.ts` | Port check, health check, model query |
| `src/semver.ts` | Version comparison, sorting (recreated for v3) |
| `src/backends/Backend.ts` | Abstract Backend class for lifecycle management |

**Rationale**:
- Rewriting working code is wasted effort
- These modules have ~300 tests covering them
- The bugs and edge cases are already fixed
- They are naturally separated from the session-owner pattern

**Consequences**:
- Codebase retains a mix of TypeScript and bash — clear module boundaries
- New commands (`connect`, `cancel`) reuse existing modules
- Only the session-owner modules are replaced

---

## ADR-109: Project structure on HPC

**Status**: Accepted

**Context**: v2 stored job data in `$HOME/<job>/` (lockfiles, logs, scripts,
configs). This made it hard for team members to find each other's jobs and
mixed job artifacts with user home data.

**Decision**: Define a fixed structure under `$PROJECTDIR/engine/`:

```
$PROJECTDIR/engine/
├── lib/
│   ├── utils.sh         ← Lockfile management, monitors, cache, shutdown
│   ├── vllm-env.sh      ← NVHPC/NCCL/Slingshot environment
│   ├── hf.sh            ← Model download
│   └── vllm-setup.sh         ← vLLM installation (one-off, run by `ivllm setup`)
├── jobs/
│   └── <jobname>/
│       ├── status.json
│       ├── jit-cache.tar.gz
│       ├── vllm.<nodeid>.log
│       ├── vllm.yaml
│       └── slurm.sh
├── vllm/
│   ├── vllm-setup.sh
│   ├── vllm_logs.json
│   ├── plugins/
│   └── <version>/       ← vLLM venv (from `ivllm setup <version>`)
├── hf/                   ← Shared HuggingFace cache
│   └── hub/
│       └── models--<name>/
└── diagnostics/
    └── <jobname>/
        └── <date>/
            ├── vllm.<id>.log
            ├── vllm.yaml
            └── slurm.sh
```

**Rationale**:
- Predictable paths: CLI doesn't need to ask where things are
- Multi-user: `jobs/` and `diagnostics/` are group-readable
- Self-contained: everything for a job is in one directory
- Clean separation: framework libs, job data, model cache, diagnostics

**Consequences**:
- `ivllm setup` must create the `engine/` directory structure
- Old `$HOME/<job>/` directories must be migrated or deprecated
- All path resolution in `src/job.ts` must be updated
- `diagnostics/` captures logs and configs from failed jobs automatically

---

## ADR-110: Backend-agnostic lockfile protocol

**Status**: Accepted (design intent)

**Context**: The lockfile protocol (ADR-102) currently has mechanism-specific
fields like `slurmJobId`, `computeHostname`, and `vllmPid`. Future backends
(Ollama, other HPCs, containers) will have different runtime metadata.

**Decision**: The `status.json` schema is backend-agnostic at the top level.
Backend-specific metadata lives in a `backend` namespace object (optional).

Top-level fields (every backend):
- `status`, `jobName`, `model`, `serverPort`, `user`, `requestedTime`, `idleTimeout`
- `startTime`, `stopTime`, `reason`, `exitCode`
- `slurmJobId`, `computeHostname`, `vllmPid` (Isambard-specific, at top level for convenience)
- `backend`: string identifier (e.g. `"isambard-vllm"`, `"ollama"`) — **planned, not yet implemented**
- `backendConfig`: optional JSON object (backend-specific, opaque to the CLI) — **planned, not yet implemented**

Example for Isambard:
```json
{
  "status": "running",
  "jobName": "qwen36",
  "model": "Qwen/Qwen3.6-35B-A3B-FP8",
  "serverPort": 49153,
  "user": "testuser",
  "requestedTime": "2026-07-14T12:00:00+00:00",
  "idleTimeout": 30,
  "slurmJobId": "123456",
  "computeHostname": "nid12345",
  "vllmPid": 12345
}
```

**Note**: The `backend` and `backendConfig` fields are planned for future multi-backend support
but are not yet implemented in the actual lockfile schema (`LockfileV3` in `src/types.ts`).

**Rationale**:
- The CLI reads only top-level fields (`status`, `jobName`, `model`, `serverPort`)
  to determine what to display and how to connect
- Backend-specific metadata is opaque to the CLI but essential for the backend
  runtime (monitors, diagnostics, cancellation)
- Adding a new backend doesn't require changing the lockfile schema

**Consequences**:
- The CLI must use `backend` field to select which backend implemention to use
- Backend implementations must write their own `backendConfig` format
- Migration: existing `slurmJobId` and `computeHostname` move into `backendConfig`
- The lockfile always has a `backend` field (even for single-backend setups)

---

## ADR-111: Backend interface abstraction

**Status**: Accepted (design intent)

**Context**: Currently all backend logic (SSH, SLURM, vLLM lifecycle) is
interwoven in the CLI code. Adding new backends requires touching the same
files and risks breaking Isambard support.

**Decision**: Define a standard `Backend` interface in TypeScript. The
Isambard backend is the first implementation; new backends implement the
same interface.

```typescript
abstract class Backend {
  abstract bootstrap(): Promise<void>;
  abstract setup(version: string): Promise<void>;
  abstract connect(job: string, port: number): Promise<CloseableEventEmitter>;
  abstract requestCancel(job: string, force: boolean): Promise<void>;
  abstract requestStart(job: string, maxTime: string, monitor: boolean, config?: string): Promise<void>;
  abstract getAllJobStatus(): Promise<LockfileV3[]>;
  abstract watchLog(job: string, node?: string, until?: string): Promise<CloseableEventEmitter>;
  getJobStatus(job: string): Promise<LockfileV3>;
  isRunning(job: string): Promise<boolean>;
  isStopped(job: string): Promise<boolean>;
  isStartable(job: string): Promise<boolean>;
  isStarting(job: string): Promise<boolean>;
}
```

**Note**: The actual `Backend` class (in `src/backends/Backend.ts`) evolved from the
interface shown above. The final signature uses `requestCancel`/`requestStart`/`getAllJobStatus`
instead of `cancel`/`forceCancel`/`status`/`list`/`tailLogs`. The `bootstrap()` and
state-helpers (`isRunning`, `isStopped`, etc.) were added during implementation.

**Rationale**:
- Clean separation: new backends don't touch existing code
- The `connect` method is the key abstraction — it owns the full startup
  sequence and returns a tunnel-able endpoint
- CLI commands (`connect`, `cancel`, `list`) delegate to the backend

**Consequences**:
- Existing `remote-ops.ts` and bash framework become the Isambard backend
- The CLI needs a backend registry (default + named backends)
- `ivllm connect <job> --backend <name>` selects an alternative
- The `Backend` interface may evolve as new backends are added

---

## ADR-112: Port pool for multi-model routing

**Status**: Accepted (design intent)

**Context**: Running multiple models requires a registory of local port
assignments per model and needs to manage conflict
avoidance when multiple models share a node and support a router to know
where each model is.

There are 2 types of port assignment:

1) server port where vllm is exposed on compute node.
This is strictly the responsibility of the backend to decide, and can used a
random high port (preferred) or vllm default 8000. This is up to the backend
but must be communicated to clients via the lockfile. In the context of a single
node running multiple models the backend will have to make sure there are no
conflicts.

2) Local ports: localhost endpoints for ssh tunnel (or passthrough when backend is local ollama).
We need a port per model running. This should default to 11434 if only a single
model is connected. The user can override this with a --local-port cli flag to
`ivllm connect`. Currently it is the users responsibility to manage local ports in multiple connections.

In the future a model router will handle multiple local ports and ssh tunnels:

Local port assignment (router mode):
- Discover running models from backends.
- Construct multiple ssh tunnels to endpoints using random high local ports.
- Maintain an emphemeral mapping in router, from model name to local port.
- Serve router on 11434, and route requests from client to backend depending on
model name
- Some routing heuristics on model name collision across backends will be required

**Rationale**:
- Stable port for client in pre-router world: agents can be configured once and
  keep working across restarts
- Port range is known by user: firewall rules, `sbx policy allow`, and SSH config
  can target the full range

**Consequences**:
Backend:
- If a single node is running multiple models the backend will need heuristics
  to prevent vllm server port collisions.
- Lockfile schema unchanged (already has `serverPort`)
Client (pre-router):
- Default single-model mode uses port 11434 unless overridden
- Client needs to check for existing ssh tunnels and port usage when connecting.
Future router implementation:
- Router runs on local but outside of any sandboxes.
- The router reads remote port allocations to build its model catalog dynamically
- Router listens on 11434.

---

## ADR-113: Each model is an independent job

**Status**: Accepted (design intent)

**Context**: Running multiple models on a single node (e.g. Qwen3.6 and
Gemma4 sharing 4 GPUs) could be implemented as a single SLURM job managing
multiple vLLM processes, or as independent SLURM jobs at the project level.

**Decision**: Each model is a completely independent job. Multiple models
on one node use the `interactive` partition with explicit GPU affinity
(`CUDA_VISIBLE_DEVICES`). Each gets its own:

- `status.json` in `$PROJECTDIR/engine/jobs/<name>/`
- SLURM job allocation (or share a single srun for the interactive partition)
- vLLM process with its own port
- `monitor_head` process
- JIT cache
- Idle timeout timer

**Rationale**:
- Independence: one model crashing doesn't affect others
- Simplicity: the same code path handles single-model and multi-model cases
- Resource control: GPU and memory limits are explicit per model
- Clean-up: cancelling one model doesn't cancel others
- The monitor triad works per-model without changes

**Consequences**:
- Must support `CUDA_VISIBLE_DEVICES` in the SLURM script preamble
- Resource tracking (`gpus`, `memoryGb`, `cpuCores`) in lockfile's
  `backendConfig` helps the router/node manage allocation
- Multi-model on a single node requires the `interactive` partition
  (or a single sbatch job with multiple srun steps)
- The monitor must be GPU-aware: shutdown of one model should not affect
  others sharing the same node

---

## ADR-114: Dual installation path — bare-metal and container

**Status**: Accepted (design intent)

**Context**: The v2 codebase installs vLLM via bare-metal `pip install` into
a versioned venv (ADR-011/013). The [UKGovernmentBEIS/isambard_containers](https://github.com/UKGovernmentBEIS/isambard_containers)
project maintains pre-built Apptainer images with vLLM, CUDA, and
Slingshot/NCCL networking pre-configured for Isambard GH200.

Bare-metal and container approaches have complementary strengths:

| Aspect | Bare-metal (pip) | Container (Apptainer) |
|--------|-----------------|----------------------|
| Install time | 10-20 min (pip compile) | ~2 min (`sifter pull`) |
| CUDA version | 12.9 (NVHPC compat libs) | 13.0.2 (native) |
| vLLM compile | Wheel (pre-compiled) | Source (aarch64) |
| Maintenance burden | High (vllm-env.sh, deps) | Low (pre-built, upstream) |
| Debugging | Easy (native process) | Harder (inside container) |
| Proven on Isambard | Yes (extensive testing) | Yes (separate project) |

**Decision**: Support both installation methods. The SLURM scripts accept a
`CONTAINER` environment variable that selects which vLLM to use:

```bash
# Bare-metal path (default)
source $PROJECTDIR/engine/vllm/0.22.0/bin/activate
vllm serve --config vllm.yaml

# Container path
CONTAINER=vllm-0.23.0_0.1.0.sif
singularity exec --nv $CONTAINER \
  --bind $PROJECTDIR:$PROJECTDIR \
  vllm serve --config vllm.yaml
```

The bash framework (lockfile management, monitors, cache, shutdown) is shared
by both paths — only the `vllm serve` invocation differs.

**Rationale**:
- Bare-metal is proven and debugged — keep it as the default
- Container path reduces maintenance and provides newer CUDA
- Users choose based on their needs (stability vs cutting-edge)
- The bash framework is the same either way
- Both paths produce the same lockfile format

**Consequences**:
- SLURM templates need a branching path (if `$CONTAINER` set → use container)
- Model recipes (ADR-115) can specify preferred runtime per model
- Must maintain both paths through testing

---

## ADR-115: Model recipe database

**Status**: Accepted (design intent)

**Context**: Users currently must write a `vllm.yaml` file for every model,
specifying parallelism, memory, dtype, and other vLLM args. The
[UKGovernmentBEIS/isambard_containers](https://github.com/UKGovernmentBEIS/isambard_containers)
project maintains a YAML recipe file (`model_recipes.yaml`) with 100+ model
configurations with inheritance, which their `vllm-serve` command uses to
auto-configure any supported model.

**Decision**: Add a model recipe database (`models.yaml`) that maps HuggingFace
model IDs to vLLM args and environment variables. Recipes support inheritance
from base configs. `ivllm connect <model>` without a `--config` file
auto-configures from the database.

```yaml
# models.yaml (shipped with ivllm)

_deepseek_base:
  vllm_args:
    enable_expert_parallel: true

deepseek-ai/DeepSeek-R1-0528:
  base: _deepseek_base
  vllm_args:
    tensor_parallel_size: 4
    pipeline_parallel_size: 4
    reasoning_parser: deepseek_r1
```

**Rationale**:
- Eliminates the need for per-job config files for popular models
- Recipes encode hard-won tuning so knowledge is shared across the team
- Inheritance avoids duplication across model families
- Users can add their own recipes locally

**Consequences**:
- `models.yaml` bundled with ivllm, installed on HPC via `ivllm setup`
- `ivllm connect <model>` falls back to requiring `--config` if model not in DB
- `--config` flag still supported for custom models

---

## ADR-116: Elimination of recursive `chmod -R` permissions logic on HPC

**Status**: Accepted

**Context**: In v2 and early v3, helper functions inside `src/engine/lib/utils.sh` (such as `resolve_vllm_dir`, `resolve_job_dir`, etc.) used recursive `chmod -R g+rw` commands to ensure all files and subdirectories inside the shared project space remained group-writable and accessible under multi-user access permissions (ADR-106).

However, recursive `chmod -R` operations are highly problematic on HPC filesystems (like Lustre on Isambard-AI or GPFS):
1. **Performance Hangs (O(N) Complexity):** As Python virtual environments and model caches grow to contain hundreds of thousands of files, walking the directory tree recursively becomes extremely slow, eventually causing scripts to hang for minutes or hours.
2. **High-Frequency Polling Contention:** Because `resolve_job_dir` was called inside status-checking functions during the 10-second polling monitor loop, the script performed `chmod -R` on the entire jobs directory every 10 seconds.
3. **Multi-User Permission Errors:** In a shared group directory, a user lacks permissions to change the mode of files owned by another user. Running `chmod` on files owned by others returned thousands of "Operation not permitted" blocks and warnings, triggering filesystem locking backups.

**Decision**: Remove all recursive `-R` flags from all `chmod` commands inside `src/engine/lib/utils.sh`. Replace them with non-recursive permissions settings on directory creation (namely, `chmod g+rwX` on specific pathways directly when directories are initialized or created). 

To ensure files created inside the shared folder natively inherit group ownership and write permissions without manual recursion support:
1. Enable the directory Set-Group-ID (**SGID**) flag (`chmod g+s dir`) upon initial setup.
2. Configure Default Access Control Lists (**ACLs**) at the project directory level (using `setfacl -d -m g::rwX dir` and `setfacl -m g::rwX dir`).

**Rationale**:
- Eliminates recursive directory traversal entirely, turning path validation from $O(N)$ and loop-contended time into an instant $O(1)$ operation.
- Natively resolves filesystem metadata lock contention on Lustre, preventing script execution hangs.
- Natively delegates permission and group inheritance to the operating system/filesystem via SGID and default ACLs, avoiding "Operation not permitted" warnings in multi-user environments.

**Consequences**:
- The installation process relies on the parent directories having the correct ACLs and SGID bit configured once from the CLI (e.g. during project space onboarding).
- No functional regressions for the test suite, as mocked sandbox behaviors continue to function seamlessly without the recursive flag.

---

## ADR-117: Support for UCCL-EP as a high-performance open-source expert-parallel alternative on HPE Slingshot

**Status**: Accepted

**Context**: Mixture-of-Experts (MoE) models (like DeepSeek-V3 or DeepSeek-R1) rely on expert-parallel (EP) communication kernels to achieve high GPU efficiency. Historically, vLLM compiles the NVIDIA-optimized `deep_ep` package for this purpose. 

However, DeepEP is strictly bound to Mellanox Connect-X InfiniBand hardware, direct NVSHMEM GPU-Initiated Networking (GIN) parameters, and Mellanox OFED developmental headers (`mlx5dv.h`). On supercomputers like Isambard-AI Phase 2 which utilize the **HPE Slingshot 11 interconnect** (powered by Cassini ASICs and standard `aws-ofi-nccl` libfabric transport), DeepEP fails to compile due to missing vendor-specific headers, and cannot execute at runtime since Cassini NICs speak libfabric rather than InfiniBand verbs.

The open-source **UCCL-EP** (Unified Collective Communication Library - Expert Parallel) project implements the identical high-performance, GPU-initiated MoE collectives interface but abstracts transport interactions. A custom fork of UCCL-EP maintained by `doublewordai` on the **`pr997-swiss-cxi`** branch explicitly integrates native HPE Slingshot 11 support via the libfabric `CXI` transport layer on Swiss Alps and Isambard-AI, and includes a full `deep_ep_wrapper` drop-in replacement package.

**Decision**: Modify `src/engine/lib/slurm-vllm-setup.sh` to implement a robust dual-path installation workflow for expert-parallel collectives:
1. Try compiling the standard `deep_ep` library first.
2. If `deep_ep` fails to compile (expected on SLES bare-metal on Slingshot), catch the block and fall back to UCCL-EP.
3. Automatically compile the user-space **`rdma-core`** library from source if standard InfiniBand development headers (`infiniband/verbs.h`) are missing on SLES bare-metal. To ensure libraries survive job tear-downs, they are compiled persistently inline inside the virtual environment directory (`$vllmVersionDir/rdma-core/`) and dynamically resolved at runtime inside **`src/engine/lib/vllm-env.sh`** by dynamically appending it to `LD_LIBRARY_PATH`. This makes compilation and runtime resolution fully stable, self-contained, and persistent on any subsequent compute nodes.
4. Automatically install `nanobind` (build dependency for UCCL-EP), clone the **`doublewordai/uccl`** fork on the **`pr997-swiss-cxi`** branch, and compile it with Slingshot CXI transport optimization flags (`USE_LIBFABRIC_CXI=1` and `USE_DMABUF=1`), installing the core `ep` package.
5. If compiled successfully, install the included **`deep_ep_wrapper`** package, which serves as a complete, pre-configured high-fidelity `deep_ep` drop-in replacement so that vLLM can seamlessly import and leverage UCCL-EP without patching downstream source files.

**Rationale**:
- **Completeness:** Promotes optimal high-performance MoE serving throughput for DeepSeek models on Isambard-AI's HPE Slingshot 11, matching the physical interconnect technology.
- **Zero Code Modification:** Creating a package shim in `site-packages/deep_ep` lets vLLM seamlessly import and leverage UCCL-EP without patching vLLM's internal python source files.
- **Resilience:** If UCCL-EP compilation fails, standard vLLM operations continue unaffected, falling back gracefully to PyTorch or default MoE kernels.

**Consequences**:
- `slurm-vllm-setup.sh` is fully self-healing: if either library fails to compile, it catches the error and proceeds, allowing standard vLLM and standard models to install successfully (exit code 0).
- Because UCCL-EP's core headers (like `proxy_ctx.hpp`) are still hard-coded to include `<infiniband/verbs.h>` for structural declarations, its bare-metal compilation still requires `libibverbs`. On HPE Slingshot SLES bare-metal (which lacks InfiniBand development headers), UCCL-EP compilation will be safely skipped with a warning.
- To run high-performance expert-parallel kernels (DeepEP/UCCL-EP), users are advised to run vLLM via the `isambard_containers` Apptainer container, which gets around SLES bare-metal header constraints by pre-installing standard InfiniBand development packages inside its Ubuntu build layer.
- The `ep` package and its `deep_ep` shim are transparently managed within the venv side-packages when compiling successfully.


