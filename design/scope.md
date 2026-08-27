# Roadmap — isambard-vllm (ivllm) v3

This document captures the current scope of the v3 rewrite.

Note: `abort` flag on cancel is documented below (§1, `ivllm cancel`).
Benchmarking (`ivllm bench`) is documented below (§1). Ray support
(`distributed-backend-executor: ray`) is documented in `architecture.md`'s
"Multiprocessing (MP) vs Ray" section. `IVLLM_PROJECTDIR`/project-dir
resolution and `IVLLM_DEBUG`/diagnostics logging levels are documented in
`ivllm-environment.md` and `knowledge-base.md` respectively (in progress —
see `design/priorities.md`).

## Current Scope

The v3 rewrite from v2 is **functionally complete**. All core commands, the
backend abstraction, the bash lifecycle framework, and the test harness are
implemented and tested.

### 1. CLI Commands

Seven command groups (`connect`, `cancel`, `status`, `config`, `setup`,
`diagnostics`, `bench` with its `submit`/`status`/`results` subcommands),
all registered via Commander.js in `src/index.ts`.

#### `ivllm connect <job>`

Primary entry point. Handles all job states in a single flow:

| Job State | Behaviour |
|-----------|-----------|
| **Never run** (no lockfile) | Creates `pending` lockfile → submits via `sbatch` → tails logs → establishes SSH tunnel once healthy |
| **Running** | Establishes SSH tunnel immediately, prints endpoint, exits |
| **Stopped / Failed** | Restarts from the stopped state (same as never-run) |
| **Initialising / Pending** | Tails remote logs and waits for `running` |

Options:
- `--config <file>` — vLLM YAML config (required for first use)
- `--local-port [port]` — Local port for the API (default `11434`)
- `--batch` — Submit to standard non-interactive queue (default: interactive partition)
- `--time <hh:mm:ss>` — SLURM time limit (default `08:00:00`)

#### `ivllm cancel <job> [--force] [--abort]`

| Mode | Behaviour |
|------|-----------|
| **Graceful** (default) | Writes `cancel` to the lockfile; the compute-side monitor detects it and shuts down vLLM cleanly |
| **Force** (`--force`) | Runs `scancel` directly on the SLURM job, then updates the lockfile |
| **Abort** (`--abort`) | Writes `abort` to the lockfile; the compute-side monitor detects it and shuts down vLLM uncleanly capturing diagnostics. useful for debugging hangs |

#### `ivllm status [job]`

- Without arguments: shows a table of all known jobs (status, model, port, timestamps, reason)
- With `<job>`: shows a formatted row for that specific job

#### `ivllm config [--login-host <host>] [--username <user>] [--project-dir <path>] [--hf-token <token>]`

- With no arguments: displays current configuration
- With flags: updates and persists the given fields to `~/.config/ivllm/config.json`

#### `ivllm setup <version> [--force]`

One-off installation of vLLM on the HPC:
- Submits a SLURM job on a compute node
- Installs NVIDIA HPC SDK 26.3 (CUDA 12.9 forward compatibility) and the specified vLLM version
- Stores the venv in a shared versioned directory at `$PROJECTDIR/engine/<version>/`
- Progress streamed to terminal; ~10–20 min on first run
- Skipped automatically if that version is already installed
- `--force` reinstalls even if the version exists

#### `ivllm diagnostics <job> [--out <path>]`

* On failure to start up the vllm config and vllm logs from the current run
are need to be copied to /engine/diagnostics/<job>/<date-time> directory
* This command syncs job diagnostics to a user local directory so that logs
can be analysed by an agent on local.

#### `ivllm bench submit|status|results` — [experimental]

Submits a set of vLLM configs as a benchmark comparison, run for real on the
production `ivllm-serve.sh` path (not a parallel/synthetic launch path — see
ADR-118 in `adr.md` for the design history and why this matters). Fully
fire-and-forget: the CLI submits and returns; a detached login-node
orchestrator (`ivllm-bench.sh`) does everything else and can keep running for
hours after the SSH session that launched it closes.

| Command | Behaviour |
|---------|-----------|
| `ivllm bench submit <comparison> [configs...]` | Uploads one or more vLLM YAML configs (job name = filename stem) to `<comparison>/`; if no configs are given, uses every `*.yaml` already in that directory. Launches `ivllm-bench.sh` detached (`nohup ... & disown`) on the login node. `--time <hh:mm:ss>` sets the per-job SLURM time limit (default `02:00:00`). |
| `ivllm bench status <comparison>` | Reads `<comparison>/benchmarking_status.json` and prints per-job counts/status. `complete: true` is the only authoritative "safe to fetch" signal. |
| `ivllm bench results <comparison> [outDir]` | If complete, rsyncs `<comparison>/results/` (default `<comparison>/result` locally) — otherwise prints the current status and exits non-zero. |

**What `ivllm-bench.sh` actually does** (standalone, sourced by tests, only
runs `main()` when executed directly):
1. Creates a **per-comparison shadow project directory** at
   `<comparison>/benchmark/`, with `engine/{vllm,nvhpc,rdma}` and `model`
   symlinked back to the real `$PROJECTDIR` (nothing re-downloaded or
   recompiled) and `engine/{jobs,diagnostics}` kept real/independent, so
   benchmark runs never collide with production jobs or each other across
   separate comparisons.
2. Prefetches every *unique* `.model` across all configs once, sequentially,
   before submitting anything (avoids N concurrent jobs racing to download
   the same model).
3. Submits every config via the real `ivllm-serve.sh -j <job> -b` (batch
   partition), in parallel, one background subshell per config.
4. Once each job reaches `running`: runs a health check, then
   `vllm bench serve` via `srun --overlap --jobid=<slurmJobId>` targeting
   the job's own `computeHostname` — this executes the bench client *on the
   compute node*, genuinely co-located with the API server (the login node
   cannot reach a compute node's port directly), which is also a more
   honest latency measurement than a login-node network hop would have
   been.
5. Requests a graceful cancel, waits for the job to reach a terminal state,
   then explicitly calls `capture_job_diagnostics()` (the cancel/idle-timeout
   exit paths don't call this themselves — only crash paths do — so a
   *successful* benchmark run needs it called explicitly to archive its
   `bench.json`/logs).
6. Writes `benchmarking_status.json` on a poll loop throughout, and a final
   `results/summary.txt` table (job, run timestamp, status, reason,
   req/s, output tok/s, mean TTFT) once every job is terminal. Deliberately
   prints no verdict — comparing the numbers is left to the caller.

Reruns of the same comparison directory *add* a new timestamped diagnostics
entry per job rather than overwriting history, so comparing a rerun against
a previous attempt after tweaking a config is directly supported.

See `design/adr.md` ADR-118 for the design history — the implementation
above is a further revision beyond what ADR-118's text currently describes
(no `IVLLM_BENCH_MODE` hook in `monitor_head()`, no `benchmarkProjectDir`
config field — both superseded by the standalone-orchestrator approach).

### 2. Backend Abstraction

An abstract `Backend` class in `src/backends/Backend.ts` defines the
lifecycle contract. A single concrete implementation — `IsambardBareMetalBackend` —
implements it for the Isambard AI HPC.

| Abstract method | Purpose |
|-----------------|---------|
| `bootstrap()` | Verify SSH, deploy engine scripts to HPC |
| `setup(version, force?)` | Install vLLM on the HPC |
| `connect(job, localPort)` | Establish SSH tunnel to a running job |
| `requestCancel(job, force, abort)` | Graceful, forced, or abort-with-diagnostics cancellation |
| `requestStart(job, maxTime, batch, config?)` | Submit a new SLURM job |
| `getAllJobStatus()` | List all jobs as lockfile objects |
| `watchLog(job, node?, start?)` | Tail log output |
| `fetchDiagnostics(job, localDest?)` | Download a job's archived diagnostics tree |
| `getJobStatus(job)` | Get status of a single job |
| `isCancelling / isRunning / isStopped / isStartable / isStarting` | Lifecycle state helpers |
| `requestBenchmark / getBenchmarkStatus / fetchBenchmarkResults` | Non-abstract, default-throws — implemented only by backends supporting `ivllm bench` (§1 above) |

The `getBackend()` factory selects the implementation from a compile-time
`backendRegistry`. Currently only `isambard` is registered.

### 3. SSH Layer

`SshRemoteOps` (`src/ops/SshRemoteOps.ts`, 252 lines) provides a reusable
SSH/SCP/rsync layer with multiplexing:

- **Multiplexed connections** via `ControlMaster` — first connection spawns a
  background master; subsequent connections within 10 minutes reuse it, avoiding
  repeated handshakes and login rate-limits
- **`runRemote()`** — execute a login-node command and capture stdout
- **`runRemoteSync()`** — execute a command and stream output to the terminal
- **`copyFile()`** — SCP a single file to the login node
- **`copyDirectory()`** — rsync a directory up or down to the login node
- **`spawnTunnel()`** — persistent SSH port-forward: `localhost:<port>` →
  `<computeHost>:<serverPort>`, registered on the existing multiplexed
  `ControlMaster` via `ssh -O forward` (not a dedicated `-N -L` session — a
  dedicated session's process lifetime isn't a reliable stand-in for the
  tunnel's once a `ControlMaster` is involved). Liveness is polled directly
  (local port check) and teardown uses `ssh -O cancel`, so the tunnel can be
  closed without touching the shared master connection
- **`checkSSH()`** — verify connectivity before any operation

### 4. Bash Framework (HPC Runtime)

15 standalone bash scripts deployed to the HPC by `ivllm setup`. They are
separate from TypeScript — editable directly on the HPC without recompilation.

#### Login-node wrapper scripts (`engine/ivllm-*.sh`)

| Script | Purpose |
|--------|---------|
| `ivllm-serve.sh` | Submit a new SLURM job for a model |
| `ivllm-cancel.sh` | Write cancel to lockfile or force scancel |
| `ivllm-status.sh` | List all lockfiles as JSON |
| `ivllm-setup.sh` | One-off vLLM installation |
| `ivllm-show-log.sh` | Tail remote log files |
| `ivllm-get-model.sh` | Query model info from a running instance |

#### Shared libraries (`engine/lib/`)

| Library | Purpose |
|---------|---------|
| `utils.sh` | Lockfile management, cache, monitors, exit traps, diagnostics — 44+ documented functions |
| `common-env.sh` | NVHPC/NCCL/Slingshot environment setup and tuning |
| `vllm-env.sh` | vLLM-specific environment variables |
| `slurm-vllm-serve.sh` | SLURM job script for running vLLM |
| `slurm-vllm-setup.sh` | SLURM job script for vLLM installation |
| `slurm-hf-download.sh` | Model download via `srun` on interactive partition |
| `run_head_vllm.sh` | Head node vLLM launcher |
| `run_worker_vllm.sh` | Worker node vLLM launcher |
| `vllm_logs.json` | vLLM logging configuration for idle timeout detection |

### 5. Lockfile Protocol

The `status.json` lockfile is the single source of truth for a job's lifecycle.
It lives on the shared parallel filesystem under
`$PROJECTDIR/engine/jobs/<job>/`.

**6-state machine:**

```
pending → initialising → running → stopped | failed
                ↑                     ↓
                └── cancel (request) ──┘
                └── abort (request) ───┘
```

**Schema:** `status`, `jobName`, `model`, `serverPort`, `user`, `requestedTime`,
`idleTimeout`, `slurmJobId`, `computeHostname`, `startTime`, `stopTime`,
`reason`, `exitCode`. (No per-vLLM-process PID is tracked in the lockfile —
process lifecycle is owned by a single orchestrator process per job that
holds every node's `srun` PID directly.)

**Atomic writes:** lockfiles are created with `set -C` (fail if exists) and
updated via write-to-tmp + `mv` (atomic rename) for safe concurrent access.

**Permissions:** `umask 0002` and `chmod g+w` for group-writable multi-user access.

### 6. Process Orchestration and Monitoring

A single orchestrator process (a background subshell within
`slurm-vllm-serve.sh`, running on the SLURM step host) `srun`-launches vLLM
on the head node and each worker node, holding every node's `srun` client PID
directly. Two monitors run alongside it, both on the step host — there is no
separate per-worker monitor; worker nodes just run vLLM and report memory
usage (`wait_report`) while the orchestrator centrally decides when to shut
down and kills each node's `srun` PID to do so.

| Monitor | Location | Role |
|---------|----------|------|
| `monitor_startup` | Step host, foreground | Blocks until vLLM is healthy (polls `/health` every 10s), saves JIT cache, sends warmup request, transitions to `running`, then detaches. Returns early if the orchestrator process has already exited. |
| `monitor_head` | Step host, background | Runs for entire job lifetime. Checks: lockfile exists, status is `cancel` → clean shutdown, orchestrator process alive, idle timeout (incremental log-parsing, no `/health` false positives). Signals the orchestrator (`SIGUSR2`) on any shutdown condition, which runs the exit-trap (`tidy_up`) that kills every tracked `srun` PID. |

**Exit traps:** All exit paths (SIGUSR1 from SLURM timeout, non-zero exit,
`tidy_up` trap) ensure clean shutdown, diagnostics capture, and lockfile update.

### 7. Shared Multi-User Architecture

- **vLLM setup is per-team, not per-user.** Once any member runs `ivllm setup
  <version>`, all team members share the same venv, NCCL build, and JIT cache.
- **Model downloads go to a shared HuggingFace cache** at `$PROJECTDIR/engine/hf/`.
  One download serves everyone — avoids duplicate disk space and HF API rate
  limits (429 errors).
- **Any project member can `connect` to or `cancel` any running job.** No
  session ownership — lifecycle is owned by the compute node, not the CLI.
- **JIT kernel caches** are saved as `jit-cache.tar.gz` on shared storage, so
  warm startup is fast for any user.

### 8. Example Configs

16 ready-to-use `vllm.yaml` configs in `examples/` covering:

| Model family | Examples |
|--------------|----------|
| Qwen | 2.5 (0.5B→3.6-35B), 3.5 (397B-A17B FP8, 3.5T Long Context), 3 Coder (Next Long Context) |
| DeepSeek | V4 (Flash, Pro) |
| Google | Gemma-4 (31B IT) |
| Other | GPT-OSS-120B, GLM-5.2-743B, MiniMax-M2.5, Nemotron-3-Super-120B |

Each specifies model, tensor parallelism, memory, dtype, and vLLM serving args.

### 9. Model Recipe Generator Skill

An [Agent Skill](https://agentskills.io) shipped in `skills/generate-vllm-config/`
that lets AI coding agents (Cursor, Claude, Windsurf, etc.) generate optimised
`vllm.yaml` configs for any HuggingFace model on Isambard hardware. Installable
via `bunx skills ai4ci/isambard-vllm`.

### 10. Test Suite

| Type | Files | Assertions | Status |
|------|-------|------------|--------|
| Bash unit | `tests/bash/unit/` | Part of 74 | ✅ Green |
| Bash sandboxed | `tests/bash/sandboxed/` (10 files) | Part of 74 | ✅ Green |
| TypeScript unit | `tests/unit/` (3 files) | Part of 62 | ✅ Green |
| TypeScript integration | `tests/integration/` (1 file) | Part of 62 | ✅ Green |
| **Total** | **14 files** | **136** | **0 failures** |

Bash tests use a bubblewrap (`bwrap`) sandbox with real `jq 1.7` and `yq 3.4.1`
binaries and mocked SLURM/vLLM commands (`sbatch`, `srun`, `scancel`, `vllm`, etc.).

### 11. Documentation

| Document | Content |
|----------|---------|
| `design/architecture.md` | Full system architecture: layers, lockfile protocol, monitor triad, multi-backend roadmap |
| `design/adr.md` | 15 Architecture Decision Records (ADR-101 through ADR-115) |
| `design/coding-standards.md` | TypeScript + bash conventions |
| `design/test-scripts.md` | Test coverage and harness details |
| `design/README.md` | Design index and contributor guide |
| Source JSDoc | All public functions documented (44+ bash functions, all TypeScript modules) |
| GitHub Pages | Auto-deployed API docs from `typedoc` on `main` push |

