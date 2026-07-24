# Roadmap — isambard-vllm (ivllm) v3

This document captures the current scope of the v3 rewrite and maps out
future directions derived from the architecture document (`architecture.md`)
and the Architecture Decision Records (`adr.md`).

---

## Current Scope

The v3 rewrite from v2 is **functionally complete**. All core commands, the
backend abstraction, the bash lifecycle framework, and the test harness are
implemented and tested.

### 1. CLI Commands

Five commands, all registered via Commander.js in `src/index.ts`.

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

#### `ivllm cancel <job> [--force]`

| Mode | Behaviour |
|------|-----------|
| **Graceful** (default) | Writes `cancel` to the lockfile; the compute-side monitor detects it and shuts down vLLM cleanly |
| **Force** (`--force`) | Runs `scancel` directly on the SLURM job, then updates the lockfile |

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

### 2. Backend Abstraction

An abstract `Backend` class in `src/backends/Backend.ts` (227 lines) defines the
lifecycle contract. A single concrete implementation — `IsambardBareMetalBackend` —
implements it for the Isambard AI HPC.

| Abstract method | Purpose |
|-----------------|---------|
| `bootstrap()` | Verify SSH, deploy engine scripts to HPC |
| `setup(version, force)` | Install vLLM on the HPC |
| `connect(job, port)` | Establish SSH tunnel to a running job |
| `requestCancel(job, force)` | Graceful or forced cancellation |
| `requestStart(job, maxTime, monitor, config)` | Submit a new SLURM job |
| `getAllJobStatus()` | List all jobs as lockfile objects |
| `watchLog(job, node, until)` | Tail log output |
| `getJobStatus(job)` | Get status of a single job |
| `isRunning / isStopped / isStartable / isStarting` | Lifecycle state helpers |

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
  `<computeHost>:<serverPort>` with keepalive and `ExitOnForwardFailure`
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
```

**Schema:** `status`, `jobName`, `model`, `serverPort`, `user`, `requestedTime`,
`idleTimeout`, `slurmJobId`, `computeHostname`, `startTime`, `stopTime`,
`vllmPid`, `reason`, `exitCode`.

**Atomic writes:** lockfiles are created with `set -C` (fail if exists) and
updated via write-to-tmp + `mv` (atomic rename) for safe concurrent access.

**Permissions:** `umask 0002` and `chmod g+w` for group-writable multi-user access.

### 6. Monitor Triad

Three monitoring processes run on the compute allocation:

| Monitor | Location | Role |
|---------|----------|------|
| `monitor_startup` | Head node, foreground | Blocks until vLLM is healthy (polls `/health` every 10s), saves JIT cache, sends warmup request, transitions to `running`, then detaches |
| `monitor_head` | Head node, background | Runs for entire job lifetime. Checks: lockfile exists, status is `cancel` → clean shutdown, vLLM process alive, idle timeout (incremental log-parsing, no `/health` false positives) |
| `monitor_worker` | Worker nodes, background | Polls lockfile; if status changes or lockfile disappears → SIGTERM local vLLM process. Reports memory/JIT cache usage during startup |

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

### Remaining (within current scope)

These items tighten quality and robustness before broader features are
added.

- [ ] **CLI handler unit tests** — `cmdConnect`, `cmdSetup`, `cmdConfig`,
      `cmdStatus`, `cmdCancel` in `src/index.ts` have no unit tests. Only
      the Backend contract is covered via integration tests.
- [ ] **TypeScript type check** — No `tsc --noEmit` or `--strict` check in
      the test pipeline. Should be a fast gate before `bun test`.
- [ ] **ESLint / linting** — No linting configured in the test workflow.
- [ ] **Monitor idle timeout unit test** — `monitor_head` log-parsing idle
      check works end-to-end but lacks a focused unit test for the
      time-pattern matching logic.
- [ ] **Version bump to v3** — Package.json still shows `v2.14.0`;
      the binary is named `ivllm2`. Needs a coherent version strategy.

---

## Future Directions

The v3 architecture is designed to support these evolutions without structural
changes. Each maps to one or more ADRs.

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


