# Backend Contract — isambard-vllm

This document defines the hard API contract for all `Backend` implementations.
It describes the expected behaviour of every method, the lockfile lifecycle,
and the responsibilities of the backend at each stage.

---

## 1. Lockfile

The lockfile (`status.json`) is the **single source of truth** for a job's
lifecycle. Every backend MUST read from and write to the same shared lockfile
location: `$PROJECTDIR/engine/jobs/<jobName>/status.json`.

All backends share the same lockfile schema (see §1.2). Backend-specific
metadata goes in the optional `backendConfig` field.

### 1.1 Lifecycle States

```
  ┌──────────┐
  │ pending  │  ← CLI creates this when starting a job
  └────┬─────┘
       │ (runtime initialises)
       ▼
  ┌──────────────┐
  │ initialising │  ← runtime writes host, pid, slurmJobId
  └──────┬───────┘
         │ (runtime healthy)
         ▼
  ┌───────────┐
  │ running   │  ← runtime is serving requests
  └─────┬─────┘
        │ (a trigger condition arises)
        ▼
  ┌───────────┐      ┌──────────┐
  │ stopped   │      │  failed  │  ← terminal states
  └───────────┘      └──────────┘
       ▲                   ▲
       │                   │
  ┌────────┐               │
  │ cancel  │  ← user request (not terminal)
  └────────┘               │
```

| State | Meaning | Terminal? | Who transitions TO it |
|-------|---------|-----------|----------------------|
| `pending` | Job created, waiting for runtime to initialise | No | CLI (before `requestStart`) |
| `initialising` | Runtime is starting up (vLLM loading model) | No | Runtime (head node) |
| `running` | Runtime is healthy, serving requests | No | Runtime (after health check) |
| `stopped` | Clean shutdown (user cancel, idle timeout, SLURM timeout) | Yes | Runtime (`tidy_up` exit trap) |
| `failed` | Unclean shutdown (vLLM crash, startup failure) | Yes | Runtime (`tidy_up` exit trap) |
| `cancel` | User-requested shutdown (transition request, not terminal) | No | CLI, or any user with lockfile write access |

**Rules:**

- `pending → initialising → running → (stopped | failed)` is the happy path.
- `cancel` can be written at any time; it is a **request**, not a terminal state.
  The runtime monitor reads `cancel` and transitions to `stopped`.
- `failed` is only written when the runtime crashes (non-zero exit).
- `stopped` is written when the runtime shuts down cleanly (exit code 0 or
  SLURM timeout).

### 1.2 Lockfile Schema

```json
{
  "status": "pending | initialising | running | failed | stopped | cancel",
  "jobName": "qwen36",
  "model": "Qwen/Qwen3.6-35B-A3B-FP8",
  "serverPort": 49153,
  "user": "testuser",
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

| Field | Present | Populated by | Description |
|-------|---------|-------------|-------------|
| `status` | Always | See §1.1 | Current lifecycle state |
| `jobName` | Always | CLI | Unique job name (provided by user) |
| `model` | Always | CLI (from config) | HuggingFace model ID |
| `serverPort` | Always (running) | Runtime | Port vLLM listens on (random high port) |
| `user` | Always | CLI | Username of the job owner |
| `requestedTime` | Always | CLI | ISO-8601 timestamp when job was created |
| `idleTimeout` | Always | CLI | Minutes of inactivity before auto-shutdown (`-1` = never) |
| `slurmJobId` | Sometimes | Runtime | HPC scheduler job ID (backend-specific) |
| `computeHostname` | Sometimes | Runtime | Hostname of the compute node (backend-specific) |
| `startTime` | Sometimes | Runtime | ISO-8601 timestamp when runtime became healthy |
| `stopTime` | Sometimes | Runtime | ISO-8601 timestamp when runtime shut down |
| `vllmPid` | Sometimes | Runtime | PID of the vLLM process |
| `reason` | Sometimes | Runtime | Human-readable reason for stopped/failed |
| `exitCode` | Sometimes | Runtime | Exit code of the vLLM process |

**Notes:**

- `serverPort`, `computeHostname`, `startTime`, `stopTime`, `slurmJobId`,
  `vllmPid`, `reason`, `exitCode` are **populated by the runtime**, not the CLI.
  They may be absent in `pending` state.
- Backends MUST write the lockfile atomically (write to temp file + rename).
- Lockfiles MUST be group-writable (`umask 0002`, `chmod g+w`).

### 1.3 Lockfile Lifecycle Rules

| Action | Status change | Who | How |
|--------|--------------|-----|-----|
| Start job | (none) → `pending` | CLI | Create lockfile via backend `requestStart()` |
| Runtime initialises | `pending` → `initialising` | Runtime | Head node writes runtime metadata |
| Runtime healthy | `initialising` → `running` | Runtime | After health check, writes serverPort, hostname |
| User cancels | (any) → `cancel` | CLI or any user | Write cancel to lockfile (request, not terminal) |
| Clean shutdown | `cancel` → `stopped` | Runtime | Exit trap writes stopped + reason |
| Idle timeout | `running` → `stopped` | Runtime | Monitor detects no API requests |
| SLURM timeout | (any) → `stopped` | Runtime | Scheduler signal triggers exit trap |
| vLLM crash | `initialising`/`running` → `failed` | Runtime | Non-zero exit triggers exit trap |
| Force cancel | (any) → `stopped` | CLI | Kill runtime + write stopped |

---

## 2. Backend Interface

Every `Backend` implementation MUST satisfy this contract. The CLI delegates
all job lifecycle operations to the backend.

### 2.1 `bootstrap(): Promise<void>`

**Purpose:** Verify connectivity and deploy any files the backend needs on the
HPC. Called once per backend instance, before any other method.

**Requirements:**
- MUST verify SSH/network connectivity to the target environment.
- MUST deploy engine files (bash scripts, config) to the remote HPC.
- MUST be idempotent — safe to call multiple times.
- MUST throw an error if connectivity is unavailable.

**Typical implementation:** Check SSH, rsync `engine/` directory to HPC.

```typescript
await backend.bootstrap();
```

### 2.2 `setup(version: string, force?: boolean): Promise<void>`

**Purpose:** Install or update the runtime (vLLM, containers, etc.) on the HPC.
This is a **one-time per-version** operation shared across all users.

**Requirements:**
- MUST submit a build/install job on a compute node (or equivalent).
- MUST install into a versioned directory: `$PROJECTDIR/engine/<version>/`.
- MUST be idempotent — skip if the version is already installed (unless `force`).
- MUST stream progress output to the terminal.
- `force: true` MUST reinstall even if the version exists.

```typescript
await backend.setup('0.22.0');       // skip if installed
await backend.setup('0.22.0', true); // force reinstall
```

### 2.3 `requestStart(job: string, maxTime: string, monitor: boolean, config?: string): Promise<void>`

**Purpose:** Start a new job or restart a stopped/failed one. Creates the
`pending` lockfile, submits the runtime, and returns when the job is queued.

**Parameters:**

| Param | Description |
|-------|-------------|
| `job` | Job name (unique within the project) |
| `maxTime` | Maximum runtime duration (e.g. `'08:00:00'`) |
| `monitor` | Whether to enable idle timeout monitoring |
| `config` | Path to a local vLLM config YAML (uploaded to the job dir) |

**Requirements:**

1. MUST create the job directory: `$PROJECTDIR/engine/jobs/<job>/`
2. MUST write a `pending` lockfile atomically.
3. MUST upload the config file (if provided) to the job directory.
4. MUST submit the runtime via the HPC scheduler (or equivalent).
5. MUST populate `slurmJobId`, `computeHostname`, and other runtime metadata
   from the scheduler output.
6. MUST return immediately after submission (does NOT wait for the runtime to
   become healthy).
7. MUST NOT overwrite an existing lockfile in `running` or `initialising` state.

**Lockfile transitions this method triggers:** `(none) → pending`

```typescript
await backend.requestStart('qwen36', '08:00:00', true, './vllm.yaml');
```

### 2.4 `connect(job: string, localPort: number): Promise<CloseableEventEmitter>`

**Purpose:** Establish an SSH tunnel from `localhost:<localPort>` to the
running vLLM instance. Used to expose the OpenAI-compatible API locally.

**Requirements:**

1. MUST read the lockfile and verify the job is in `running` state.
2. MUST check that `computeHostname` and `serverPort` are present.
3. MUST throw an error if the local port is already in use.
4. MUST establish a persistent SSH port-forward tunnel.
5. MUST return a `CloseableEventEmitter` that represents the tunnel process.
6. MUST keep the tunnel alive until `.kill()` is called.
7. MUST be idempotent — safe to call multiple times (returns existing tunnel
   or creates a new one).

**Return value:** A `CloseableEventEmitter` (extends Node's `EventEmitter`) with:
- `.kill(signal?)` — terminate the tunnel process
- `'close'` event emitted when the process exits
- `'error'` event emitted on fatal errors

```typescript
const tunnel = await backend.connect('qwen36', 11434);
// Local machine: curl http://localhost:11434/v1/chat/completions
// Tunnel forwards to: <computeNode>:<serverPort>
tunnel.kill(); // close the tunnel
```

### 2.5 `requestCancel(job: string, force: boolean): Promise<void>`

**Purpose:** Request graceful or forced shutdown of a running job.

**Parameters:**

| Param | Description |
|-------|-------------|
| `job` | Job name |
| `force` | If `true`, kill immediately via scheduler; if `false`, request graceful shutdown |

**Requirements for graceful cancel (`force: false`):**
1. MUST read the lockfile and verify the job is in a cancellable state
   (`pending`, `initialising`, `running`).
2. MUST write `cancel` to the lockfile.
3. MUST return immediately — the runtime monitor will detect the cancel
   and transition to `stopped`.

**Requirements for force cancel (`force: true`):**
1. MUST read the lockfile to get the scheduler job ID.
2. MUST kill the scheduler job directly (e.g. `scancel`).
3. MUST write `stopped` to the lockfile.

**Lockfile transitions:**
| Mode | Transition | Who acts |
|------|-----------|----------|
| Graceful | (any) → `cancel` → `stopped` | Runtime monitor detects `cancel`, exits cleanly |
| Force | (any) → `stopped` | CLI kills scheduler, writes lockfile |

```typescript
await backend.requestCancel('qwen36', false);  // graceful
await backend.requestCancel('qwen36', true);   // force kill
```

### 2.6 `getAllJobStatus(): Promise<LockfileV3[]>`

**Purpose:** List all jobs for this backend. Returns the lockfile contents
for every job found.

**Requirements:**
1. MUST scan the jobs directory (or equivalent).
2. MUST parse and return every valid lockfile.
3. MUST return an empty array if no jobs exist.
4. MUST NOT throw on malformed lockfiles — skip them silently.

```typescript
const jobs = await backend.getAllJobStatus();
// jobs: LockfileV3[] — one per job, including stopped and failed
```

### 2.7 `watchLog(job: string, node?: string, until?: string): Promise<CloseableEventEmitter>`

**Purpose:** Stream log output for a job, optionally stopping when a
pattern is matched. Used for monitoring startup progress.

**Parameters:**

| Param | Description |
|-------|-------------|
| `job` | Job name |
| `node` | Node identifier — `'0'` for head node, `'1'`+ for workers (optional, defaults to head) |
| `until` | A string pattern that, when matched in the log output, causes the stream to close (optional) |

**Requirements:**

1. MUST tail the job's log file(s) from the current end position.
2. MUST stream log lines to stdout in real time.
3. MUST stop streaming once `until` pattern is matched (if provided).
4. MUST return a `CloseableEventEmitter` for the stream.
5. MUST handle the case where the log file does not yet exist.

```typescript
const stream = await backend.watchLog('qwen36', '0', '[startup] Startup complete');
// Streams log output until "[startup] Startup complete" is seen
stream.kill(); // force close
```

---

## 3. State-Helper Methods

These are convenience methods provided by the abstract `Backend` base class.
They query the lockfile and return boolean results for common state checks.
Backends MUST NOT override these.

| Method | Returns `true` when |
|--------|-------------------|
| `isRunning(job)` | Lockfile status is `running` |
| `isStopped(job)` | Lockfile status is `stopped` or `failed` (or lockfile missing) |
| `isStartable(job)` | Lockfile status is `stopped` or `failed` (or lockfile missing — never started) |
| `isStarting(job)` | Lockfile status is `pending` or `initialising` |
| `getJobStatus(job)` | Returns the full `LockfileV3` object (throws if not found) |

**Note on missing lockfiles:** `isStopped()` and `isStartable()` return `true`
when no lockfile exists, because a missing lockfile means the job was never
started and is therefore in a stopped/startable state.

---

## 4. Backend Responsibilities Summary

Each method has a clear responsibility boundary. The table below summarises
what the backend does (and does NOT do) for each lifecycle operation.

| Method | Writes lockfile? | Starts runtime? | Waits for healthy? | Returns tunnel? |
|--------|-----------------|-----------------|-------------------|----------------|
| `bootstrap()` | No | No | No | No |
| `setup()` | No | No (installs, not runs) | No | No |
| `requestStart()` | Yes → `pending` | Yes (submits) | No | No |
| `connect()` | No | No | No (assumes running) | Yes |
| `requestCancel()` | Yes (writes `cancel` or `stopped`) | No | No | No |
| `getAllJobStatus()` | No | No | No | No |
| `watchLog()` | No | No | No | No |

---

## 5. CLI Usage Pattern

The CLI uses the backend in this sequence:

```
ivllm connect <job>
  ├── getBackend(config)
  ├── backend.isRunning(job)
  │   ├── getAllJobStatus()
  │   └── parse status.json
  │
  ├── IF NOT running:
  │   ├── backend.isStartable(job)
  │   │   └── parse status.json (stopped/failed/missing)
  │   ├── backend.isStarting(job)
  │   │   └── parse status.json (pending/initialising)
  │   ├── IF startable:
  │   │   └── backend.requestStart(job, time, monitor, config)
  │   └── IF starting:
  │       └── backend.watchLog(job, '0', '[startup] Startup complete')
  │
  └── IF running:
      └── backend.connect(job, localPort) → CloseableEventEmitter
```

This pattern ensures idempotent behaviour: calling `connect` repeatedly when
the job is already running simply re-establishes the tunnel.

---

## 6. Lockfile Write Guarantees

All lockfile writes MUST follow these rules:

1. **Atomic writes:** Write to a temp file, then `rename()` to the final path.
   This prevents partial reads during writes.
2. **Create-only for initial lockfile:** The first lockfile (`pending`) MUST
   use `set -C` (bash) or equivalent — fail if the file already exists.
3. **Group-writable:** All lockfiles MUST be created with `umask 0002` and
   `chmod g+w` for multi-user access.
4. **Schema validation:** Parsed lockfiles MUST have `status` and `jobName`
   as string fields. Malformed lockfiles are silently skipped.

---

## 7. Backend Implementation Checklist

When implementing a new backend, verify:

- [ ] `bootstrap()` verifies connectivity and deploys files
- [ ] `setup()` installs runtime into versioned directory, skips if present
- [ ] `requestStart()` creates lockfile, submits runtime, returns immediately
- [ ] `connect()` reads lockfile, validates `running` state, opens SSH tunnel
- [ ] `requestCancel()` writes `cancel` (graceful) or kills scheduler (force)
- [ ] `getAllJobStatus()` scans jobs directory, returns parsed lockfiles
- [ ] `watchLog()` tails logs, supports `until` pattern matching
- [ ] All lockfile writes are atomic and group-writable
- [ ] Lockfile schema matches §1.2 (required fields present, optional fields
      populated at correct lifecycle points)
- [ ] State-helper methods (`isRunning`, etc.) work correctly including
      missing-lockfile edge cases
- [ ] `CloseableEventEmitter` implementations support `.kill()`, `'close'`
      event, `'error'` event
