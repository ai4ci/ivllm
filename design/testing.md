# Testing Plan — isambard-vllm v3

## Current state

This document originally described a state where all existing tests were
**template-centric** (verifying generated bash scripts contained specific
strings) rather than testing real behaviour. That gap has now been
substantially closed for the bash layer: bash tests run inside a bubblewrap
sandbox against real subprocess/signal semantics and the real `jq`/`yq`
binaries — see "Bubblewrap (bwrap) sandbox architecture" below. Building
that harness and writing the first real tests against it immediately found
several genuine bugs (see `design/issues.md`, issues 7–13), which is the
intended outcome of testing tactically rather than just for surface area.

The TypeScript layers (Layer 1 unit tests, the `mock` RemoteOps mode, Layer
3 integration, Layer 4 E2E) described below have **not** been reviewed as
part of this bash-testing pass and may still reflect the original
template-centric/aspirational state — treat that content as a plan, not a
status report, until it gets the same treatment.

---

## Target: Three-layer test architecture

```
┌──────────────────────┐
│   Unit tests         │  Fast, isolated. Test one function/module.
│   (TypeScript + Bash)│  Bash unit tests (unit/) run directly; anything
│                      │  touching external commands runs sandboxed
│                      │  (sandboxed/, via bubblewrap) instead — see below.
├──────────────────────┤
│   Integration tests  │  Test interactions between layers without
│   (Mock E2E)         │  real HPC. Full lifecycle with mock vLLM.
├──────────────────────┤
│   E2E tests          │  Real Isambard runs. Slow, expensive, run
│   (Isambard)         │  before releases only.
└──────────────────────┘
```

### Where tests live

| Layer | Location | Runner | Speed |
|-------|----------|--------|-------|
| TypeScript unit | `tests/unit/*.test.ts` | `bun test` | ms |
| Bash unit (plain) | `tests/bash/unit/test-*.sh` | `bash tests/bash/run.sh unit` | ms |
| Bash unit (sandboxed) | `tests/bash/sandboxed/test-*.sh` | `bash tests/bash/run.sh sandboxed` | s (bwrap overhead per test) |
| Integration | `tests/integration/*.test.ts` | `bun test` | s |
| E2E | `tests/e2e/smoke.sh` | Manual / CI on Isambard | 5–30 min |

---

## Layer 1: TypeScript Unit Tests

### Current test files (post-v3 rewrite)

All old `tests/cli/` and `tests/integration/` files were removed in favour of
new tests targeting the v3 architecture (`Backend`, `RemoteOps`, `local-ops`).

| Test file | What it tests | Status |
|-----------|---------------|--------|
| `unit/Backend.test.ts` | `parseV3Lockfile()`, `getJobStatus()`, `isRunning()`, `isStopped()`, `isStartable()`, `isStarting()` on the abstract Backend base class | ✅ 20 tests |
| `unit/RemoteOps.mock.test.ts` | MockRemoteOps — records and validates what each RemoteOps method would send without any SSH | ✅ 8 tests |
| `unit/local-ops.test.ts` | `isLocalPortInUse()`, `isHealthy()`, `queryModels()` using real http.Server | ✅ 7 tests |
| `unit/semver.test.ts` | `semverGte()`, `revSemverSort()` — semantic version comparison | ✅ 11 tests |
| `integration/CLI.lifecycle.test.ts` | Full Backend lifecycle (requestStart, requestCancel, getAllJobStatus, bootstrap, lifecycle helpers) via a MockRemoteOps | ✅ 11 tests |

**Total: 57 TypeScript tests** across 5 files.

### Test architecture

The v3 TypeScript tests use a three-layer approach:

| Layer | Purpose | Implementation |
|-------|---------|----------------|
| **Pure unit** | Test functions that take/return only data | `Backend.test.ts` (JSON parsing), `semver.test.ts`, `local-ops.test.ts` (http.Server) |
| **Interface mock** | Test RemoteOps contract without SSH | `RemoteOps.mock.test.ts` — MockRemoteOps records every call |
| **Integration** | Test the full Backend→RemoteOps chain | `CLI.lifecycle.test.ts` — TestBackend with TestRemoteOps |

### MockRemoteOps

The `RemoteOps` interface is implemented by `MockRemoteOps` in tests — a
concrete class that records every call and returns canned responses. This
replaces the old `makeRemoteOps('dry-run')` boolean approach with something
that has full interface coverage.

MockRemoteOps simulates:
- `sbatch` → returns `'Submitted batch job 123456'`
- `squeue` → returns `'123456 RUNNING'`
- `checkSSH` → returns `true` without any socket activity
- `spawnTunnel` → returns a closeable mock EventEmitter
- `copyFile`/`copyDirectory` → record the call for assertion

---

## Mock Remote Ops (TypeScript)

The `makeRemoteOps` factory in `src/remote-ops.ts` currently has two modes:
`real` (SSH) and `dry-run` (canned text responses). For v3 connect/cancel
testing we need a **third mode** that simulates a real filesystem on the
remote side without requiring actual SSH.

### Current problems with `dry-run` mode

| Issue | Example | Impact |
|-------|---------|--------|
| Canned `cat` response returns `'lockfile'` (not JSON) | `command.startsWith('cat') → 'lockfile'` | Can't test lockfile parsing |
| No writable filesystem | `set -C; cat > status.json <<...` returns empty string | Can't test lockfile creation |
| No state tracking | Every call is independent | Can't simulate pending→running transitions |
| `dryRun` is boolean | No middle ground | Can't test with real filesystem but fake SSH |

### Solution: add a `mock` mode

```typescript
type OpsMode = 'real' | 'mock' | 'dry-run';

function makeRemoteOps(config: Credentials, mode: OpsMode): RemoteOps;
```

In `mock` mode:

| Method | Behaviour |
|--------|-----------|
| `runRemote` | Executes commands against a local temp filesystem. `cat`, `mkdir`, `jq` operations work on real files. `sbatch` returns a fake job ID. `scancel` is a no-op. |
| `copyFile` | Copies to a local temp directory (same as dry-run) |
| `spawnTunnel` | Returns a mock emitter (same as dry-run) |
| `checkSSH` | Returns true (same as dry-run) |
| `streamSrun` | Logs and returns mock emitter (same as dry-run) |

For lockfile simulation, the mock `runRemote` detects `cat` and `jq` patterns
and delegates to local bash. A `MockLockfile` helper tracks simulated state
and can inject failures:

```typescript
class MockRemoteFs {
  private baseDir: string;

  /**
   * Simulate a SLURM job progressing from pending → initialising → running.
   * After `delayMs`, runs `jq '.status = "running"'` on the lockfile.
   */
  simulateJobStart(jobName: string, delayMs: number): void;

  /**
   * Read back the current lockfile for assertions.
   */
  readLockfile(jobName: string): LockfileV3 | null;
}
```

Test flow for `ivllm connect`:

```typescript
test('connect creates lockfile, submits sbatch, transitions to running', async () => {
  const fs = new MockRemoteFs();
  const ops = makeMockRemoteOps(fs.baseDir);

  // Inject a SLURM job that becomes "running" after 2s
  fs.simulateJobStart('test-job', 2000);

  await cmdConnect(['test-job', '--config', testConfig, '--local-port', '9999'], ops);

  // Verify lockfile was created
  const lockfile = fs.readLockfile('test-job');
  expect(lockfile?.status).toBe('running');
});
```

### Migration path

The `mock` mode is now implemented via dedicated `MockRemoteOps` and
`TestBackend` classes in the test suite rather than a runtime mode switch
in `makeRemoteOps`. This keeps production code simpler (no `OpsMode` enum
needed) and gives tests full control over the RemoteOps contract.

The bash framework must be tested without an HPC connection, but the old
"mock harness" approach (overriding `srun`/`sbatch`/`scancel`/`vllm` as bash
**functions** sourced into the same shell as the code under test) had real
gaps: it never exercised actual subprocess/exec semantics, background
processes could leak onto the host if a test crashed before its trap fired,
and there was no isolation from whatever HPC tools happened to be on the
test runner's own PATH.

### Bubblewrap (bwrap) sandbox architecture

Bash tests now run inside a [bubblewrap](https://github.com/containers/bubblewrap)
sandbox (`tests/bash/lib/sandbox.sh`), which:

- Intercepts external commands via **PATH shims** (`tests/bash/shims/*`) —
  each a standalone executable, not a sourced bash function, so real
  argument-passing and exit-code semantics get exercised.
- Binds in **real system tools we are not trying to mock** (`bash`, `jq`,
  `yq`, `tar`, `awk`, `python3`, `curl`, coreutils, ...) read-only from the
  host, so tests run against actual tool behaviour rather than an idealised
  stand-in. This is deliberate and already paid off: it is how issues 7–9
  and 13 in `design/issues.md` were found — `utils.sh` was written assuming
  `yq` v4's jq-style filter syntax, but the HPC (and this sandbox) has the
  real `yq 3.4.1`, which has a completely different CLI.
- Uses `--unshare-net` to block all real network access (loopback still
  works, so a mock vLLM HTTP server on `localhost` is reachable) — it is
  physically impossible for a test to accidentally hit a real HPC,
  HuggingFace, or NVIDIA download URL.
- Uses `--unshare-pid` + `--die-with-parent`, giving each test its own PID
  namespace with bwrap itself as the reaper (PID 1). When the sandboxed
  script exits, **every** process it spawned — monitors, mock vLLM servers,
  backgrounded mock `srun` children — is killed automatically, even if the
  test crashed. No leaked background processes, no trap-based cleanup
  required in every test.
- Provides two **profiles**, matching where a script actually runs:

  | Profile | Env | Used for |
  |---------|-----|----------|
  | `login` | No `SLURM_*` vars | Top-level `ivllm-*.sh` wrapper scripts that run on the LOGIN node and submit work (`sbatch`/`srun`) to the scheduler |
  | `compute` | `SLURM_JOB_ID`/`SLURM_NODEID`/`SLURM_NNODES`/`SLURM_JOB_NODELIST`/`COMPUTE_HOSTNAME` set | Code that runs *inside* a SLURM allocation: lockfile state transitions, the monitor triad, exit traps/signals, `run_head_vllm.sh`/`run_worker_vllm.sh` |

See the full design rationale and implementation notes in the header
comment of `tests/bash/lib/sandbox.sh`.

### Two-tier test layout

```
tests/bash/
├── run.sh              — runs unit/ then sandboxed/ (or a single scope)
├── lib/
│   ├── sandbox.sh        — bwrap harness: sandbox_run(), sandbox_run_test()
│   ├── assertions.sh      — assert_* helpers (file/json/status/shim-called)
│   └── test-utils.sh      — setup_test_env/cleanup_test_env for unit/ tests
├── shims/                — PATH-shim executables (see table below)
├── fixtures/              — sample vllm.yaml configs (minimal/with-env/multi-node)
├── unit/                  — fast, non-sandboxed: pure bash logic only
└── sandboxed/             — bwrap-sandboxed: real yq/jq, mocked SLURM/vLLM,
                             process/signal isolation
```

`unit/` is for logic that touches no external command needing a mock (e.g.
semver comparison) — it runs directly on the host for fast iteration.
Anything that shells out to `jq`/`yq`/`srun`/`sbatch`/`scancel`/`vllm`, or
needs real process/signal behaviour (the monitor triad, exit traps), belongs
in `sandboxed/`.

### Shim inventory

| Shim | Replaces | Behaviour |
|------|----------|-----------|
| `sbatch` | `sbatch` | Logs the call, returns a fake job id (`MOCK_SBATCH_JOB_ID`). Does not execute the submitted script — see "Fidelity tiers" below. |
| `srun` | `srun` | Two modes via `MOCK_SRUN_MODE`: `log-only` (default; records the call, returns `MOCK_SRUN_EXIT`) or `exec` (actually backgrounds the wrapped command and forwards TERM/INT/USR1/USR2 to it) |
| `scancel` | `scancel` | Logs the call, returns `MOCK_SCANCEL_EXIT` |
| `squeue` | `squeue` | Returns a job id if it's listed in `MOCK_SQUEUE_ACTIVE_JOBS` |
| `scontrol` | `scontrol show hostnames` | Expands a comma/whitespace-separated nodelist |
| `dig` | `dig +short` | Returns `MOCK_DIG_IP` (default `10.0.0.1`) |
| `module` | `module load/purge` | No-op (logged) — NVHPC_ROOT/CC/CXX are resolved independently, not via module side effects |
| `gcc` / `g++` | compilers | Exist and are resolvable via `which`, so `CC`/`CXX` are non-empty |
| `hf` | `hf cache ls` / `hf download` | Configurable via `MOCK_HF_CACHED_MODELS` / `MOCK_HF_DOWNLOAD_EXIT` |
| `uv` | `uv venv` / `uv pip *` | Creates a fake venv (`bin/activate`, `bin/pip`); configurable pip list/show/install results |
| `vllm` | `vllm serve` | Real Python HTTP server: `/health`, `/v1/models`, `/v1/chat/completions`; configurable startup delay (`MOCK_VLLM_DELAY`), crash-after-N-requests (`MOCK_VLLM_CRASH_AFTER`); honours SIGUSR1/SIGUSR2 (exit 200/201, matching `tidy_up`'s contract); access-log lines are timestamped to match `IVLLM_TIME_FMT` for idle-timeout tests |
| `wget`, `git` | real downloads/clones | Hard-fail with a clear message — real network installs (NVHPC SDK, DeepGEMM/DeepEP) are out of scope for the bash test suite (see "Fidelity tiers") |

`curl` is deliberately **not** shimmed — the sandbox's `--unshare-net`
already makes real external URLs unreachable, while `http://localhost/...`
(used for real vLLM health checks against the `vllm` shim's mock server)
still works via loopback. This gives faithful behaviour for free.

### Fidelity tiers (why `sbatch`/`srun` don't execute the real job)

Testing the full chain "CLI submits sbatch → SLURM eventually runs the job
on a compute node" end-to-end would require nested sandboxes with different
env profiles (login vs compute) for the *same* invocation, which isn't
practical. Instead there are two deliberately separate tiers:

1. **Login-side handoff tests** (`login` profile): verify that
   `ivllm-serve.sh`/`ivllm-setup.sh`/`ivllm-cancel.sh`/etc. build the correct
   `sbatch`/`srun` invocation and handle its result (job id, exit code)
   correctly. The `sbatch`/`srun` shims just record the call — they do not
   execute the wrapped script.
2. **Compute-side execution tests** (`compute` profile): invoke the
   compute-side scripts/functions directly (`slurm-vllm-serve.sh`,
   `run_head_vllm.sh`, `monitor_head`, ...) as if already inside a SLURM
   allocation. Here `MOCK_SRUN_MODE=exec` makes the `srun` shim actually
   background the wrapped command, matching what `srun` does when it's
   simply launching a task step on already-allocated resources.

Real network installation (`slurm-vllm-setup.sh` downloading the NVHPC SDK,
compiling DeepGEMM/DeepEP from git) is out of scope for the bash test suite
— the `wget`/`git` shims fail loudly if anything tries to reach them, so an
unexpected call surfaces as a clear test failure rather than a silent no-op.

### Current sandboxed test files

| File | Covers | Status |
|------|--------|--------|
| `sandboxed/test-lockfile.sh` | Lockfile state machine (`create_status_pending`, `update_status_*`, `request_cancel`, `is_status`) | All passing |
| `sandboxed/test-cache.sh` | JIT cache save/restore, permissions, node gating | All passing |
| `sandboxed/test-vllm-env.sh` | `common-env.sh`/`vllm-env.sh` preamble sourcing and env vars | Mostly passing; one intentional red test for Issue 12 |
| `sandboxed/test-config.sh` | `get_job_config_setting`/`resolve_stripped_job_config`/`get_job_config_exports` against the real `yq 3.4.1` binary | Mostly **red** — demonstrates issues 7–9 |
| `sandboxed/test-monitor-head.sh` | `monitor_head()`: cancel detection, vLLM process death, lockfile deletion, idle timeout, active-traffic | Mostly **red** — demonstrates issue 13 |

Red tests here are the correct TDD starting state (AGENTS.md: "all tests
must be proven to fail before starting development of a feature") — they
were written first, found real bugs, and are left failing until those bugs
are fixed as a dedicated follow-up.

### Still to be written

The following scenarios from the original test-scenario tables below remain
to be implemented as sandboxed tests (contributions should follow the same
`sandbox_run_test` pattern):

- `monitor_startup` — health-check polling, warmup request, cache save
  timing, startup failure paths
- `monitor_worker` — worker-node lockfile watching, shutdown on
  cancel/failed/stopped, head-node misuse guard
- Exit trap / signals (`tidy_up`) — SIGUSR1 (SLURM timeout), SIGUSR2
  (cancel/idle), crash-during-startup vs crash-during-runtime, scancel
  ordering
- Login-node handoff tests for `ivllm-serve.sh`/`ivllm-cancel.sh`/
  `ivllm-status.sh`/`ivllm-setup.sh` (asserting on `calls.log` via
  `assert_shim_called`)
- Multi-node: `scontrol`/`dig` head/worker node resolution in
  `slurm-vllm-serve.sh`

The scenario tables below (by bash function) describe the intended
coverage; they predate the bwrap rewrite and use the old `test_*` naming
convention, but the scenarios themselves are still the right target list.


---

## Layer 3: Integration Tests (Mock E2E)

Integration tests verify the **handoff** between layers — the full lifecycle
with mock backend, real TypeScript Backend methods, and the Bash framework
sourced in bwrap sandboxes.

### Current integration tests

| Test file | Covers | Runner |
|-----------|--------|--------|
| `integration/CLI.lifecycle.test.ts` | Full Backend lifecycle through mock RemoteOps: requestStart→sbatch, requestCancel→ivllm-cancel.sh, getAllJobStatus→lockfile list, lifecycle helpers | `bun test` |

### Handoff Interface

The critical interface between TypeScript (CLI) and Bash (compute) is the
lockfile. Both sides write to it. The following handoff scenarios must all
be tested:

```
CLI creates status.json → bash reads it
bash updates status      → CLI reads it (via SSH polling)
CLI writes cancel        → bash detects it
CLI establishes tunnel   → traffic flows through it
CLI disconnects          → bash keeps running (no effect)
CLI reconnects           → bash shows same state
```

These are now tested via the bwrap sandbox (bash side) and MockRemoteOps
(TypesScript side). The handoff boundary is verified end-to-end by the
integration tests that exercise the full Backend→RemoteOps chain.

---

## Layer 4: End-to-End Tests (Isambard)

E2E tests run on real Isambard hardware. They are slow (minutes), expensive
(SUs), and should be run only before releases or after major changes.

### Smoke test

```bash
#!/bin/bash
# tests/e2e/smoke.sh
# Run: bash tests/e2e/smoke.sh <version>

set -euo pipefail
VERSION=${1:-"0.22.0"}

echo "=== ivllm E2E Smoke Test ==="

# 1. Setup vLLM (idempotent)
echo "--- Phase 1: ivllm setup ---"
ivllm setup "$VERSION"

# 2. Create test config
cat > /tmp/e2e-test.yaml <<EOF
model: Qwen/Qwen2.5-0.5B-Instruct
tensor-parallel-size: 1
max-model-len: 8192
gpu-memory-utilization: 0.85
dtype: bfloat16
EOF

# 3. Connect with small model
echo "--- Phase 2: ivllm connect ---"
ivllm connect e2e-test --config /tmp/e2e-test.yaml --detach

# 4. Verify API endpoint
sleep 60  # Wait for startup
echo "--- Phase 3: API test ---"
curl -s http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen2.5-0.5B-Instruct","messages":[{"role":"user","content":"Hello"}]}'

# 5. Cancel job
echo "--- Phase 4: ivllm cancel ---"
ivllm cancel e2e-test

# 6. Verify cleanup
sleep 10
if ivllm list | grep e2e-test | grep -q running; then
  echo "ERROR: Job still running after cancel"
  ivllm cancel e2e-test --force
  exit 1
fi

echo "=== E2E smoke test PASSED ==="
```

### E2E test scenarios (manual)

| Scenario | Frequency | Approx time | Notes |
|----------|-----------|-------------|-------|
| Single-node: Qwen2.5-0.5B-Instruct | Every release | 5 min | Fastest model |
| Single-node: Qwen2.5-7B-Instruct | Every release | 8 min | Representative |
| Detach+reattach | Every release | 10 min | Kill CLI, reconnect |
| Multi-user: two users connect simultaneously | Per major release | 15 min | Requires 2 accounts |
| Multi-node: 2-node config | Per major release | 20 min | Expensive, verify ray |
| Idle timeout | Per major release | 35 min | Must wait for timeout |
| Force cancel during startup | Per major release | 3 min | Edge case |
| Model download from scratch | On setup change | 10 min | Clear HF cache first |

---

## Testing timeline by phase

### Phase M1: Bash Framework

Focus: Unit tests for every bash function.

| Test area | Count | Runner |
|-----------|-------|--------|
| Lockfile state machine | 20 | `bash tests/bash/sandboxed/test-lockfile.sh` |
| JIT cache save/restore | 5 | `bash tests/bash/sandboxed/test-cache.sh` |
| Monitor startup (health, retry) | 4 | `bash tests/bash/sandboxed/test-monitor-startup.sh` |
| Monitor head (cancel, death, timeout) | 5 | `bash tests/bash/sandboxed/test-monitor-head.sh` |
| Monitor worker (head rejection, cancel) | 5 | `bash tests/bash/sandboxed/test-monitor-worker.sh` |
| Exit trap / signals (`tidy_up`) | 7 | `bash tests/bash/sandboxed/test-exit-trap.sh` |
| Login-node handoff (ivllm-*.sh) | 6 | `bash tests/bash/sandboxed/test-login-handoff.sh` |
| Environment preamble (`common-env.sh`) | 4 | `bash tests/bash/sandboxed/test-vllm-env.sh` |
| Config parsing (`yq 3.4.1`) | 6 | `bash tests/bash/sandboxed/test-config.sh` |
| Pure bash logic (semver, utils) | 12 | `bash tests/bash/unit/test-utils.sh` |
| **Total** | **74** | `bash tests/bash/run.sh` |

**Success criteria**: `bash tests/bash/run.sh` exits 0.

### Phase M2: New Commands

Focus: TypeScript unit tests for new commands + dry-run mode.

| Test area | Tests | Runner |
|-----------|-------|--------|
| Backend state machine | 20 | `bun test tests/unit/Backend.test.ts` |
| RemoteOps mock | 8 | `bun test tests/unit/RemoteOps.mock.test.ts` |
| local-ops helpers | 7 | `bun test tests/unit/local-ops.test.ts` |
| semver utilities | 11 | `bun test tests/unit/semver.test.ts` |
| CLI lifecycle (integration) | 11 | `bun test tests/integration/CLI.lifecycle.test.ts` |
| **Total** | **57** | `bun test` |

**Success criteria**: `bun test` exits 0 with 57 passing tests.

### Phase M3: Self-Managed Lifecycle

Focus: Integration tests for handoff. First mock E2E scenarios.

| Test area | Tests | Runner |
|-----------|-------|--------|
| Full Backend lifecycle (mock) | 11 | `bun test tests/integration/CLI.lifecycle.test.ts` |
| Bash handoff (bwrap sandboxed) | ~45 | `bash tests/bash/run.sh sandboxed` |
| **Total** | **56+** | `bun test` + bash |

**Success criteria**: Both bash and TypeScript integration tests pass.

### Phase M4: Detach Mode

Focus: Scenarios involving client disconnect/reconnect.

| Test area | Tests | Runner |
|-----------|-------|--------|
| CLI disconnect, job persists | 3 | Integration (mock) |
| CLI reconnect to running job | 3 | Integration (mock) |
| Multiple simultaneous tunnels | 2 | Integration (mock) |
| **Total** | **8** | `bun test` |

### Phase M5: Idle Timeout

Focus: Idle timeout detection in monitor_head.

#### Log format

With `VLLM_LOGGING_CONFIG_PATH` pointing to the `vllm_logs.json` config,
vLLM produces access log lines like:

```
(APIServer pid=34633) [2026-07-14 22:37:50,765] INFO:     10.242.0.28:38194 - "GET /health HTTP/1.1" 200 OK
(APIServer pid=34633) [2026-07-14 22:37:50,935] INFO:     10.242.0.28:45178 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=34633) [2026-07-14 22:38:02,993] INFO:     10.242.0.28:45178 - "POST /v1/chat/completions HTTP/1.1" 200 OK
```

Key format elements:
- Prefix: `(APIServer pid=34633)` added by vLLM process wrapper
- Timestamp: `[2026-07-14 22:37:50,765]` — ISO-8601 date with milliseconds
- HTTP method and path: `"GET /health"` or `"POST /v1/chat/completions"`
- Status code: `200 OK`

The `vllm_logs.json` config (from `design/prototype/vllm_logs.json`) uses
uvicorn's `AccessFormatter` with `%(asctime)s`. The vLLM `(APIServer pid=N)`
prefix is added by vLLM's own logging wrapper around uvicorn.

#### Idle timeout detection

`monitor_head` checks for recent API activity by:

1. For `idleTimeout` minutes, generate time patterns for each minute in the
   window (e.g. for a 5-minute timeout, generate 5 patterns)
2. Search the log file for any line containing:
   - One of the time patterns (timestamp within the window)
   - One of the target endpoint patterns (`/v1/chat/completions`, `/v1/models`, etc.)
3. If no match found → shutdown with reason "idle timeout"

The time format for matching uses `date` to format timestamps to match the
log prefix. Since the log format is `[2026-07-14 22:37:50,765]`, the match
pattern strips milliseconds and matches on `[YYYY-MM-DD HH:MM` prefix.

```bash
# Generate matching patterns for the last N minutes
for i in $(seq 0 $idle_timeout); do
    pattern=$(date -d "$i minutes ago" "+%Y-%m-%d %H:%M")
    # Matches: [2026-07-14 22:37 in the log line
    patterns+=("-e" "$pattern")
done
```

#### Test cases

| Test area | Tests | Runner |
|-----------|-------|--------|
| Idle timeout fires correctly | 3 | Bash unit |
| Active traffic prevents timeout | 2 | Bash unit |
| idleTimeout: -1 never times out | 1 | Bash unit |
| Log format parsing (real vLLM output) | 3 | Bash unit |
| **Total** | **9** | `bash tests/bash/test-idle-timeout.sh` |

**Log format test cases**:
- `test_parse_real_log` — feed in the exact log lines from a real vLLM run,
  verify monitor detects recent `/health` and `/v1/chat/completions` calls
- `test_parse_log_timestamp_window` — API call 4 minutes ago, timeout 5 min →
  no shutdown; API call 6 minutes ago, timeout 5 min → shutdown
- `test_parse_log_malformed` — lines with missing timestamps, non-standard
  formats, empty log file → no crash

### Phase M6: Multi-User

Focus: Permission and concurrent access scenarios.

| Test area | Tests | Runner |
|-----------|-------|--------|
| Lockfile group permissions | 2 | Bash unit |
| Second user connects | 2 | Integration (mock) |
| Second user cancels | 2 | Integration (mock) |
| **Total** | **6** | bash + bun |

### Phase M7: Clean Up

Focus: All tests pass, no regressions.

| Suite | Tests | Runner |
|-------|-------|--------|
| Bash unit tests (unit + sandboxed) | ~75 | `bash tests/bash/run.sh` |
| TypeScript unit tests | 57 | `bun test` |
| Integration tests (mock E2E) | 11 | `bun test` (integration/) |
| E2E smoke (Isambard) | 5 | Manual |
| **Total** | **~148+** | CI pipeline |

---

## Running tests

### During development

```bash
# A single test file (works for both unit/ and sandboxed/)
bash tests/bash/sandboxed/test-lockfile.sh
bash tests/bash/sandboxed/test-config.sh

# One test scope (faster than running everything)
bash tests/bash/run.sh unit          # ~ms — pure bash logic
bash tests/bash/run.sh sandboxed     # ~37s — bwrap-sandboxed, real jq/yq

# TypeScript unit tests
bun test tests/unit/Backend.test.ts
bun test tests/unit/RemoteOps.mock.test.ts

# All TypeScript
bun test
```

### Before commit

```bash
# TypeScript
bun test

# Bash: both scopes pass, no red tests
bash tests/bash/run.sh
```

Currently all sandboxed tests pass. Each file is also a regression guard
against the class of bug it originally found (yq dialect mismatches,
subprocess/signal behaviour, env ordering) — the sandboxed harness runs
the tests against the *real* tools from the host, preventing similar
regressions silently returning.

### Before release

```bash
# All bash: both scopes pass, no red tests
bash tests/bash/run.sh

# E2E smoke test (requires HPC access)
bash tests/e2e/smoke.sh 0.22.0
```

---

### Key metrics

| Metric | Target |
|--------|--------|
| Total tests | 57 TS + ~75 bash |
| TypeScript unit tests | 57 (`bun test`) |
| Bash unit (unit/) | ~10 (`bash tests/bash/run.sh unit`) |
| Bash sandboxed tests | ~75 (`bash tests/bash/run.sh sandboxed`) |
| Integration tests (mock E2E) | 11 (`bun test` integration) |
| Tests covering lockfile transitions | ~30 (explicit across TS + bash) |
| Tests covering failure paths | ~40+ |
| Test runtime (dev) | ~5s TS + ~37s bash |
| Test runtime (full) | ~42s (no E2E) |
