# Testing Plan — isambard-vllm v3

## Current state

The existing tests (~3,000 lines across 18 files) are **template-centric**:
they verify that generated bash scripts contain specific strings (e.g.
`"NVHPC_ROOT"`, `"cuda/12.9/compat"`) and that TypeScript functions parse
arguments correctly. They do not test:

- Whether the generated bash actually **works** correctly
- Whether the lockfile state machine transitions reliably
- Whether the handoff between CLI and compute-side handles edge cases
- Whether timeouts, cancellations, and failure paths behave correctly

The user's observation is correct: the tests are strategic (covering surface
area) but not tactical (covering behaviour under realistic conditions).

---

## Target: Three-layer test architecture

```
┌──────────────────────┐
│   Unit tests         │  Fast, isolated. Test one function/module.
│   (TypeScript + Bash)│  Bash tests use mock harness.
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
| TypeScript unit | `tests/cli/*.test.ts` | `bun test` | ms |
| Bash unit | `tests/bash/test-*.sh` | `bash tests/bash/run.sh` | ms–s |
| Integration | `tests/integration/*.test.ts` | `bun test` | s |
| E2E | `tests/e2e/smoke.sh` | Manual / CI on Isambard | 5–30 min |

---

## Layer 1: TypeScript Unit Tests

Existing test files in `tests/` are retained and expanded. New commands
(`connect`, `cancel`) get new test files.

| Test file | What it tests | Status |
|-----------|---------------|--------|
| `cli/connect.test.ts` | Option parsing, lockfile creation, state-dependent behaviour | New |
| `cli/cancel.test.ts` | Cancel command, force cancel, state validation | New |
| `cli/config.test.ts` | Existing — extend for `idleTimeout` field | Expand |
| `cli/vllm-config.test.ts` | Existing — extend for `idleTimeout` in YAML | Expand |
| `cli/semver.test.ts` | Existing — no changes needed | Preserve |
| `cli/setup.test.ts` | Existing — minor path updates for new project structure | Expand |
| `cli/assistant.test.ts` | Existing — no changes needed | Preserve |
| `cli/agent.test.ts` | Existing — no changes needed | Preserve |

**Key principle**: OptionParser tests verify that `--flag value` maps to
`{ flag: "value" }` in the options object. Do NOT test individual option
values in isolation — test the command handler with a set of options and
verify the derived state.

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

1. Refactor `makeRemoteOps` to accept `OpsMode` instead of `boolean`
2. Add `MockRemoteFs` class (wraps `mkdir`, `cat`, `jq` against a temp dir)
3. Build `mock` mode that delegates to `MockRemoteFs` for filesystem commands
4. Update existing tests that pass `dryRun: true` → pass `'dry-run'`
5. Add new integration tests using `'mock'` mode

The bash framework must be tested without an HPC connection. All tests use
the mock harness (`tests/bash/lib/test-utils.sh`) which provides:

| Mock | What it replaces | Behaviour |
|------|-----------------|-----------|
| `mock_srun` | `srun` | Spawns the real command in background, captures PID |
| `mock_srun_fail` | `srun` | Exits immediately with a configurable exit code |
| `mock_scancel` | `scancel` | Logs the cancel request, no-ops |
| `mock_vllm` | `vllm serve` | Lightweight Python HTTP server serving `/health`, `/v1/models`, `/v1/chat/completions`. Configurable startup delay, configurable failure mode. |
| `mock_vllm_slow` | `vllm serve` | Same as mock_vllm but with configurable delay before `/health` responds (for testing startup monitor timing) |
| `mock_vllm_crash` | `vllm serve` | Starts healthy, then crashes after N requests (for testing head monitor) |

### Test scenarios by bash function

#### Lockfile operations (`lib/utils.sh`)

| Test | Scenario | Expected |
|------|----------|----------|
| `test_create_pending` | Create lockfile for new job | File exists with `status: "pending"`, correct fields |
| `test_create_pending_existing` | Create lockfile when one exists | `set -C` causes failure, script reports error |
| `test_update_initialise` | Update from pending to initialising | SLURM job ID, hostname, PID written correctly |
| `test_update_running` | Update from initialising to running | Status changes to `running` |
| `test_update_clean_shutdown` | Update to stopped | Status changes to `stopped` with stop time |
| `test_update_unclean_shutdown` | Update to failed | Status changes to `failed` with reason and exit code |
| `test_request_cancel` | Write cancel to lockfile | Status changes to `cancel` |
| `test_is_status` | Check lockfile status | Returns true/false correctly |
| `test_is_status_missing_file` | Check status on missing lockfile | Returns false, no crash |
| `test_is_status_malformed` | Check status on corrupt JSON | Handles gracefully |

#### Cache functions (`lib/utils.sh`)

| Test | Scenario | Expected |
|------|----------|----------|
| `test_cache_save_restore` | Save cache, verify tar exists, restore to new dir | Files match original |
| `test_cache_restore_missing` | Restore from non-existent tar | Graceful message, no crash |
| `test_cache_save_empty` | Save empty directory | Tar created with zero-size content |
| `test_cache_permissions` | Saved tar has group-read permissions | `chmod 664` verified |

#### Monitor startup (`lib/utils.sh:monitor_startup`)

| Test | Scenario | Expected |
|------|----------|----------|
| `test_startup_normal` | Start mock vLLM, wait for /health, warmup succeeds | Status transitions to `running`, cache saved twice (pre + post warmup) |
| `test_startup_delayed` | Mock vLLM takes 30s to respond | Monitor waits, eventually succeeds |
| `test_startup_fail_health` | Mock vLLM never responds to /health | Monitor times out, status → `failed` |
| `test_startup_fail_warmup` | /health responds but warmup fails | After 5 retries, status → `failed` |
| `test_startup_process_dies` | vLLM process dies during startup | Monitor detects, status → `failed` |
| `test_startup_cancel_during` | User writes cancel during startup | Monitor detects cancel, triggers shutdown |

#### Monitor head (`lib/utils.sh:monitor_head`)

| Test | Scenario | Expected |
|------|----------|----------|
| `test_head_no_lockfile` | Lockfile deleted during runtime | Shutdown with reason "lockfile deleted" |
| `test_head_cancel_request` | User writes cancel | SIGUSR2 sent to parent, shutdown initiated |
| `test_head_process_death` | vLLM process dies | Detected in 10s, shutdown initiated |
| `test_head_idle_timeout` | No API requests within idle window | Shutdown with reason "idle timeout" |
| `test_head_active_traffic` | API requests keep coming | No shutdown |
| `test_head_idle_timeout_edge` | Request just inside timeout window | No shutdown |
| `test_head_idle_timeout_disabled` | `idleTimeout: -1` | Never shuts down, even with no traffic |
| `test_head_missing_log` | Log file deleted during runtime | Shutdown with reason "missing log file" |

#### Monitor worker (`lib/utils.sh:monitor_worker`)

| Test | Scenario | Expected |
|------|----------|----------|
| `test_worker_normal` | Lockfile stays in running | No action |
| `test_worker_cancel` | Lockfile transitions to cancel | Worker shuts down local process |
| `test_worker_failed` | Lockfile transitions to failed | Worker shuts down |
| `test_worker_lockfile_deleted` | Lockfile disappears | Worker shuts down |
| `test_worker_head_node` | Accidentally run on head node | Error, kills immediately |
| `test_worker_stopped` | Lockfile transitions to stopped | Worker shuts down |

#### Exit trap and signals (`lib/utils.sh:tidy_up`)

| Test | Scenario | Expected |
|------|----------|----------|
| `test_exit_0` | vLLM exits normally (code 0) | Clean shutdown, status set appropriately |
| `test_exit_sigusr1` | SIGUSR1 received (SLURM timeout) | Status → `stopped`, reason "SLURM timeout" |
| `test_exit_sigusr2` | SIGUSR2 received (idle/cancel) | Status → `stopped`, reason preserved |
| `test_exit_crash_startup` | Non-zero exit during initialising | Status → `failed`, reason "failed to start" |
| `test_exit_crash_runtime` | Non-zero exit during running | Status → `failed`, reason "crashed during inference" |
| `test_exit_tidy_up` | vLLM PID killed, scancel called | Process is killed, slurm cancelled |
| `test_exit_scancel_after_kill` | tidy_up kills vLLM, then scancel | Both happen in order |

#### Environment preamble (`lib/vllm-env.sh`)

| Test | Scenario | Expected |
|------|----------|----------|
| `test_preamble_sources` | Source vllm-env.sh | No errors, env vars set |
| `test_preamble_nvhpc_root` | NVHPC_ROOT correctly set | Points to existing directory |
| `test_preamble_cuda_home` | CUDA_HOME correctly set | Contains `cuda/12.9/` |
| `test_preamble_path` | PATH includes CUDA bin | `which nvcc` works after source |
| `test_preamble_ld_library` | LD_LIBRARY_PATH includes compat libs | compat dir comes first |
| `test_preamble_cc_cxx` | CC and CXX set to gcc | `gcc` and `g++` are the compilers |

### Mock test runner structure

```bash
# tests/bash/run.sh — runs all bash tests
#!/bin/bash
set -euo pipefail
FAIL=0

for test in tests/bash/test-*.sh; do
    echo "=== Running $test ==="
    if bash "$test"; then
        echo "✓ $test"
    else
        echo "✗ $test"
        FAIL=1
    fi
done

exit $FAIL
```

Each test file sources the shared mock utilities and then runs individual
test functions:

```bash
#!/bin/bash
# tests/bash/test-lockfile.sh

source tests/bash/lib/test-utils.sh
source src/templates/lib/utils.sh

test_create_pending() {
    local job="test-job"
    local work_dir=$(mktemp -d)
    
    ENGINE_DIR="$work_dir/engine"
    create_status_pending "$job" "test-model" 8000 30
    
    local lockfile="$ENGINE_DIR/jobs/$job/status.json"
    assert_file_exists "$lockfile"
    assert_json_eq "$lockfile" ".status" '"pending"'
    assert_json_eq "$lockfile" ".jobName" '"test-job"'
    assert_json_eq "$lockfile" ".idleTimeout" '30'
    
    rm -rf "$work_dir"
    echo "✓ test_create_pending"
}

test_create_pending_existing() {
    local job="test-job"
    local work_dir=$(mktemp -d)
    
    ENGINE_DIR="$work_dir/engine"
    create_status_pending "$job" "test-model" 8000 30
    # Second create must fail (set -C)
    if create_status_pending "$job" "test-model" 8000 30 2>/dev/null; then
        echo "✗ Should have failed on duplicate"
        return 1
    fi
    
    rm -rf "$work_dir"
    echo "✓ test_create_pending_existing"
}

# Run all tests
test_create_pending
test_create_pending_existing
# ... etc
```

---

## Layer 3: Integration Tests (Mock E2E)

Integration tests verify the **handoff** between layers — the full lifecycle
with mock vLLM, real TypeScript CLI (in `--mock` mode), and real bash
framework (sourced, not templated).

### The Handoff Interface

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

### Integration test scenarios

| # | Scenario | Steps | Expected |
|---|----------|-------|----------|
| 1 | **Happy path: connect → serve → cancel** | CLI creates lockfile, submits sbatch (mock), monitor polls, mock vLLM starts, tunnel established, CLI writes cancel | All status transitions correct, clean shutdown |
| 2 | **CLI disconnects mid-flight** | Connect, wait for running, kill CLI process (SIGKILL) | Job stays running, lockfile still says running |
| 3 | **CLI reconnects** | After scenario 2, run connect again | CLI reads lockfile, establishes new tunnel |
| 4 | **Cancel during startup** | Connect, while initialising, cancel from another terminal | Clean shutdown, status → stopped |
| 5 | **Double connect (same job)** | Connect, try connect again from another terminal | Second connect attaches to running instance |
| 6 | **Force cancel** | Connect, then cancel --force | scancel called, lockfile updated to stopped |
| 7 | **SLURM timeout** | Mock srun sends SIGUSR1 after N seconds | Clean shutdown, reason "SLURM timeout" |
| 8 | **Idle timeout** | Connect, wait, no API calls | After idleTimeout minutes, job shuts down |
| 9 | **Startup failure** | mock_vllm_crash on first request | Status → failed, reason captured |
| 10 | **Model download failure** | Mock hf.sh exits non-zero | CLI reports error, lockfile cleaned up |
| 11 | **Multi-node lifecycle** | Connect with pp=2, mock 2 nodes | Both nodes start, monitor_head + monitor_worker run |

### Integration test structure in TypeScript

```typescript
// tests/integration/lifecycle.test.ts

import { describe, test, expect, beforeAll, afterAll } from 'bun:test';
import { execSync, spawn } from 'child_process';
import { mkdtempSync, writeFileSync } from 'fs';
import { join } from 'path';

describe('Full lifecycle with mock vLLM', () => {
  const tmpDir = mkdtempSync('ivllm-test-');
  const configYaml = join(tmpDir, 'vllm.yaml');
  const jobName = 'test-lifecycle';
  
  beforeAll(() => {
    // Write a minimal vllm.yaml
    writeFileSync(configYaml, `
model: test-model
tensor-parallel-size: 1
`);
    // Set up mock environment
    process.env.IVLLM_TEST_MOCK = '1';
    process.env.IVLLM_TEST_DIR = tmpDir;
  });
  
  test('connect creates pending lockfile', async () => {
    const result = execSync(
      `ivllm connect ${jobName} --config ${configYaml} --mock --dry-run`,
      { encoding: 'utf-8' }
    );
    // Verify dry-run output mentions lockfile creation
    expect(result).toContain('lockfile');
    // Verify the generated script is valid bash
    const scriptPath = join(tmpDir, `${jobName}.slurm.sh`);
    execSync(`bash -n ${scriptPath}`);
  });
  
  // More tests...
});
```

### Integration test for bash handoff (pure bash, no TypeScript)

```bash
# tests/integration/test-handoff.sh
# Tests the CLI↔Compute lockfile interface entirely in bash

source tests/bash/lib/test-utils.sh
source src/templates/lib/utils.sh

test_handoff_cli_creates_bash_reads() {
    local job="handoff-test"
    local work_dir=$(mktemp -d)
    export ENGINE_DIR="$work_dir/engine"
    
    # Simulate CLI creating the lockfile (as TypeScript would)
    create_status_pending "$job" "test-model" 49153 30
    
    # Simulate bash reading it (as the SLURM script would)
    local lockfile=$(resolve_job_status "$job")
    assert_file_exists "$lockfile"
    assert_json_eq "$lockfile" ".status" '"pending"'
    
    # Simulate bash updating it
    export SLURM_JOB_ID=99999
    export COMPUTE_HOSTNAME="nid-test"
    export SLURM_NODEID=0
    update_status_initialise "$job" 12345
    
    # Verify CLI-relevant fields
    assert_json_eq "$lockfile" ".status" '"initialising"'
    assert_json_eq "$lockfile" ".slurmJobId" '99999'
    assert_json_eq "$lockfile" ".computeHostname" '"nid-test"'
    assert_json_eq "$lockfile" ".vllmPid" '12345'
    
    rm -rf "$work_dir"
    echo "✓ test_handoff_cli_creates_bash_reads"
}
```

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
| Lockfile operations | 8 | `bash tests/bash/test-lockfile.sh` |
| Cache functions | 4 | `bash tests/bash/test-cache.sh` |
| Monitor startup | 6 | `bash tests/bash/test-monitor-startup.sh` |
| Monitor head | 9 | `bash tests/bash/test-monitor-head.sh` |
| Monitor worker | 6 | `bash tests/bash/test-monitor-worker.sh` |
| Exit trap / signals | 6 | `bash tests/bash/test-shutdown.sh` |
| Environment preamble | 6 | `bash tests/bash/test-vllm-env.sh` |
| **Total** | **45** | `bash tests/bash/run.sh` |

**Success criteria**: `bash tests/bash/run.sh` exits 0.

### Phase M2: New Commands

Focus: TypeScript unit tests for new commands + dry-run mode.

| Test area | Tests | Runner |
|-----------|-------|--------|
| `connect.ts` option parsing | 5+ | `bun test tests/cli/connect.test.ts` |
| `cancel.ts` option parsing | 4+ | `bun test tests/cli/cancel.test.ts` |
| `list.ts` with new lockfile | 3+ | `bun test tests/cli/list.test.ts` |
| Lockfile schema parsing | 5+ | Add to existing vllm-config tests |
| **Total** | **17+** | `bun test` |

**Success criteria**: `bun test` exits 0. Old tests still pass.

### Phase M3: Self-Managed Lifecycle

Focus: Integration tests for handoff. First mock E2E scenarios.

| Test area | Tests | Runner |
|-----------|-------|--------|
| Handoff: CLI→bash lockfile | 5 | `bash tests/integration/test-handoff.sh` |
| Full lifecycle E2E (mock) | 5+ | `bun test tests/integration/lifecycle.test.ts` |
| Cancel scenarios | 4+ | `bun test tests/integration/cancel.test.ts` |
| Startup failure paths | 3+ | `bun test tests/integration/failure.test.ts` |
| **Total** | **17+** | bash + bun |

**Success criteria**: Both bash and TypeScript integration tests pass.
Manual E2E smoke test passes on Isambard (scenarios 1-2).

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
| Bash unit tests | ~53 | `bash tests/bash/run.sh` |
| TypeScript unit tests | ~40+ | `bun test` |
| Integration tests (mock E2E) | ~25+ | `bun test` |
| E2E smoke (Isambard) | 5 | Manual |
| **Total** | **~123+** | CI pipeline |

---

## Running tests

### During development

```bash
# Bash framework changes (fast)
bash tests/bash/test-lockfile.sh         # Single test file
bash tests/bash/run.sh                   # All bash tests

# TypeScript changes (fast)
bun test tests/cli/connect.test.ts       # Single test file
bun test                                  # All TypeScript tests

# Integration tests (moderate)
bash tests/integration/test-handoff.sh   # Bash integration
bun test tests/integration/              # TypeScript integration
```

### Before commit

```bash
bash tests/bash/run.sh && bun test       # ~30s
```

### Before release

```bash
# Full test suite
bash tests/bash/run.sh && bun test

# E2E smoke test (requires HPC access)
bash tests/e2e/smoke.sh 0.22.0
```

---

## Key metrics

| Metric | Current | Target |
|--------|---------|--------|
| Total tests | ~80 | ~120+ |
| Bash unit tests | 0 | ~53 |
| Integration tests | 0 | ~25+ |
| Tests covering lockfile transitions | ~2 (implicit) | ~20 (explicit) |
| Tests covering failure paths | ~3 | ~20+ |
| Test runtime (dev) | ~10s | ~30s |
| Test runtime (full) | ~10s | ~30s (no E2E) |
| E2E test coverage | Manual only | Scripted smoke test |
