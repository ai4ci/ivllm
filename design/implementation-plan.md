# Implementation Plan — Day 1

Pick up here tomorrow. This plan focuses on Phase M1 (Bash Framework) —
the additive phase that creates the foundation without touching existing code.

---

## Current state

- **364 tests**: 336 pass, 28 fail (stale template tests — expected, these
  test the v2 template system we're replacing)
- **Design docs**: Complete — architecture, ADRs (101–115), roadmap (M1–M7 + F3–F6),
  coding standards, testing plan
- **Prototype**: `design/prototype/prototype.sh` (687 lines) — working bash
  framework with lockfile management, monitor triad, cache, shutdown
- **Prototype test harness**: `design/prototype/test-vllm.sh` (199 lines) —
  mocks srun, scancel, vllm

## What we're building first

The bash framework (`lib/` directory) — the HPC-side runtime that manages
vLLM jobs independently of the CLI. This is additive (no existing code changes).

```
src/templates/lib/
├── utils.sh          ← Lockfile, monitors, cache, shutdown (from prototype)
├── preamble.sh       ← NVHPC/NCCL/Slingshot env vars (from inference.ts)
├── hf.sh             ← Model download (new)
├── vllm_logs.json    ← vLLM logging config (from prototype)

tests/bash/lib/
├── test-utils.sh     ← Mock srun/scancel/vllm (from test-vllm.sh)
├── test-lockfile.sh  ← Lockfile state machine tests
├── test-cache.sh     ← Cache save/restore tests
├── test-monitor.sh   ← Monitor startup/head/worker tests
├── test-shutdown.sh  ← Exit trap and signal tests
├── test-preamble.sh  ← Environment validation tests
└── run.sh            ← Test runner (runs all, exits 0 on pass)
```

---

## Step-by-step tasks

### Step 1: Set up the bash test infrastructure

**Do this first** — we need to be able to run bash tests before writing any
bash code.

1. Create `tests/bash/lib/test-utils.sh` — extract mock functions from
   `design/prototype/test-vllm.sh` into a reusable library:

   ```bash
   # tests/bash/lib/test-utils.sh
   
   # Mock srun — spawns command in background
   mock_srun() {
       "$@" & MOCK_SRUN_PID=$!
       echo $MOCK_SRUN_PID
   }
   
   # Mock scancel — logs and no-ops
   mock_scancel() {
       echo "[mock] scancel $1"
   }
   
   # Mock vLLM — lightweight Python HTTP server
   mock_vllm() {
       # Serves /health, /v1/models, /v1/chat/completions
       # Configurable startup delay (--delay N)
       # Configurable crash-after-N-requests (--crash-after N)
       exec python3 -c "
   import http.server, json, sys, os, time
   delay = int(os.environ.get('MOCK_VLLM_DELAY', '0'))
   time.sleep(delay)
   # ... HTTP server ...
   "
   }
   
   # Assert helpers
   assert_file_exists() { [[ -f "$1" ]] || { echo "FAIL: $1 not found"; return 1; }; }
   assert_json_eq() { local val=$(jq -r "$2" "$1"); [[ "$val" == "$3" ]] || { echo "FAIL: $1 $2 expected $3 got $val"; return 1; }; }
   ```

2. Create `tests/bash/run.sh` — test runner:
   ```bash
   #!/bin/bash
   set -euo pipefail
   FAIL=0
   for test in tests/bash/test-*.sh; do
       echo "=== $test ==="
       bash "$test" || { FAIL=1; echo "FAIL: $test"; }
   done
   exit $FAIL
   ```

3. **Verify**: `bash tests/bash/run.sh` prints "No test files found" (expected
   — no tests written yet, but the runner works). Commit.

### Step 2: Port prototype.sh → utils.sh

Refactor `design/prototype/prototype.sh` into a clean library file at
`src/templates/lib/utils.sh`. The prototype works — now make it production-ready:

| Prototype function | Target | Changes needed |
|-------------------|--------|----------------|
| `resolve_location`, `resolve_lockfile`, `resolve_logfile`, `resolve_cachetar`, `resolve_localdir`, `resolve_setting` | Keep | Use `ENGINE_DIR` variable; validate paths exist |
| `create_status_pending` | Keep | Add `ENGINE_DIR` prefix; validate all required args |
| `update_status_initialise` | Keep | Add SLURM_NODEID guard; validate jq output |
| `update_status_running` | Keep | Same pattern |
| `update_status_clean_shutdown` | Keep | Add stop_time |
| `update_status_unclean_shutdown` | Keep | Add exit_code and stop_time |
| `request_cancel` | Keep | Same |
| `is_status` | Keep | Validate lockfile exists first |
| `tidy_up` | Keep | Add timeout between SIGTERM and SIGKILL |
| `setup_traps` | Keep | Same |
| `report_memory` | Keep | Add GPU metrics (nvidia-smi) |
| `monitor_startup` | Keep | Add warmup retry logic |
| `monitor_head` | Keep | Add idle timeout logic |
| `monitor_worker` | Keep | Same |
| `restore_cache` | Keep | Same |
| `save_cache` | Keep | Same |
| `clear_localdir` | Keep | Same |

**Don't change the logic** — the prototype has been manually tested. Just
clean up: add `set -euo pipefail` guards, validate paths, add comments.

### Step 3: Extract preamble.sh

Move the NVHPC/NCCL/Slingshot environment from `inference.ts:renderNVHPCPreamble()`
into a standalone `src/templates/lib/preamble.sh`. This preserves the hard-won
tuning (years of trial and error) in a directly usable bash file.

The preamble should be a bash file that sets all the environment variables
and can be sourced at the top of any SLURM script:

```bash
# src/templates/lib/preamble.sh
# Source this at the top of any SLURM script for Isambard GH200 tuning.
# Contains: NVHPC SDK paths, CUDA forward compat, NCCL/Slingshot tuning,
# compiler selection, JIT compilation limits, vLLM overrides.

export NVHPC_ROOT=${NVHPC_ROOT:-/path/to/nvhpc}
export CUDA_HOME=$NVHPC_ROOT/cuda/12.9
export PATH=$CUDA_HOME/bin:$PATH
# ... all the env vars from renderNVHPCPreamble() ...
```

### Step 4: Write lockfile tests (TDD)

Before writing any more bash code, write the tests. Each test file follows
the same pattern: source test-utils.sh, source the library, run test functions.

```bash
# tests/bash/test-lockfile.sh

source tests/bash/lib/test-utils.sh
source src/templates/lib/utils.sh

ENGINE_DIR=$(mktemp -d)

test_create_pending() {
    local job="test-job"
    create_status_pending "$job" "test-model" 49153 30
    local lockfile="$ENGINE_DIR/jobs/$job/status.json"
    assert_file_exists "$lockfile"
    assert_json_eq "$lockfile" ".status" '"pending"'
    assert_json_eq "$lockfile" ".jobName" '"test-job"'
    assert_json_eq "$lockfile" ".serverPort" '49153'
    assert_json_eq "$lockfile" ".idleTimeout" '30'
    echo "✓ test_create_pending"
}

test_create_pending_existing() {
    local job="test-job"
    # Second create must fail (set -C)
    if create_status_pending "$job" "test-model" 49153 30 2>/dev/null; then
        echo "FAIL: should have errored on duplicate"
        return 1
    fi
    echo "✓ test_create_pending_existing"
}

# Run tests
test_create_pending
test_create_pending_existing

rm -rf "$ENGINE_DIR"
```

**Test cases for lockfile** (`tests/bash/test-lockfile.sh`):

| Test | What it checks |
|------|---------------|
| `test_create_pending` | Lockfile created with correct fields |
| `test_create_pending_existing` | `set -C` prevents overwrite |
| `test_update_initialise` | SLURM job ID, hostname, PID written |
| `test_update_running` | Status changes to running |
| `test_update_clean_shutdown` | Status changes to stopped with stopTime |
| `test_update_unclean_shutdown` | Status changes to failed with reason + exitCode |
| `test_request_cancel` | Status changes to cancel |
| `test_is_status` | Returns true/false correctly |
| `test_is_status_missing_file` | No crash on missing lockfile |
| `test_is_status_malformed` | No crash on corrupt JSON |

**Test cases for cache** (`tests/bash/test-cache.sh`):

| Test | What it checks |
|------|---------------|
| `test_cache_save_restore` | Round-trip: save tar, restore to new dir, files match |
| `test_cache_restore_missing` | No crash on missing tar |
| `test_cache_save_empty` | Empty tar created |
| `test_cache_permissions` | Tar has group-read (664) |

**Test cases for monitors** (`tests/bash/test-monitor.sh`):

| Test | What it checks |
|------|---------------|
| `test_startup_normal` | Healthy vLLM → status transitions to running, cache saved |
| `test_startup_delayed` | vLLM takes 30s → monitor waits, eventually succeeds |
| `test_startup_fail_health` | vLLM never responds → failed status |
| `test_startup_process_dies` | vLLM dies during startup → failed status |
| `test_startup_cancel_during` | User cancels during startup → shutdown |
| `test_head_cancel_request` | User writes cancel → SIGUSR2 sent |
| `test_head_process_death` | vLLM dies → detected in 10s |
| `test_head_no_lockfile` | Lockfile deleted → shutdown |
| `test_head_idle_timeout` | No API calls → shutdown with "idle timeout" |
| `test_head_active_traffic` | API calls keep coming → no shutdown |
| `test_worker_cancel` | Lockfile says cancel → worker shuts down |
| `test_worker_lockfile_deleted` | Lockfile disappears → worker shuts down |

**Monitor test data**: Real vLLM access log lines (from a live run with
`vllm_logs.json` config):

```
(APIServer pid=34633) [2026-07-14 22:37:50,765] INFO:     10.242.0.28:38194 - "GET /health HTTP/1.1" 200 OK
(APIServer pid=34633) [2026-07-14 22:37:50,935] INFO:     10.242.0.28:45178 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=34633) [2026-07-14 22:38:02,993] INFO:     10.242.0.28:45178 - "POST /v1/chat/completions HTTP/1.1" 200 OK
(APIServer pid=34633) [2026-07-14 22:38:03,533] INFO:     10.242.0.28:34266 - "GET /health HTTP/1.1" 200 OK
```

These lines should be embedded in the monitor test data to verify that
`monitor_head` correctly parses real vLLM output. The time format for
matching is `[%Y-%m-%d %H:%M` (strip milliseconds, match minute granularity).

See `design/prototype/multi-user-structure.md` (log files section) and
`design/prototype/vllm_logs.json` for the source logging configuration.

**Test cases for shutdown** (`tests/bash/test-shutdown.sh`):

| Test | What it checks |
|------|---------------|
| `test_exit_0` | Normal exit, status set appropriately |
| `test_exit_sigusr1` | SLURM timeout → "SLURM timeout" reason |
| `test_exit_sigusr2` | User cancel → reason preserved |
| `test_exit_crash_startup` | Non-zero during initialising → failed |
| `test_exit_crash_runtime` | Non-zero during running → failed |
| `test_tidy_up_kills_vllm` | vLLM PID is killed, scancel is called |

**Test cases for preamble** (`tests/bash/test-preamble.sh`):

| Test | What it checks |
|------|---------------|
| `test_preamble_sources` | Source preamble.sh without errors |
| `test_preamble_nvhpc_root` | NVHPC_ROOT points to existing dir |
| `test_preamble_cuda_home` | CUDA_HOME contains cuda/12.9 |
| `test_preamble_ld_library` | compat dir is first in LD_LIBRARY_PATH |

### Step 5: Run red/green on each test file

For each test file:

1. **Red**: Run the test file — all tests fail (library not written yet)
2. **Green**: Write the corresponding library function until the test passes
3. **Commit**: Commit the test + implementation together

### Step 6: Integrate with existing project

Once the test suite passes:

```bash
bash tests/bash/run.sh   # all bash tests pass
bun test                  # all 336 existing TS tests still pass
```

Add a `test:bash` script to `package.json`:

```json
{
  "scripts": {
    "test:bash": "bash tests/bash/run.sh",
    "test": "bash tests/bash/run.sh && bun test"
  }
}
```

---

## What success looks like

By end of Day 1:

```
bash tests/bash/run.sh
=== tests/bash/test-lockfile.sh ===
✓ test_create_pending
✓ test_create_pending_existing
✓ test_update_initialise
... (all pass)
=== tests/bash/test-cache.sh ===
✓ test_cache_save_restore
... (all pass)
=== tests/bash/test-monitor.sh ===
✓ test_startup_normal
... (all pass)
=== tests/bash/test-shutdown.sh ===
✓ test_exit_0
... (all pass)
=== tests/bash/test-preamble.sh ===
✓ test_preamble_sources
... (all pass)
```

~45 bash tests passing, creating the foundation for Phase M2 (new CLI commands).

---

## Cheat sheet

```bash
# Run a single bash test file during development
bash tests/bash/test-lockfile.sh

# Run all bash tests
bash tests/bash/run.sh

# Run a single TypeScript test
bun test tests/vllm-config.test.ts

# Run all TypeScript tests
bun test

# Run everything
bash tests/bash/run.sh && bun test
```
