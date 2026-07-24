# Test Scripts — isambard-vllm

This document describes the current test suite and how tests are structured,
run, and maintained. All tests pass.

---

## Test Counts

| Scope | Type | Files | Assertions | Status |
|-------|------|-------|------------|--------|
| Bash unit | `tests/bash/unit/` | 1 | Part of 74 | ✅ Green |
| Bash sandboxed | `tests/bash/sandboxed/` | 10 | 74 | ✅ Green |
| TypeScript unit | `tests/unit/` | 2 | 28 | ✅ Green |
| TypeScript integration | `tests/integration/` | 1 | 35 | ✅ Green |
| **Total** | | **14** | **115** | **0 failures** |

Bash tests use a bubblewrap (`bwrap`) sandbox with real `jq 1.7` and `yq 3.4.1`
binaries and mocked SLURM/vLLM commands (`sbatch`, `srun`, `scancel`, `vllm`, etc.).

TypeScript tests use a mock backend (`TestBackend`) with a `TestRemoteOps` that
records all calls but never touches the network. Module mocking via
`bun:test`'s `mock.module()` intercepts `getBackend()` to swap in the mock.

---

## Running Tests

```bash
# All tests (TypeScript + bash unit only)
bun test

# All tests including sandboxed bash (slow, ~5 min)
RUN_LONG_TESTS=true bun test

# Bash tests only
bash tests/bash/run.sh

# Bash unit tests only
bash tests/bash/run.sh unit

# Bash sandboxed tests only
bash tests/bash/run.sh sandboxed

# Run CLI directly
bun run start
```

The `bun test` pipeline runs all TypeScript test files and the bash unit suite
via `tests/unit/bash-integration.test.ts`. Sandboxed bash tests are skipped by
default (controlled by `RUN_LONG_TESTS` env var) because they require bubblewrap
and can take up to 5 minutes.

---

## TypeScript Tests

### `tests/unit/Backend.test.ts`

Tests the **`Backend` abstract class** — lifecycle state helpers and lockfile
parsing. Uses `TestRemoteOps` (a mock `RemoteOps` that records calls and
holds injected lockfile state).

| Test | What it verifies |
|------|-----------------|
| `parseV3Lockfile` valid JSON | Parses status, jobName, returns object |
| `parseV3Lockfile` missing status | Returns `null` |
| `parseV3Lockfile` missing jobName | Returns `null` |
| `parseV3Lockfile` empty input | Returns `null` |
| `parseV3Lockfile` malformed JSON | Returns `null` |
| `isRunning` — running | Returns `true` |
| `isRunning` — stopped | Returns `false` |
| `isRunning` — missing lockfile | Returns `false` |
| `isStopped` — stopped | Returns `true` |
| `isStopped` — failed | Returns `true` |
| `isStopped` — running | Returns `false` |
| `isStopped` — missing lockfile | Returns `true` (never started → stopped) |
| `isStartable` — stopped | Returns `true` |
| `isStartable` — running | Returns `false` |
| `isStartable` — missing lockfile | Returns `true` (never started → startable) |
| `isStarting` — pending | Returns `true` |
| `isStarting` — initialising | Returns `true` |
| `isStarting` — running | Returns `false` |
| `isStarting` — missing lockfile | Returns `false` |

### `tests/unit/local-ops.test.ts`

Tests **`local-ops.ts`** — HTTP operations performed on the local machine
after the SSH tunnel is established. Uses a real `http.Server` on random high
ports (59000+) for each test.

| Test | What it verifies |
|------|-----------------|
| `isLocalPortInUse` — free port | Returns `false` on unused port |
| `isLocalPortInUse` — used port | Returns `true` on active server |
| `isHealthy` — healthy server | Returns `true` when `/health` returns 200 |
| `isHealthy` — no /health endpoint | Returns `false` when `/health` returns 404 |
| `queryModels` — model list | Parses `/v1/models` response correctly |
| `queryModels` — non-2xx response | Throws error |

### `tests/unit/bash-integration.test.ts`

Integrates the **bash test suite** into the Bun test runner. Two test groups:

| Test | Description |
|------|-------------|
| `Bash test suite — unit only` | Runs `bash tests/bash/run.sh unit`, asserts exit 0 |
| `Bash test suite — sandboxed only` | Runs `bash tests/bash/run.sh sandboxed`, skipped unless `RUN_LONG_TESTS=true` |

Note: the old `Bash test suite` (all scopes combined) was removed to allow
independent control of sandboxed tests.

### Removed

| File | Reason |
|------|--------|
| `tests/unit/semver.test.ts` | `src/semver.ts` deleted — semver logic lives only in bash `utils.sh` |
| `tests/unit/RemoteOps.mock.test.ts` | Mock moved into `CLI.lifecycle.test.ts` as `TestRemoteOps` (cleaner, no separate file) |

---

## Integration Test Architecture

### `tests/integration/CLI.lifecycle.test.ts`

Full lifecycle tests using a **mock backend** that records calls but never
touches the network. Tests the Backend contract end-to-end.

#### MockRemoteOps

A `RemoteOps` implementation that records every call (`runRemote`, `copyFile`,
`copyDirectory`, `checkSSH`, `spawnTunnel`) and holds a map of lockfile state
injected by `setLockfile()`. `runRemote()` returns canned responses for known
commands (`ivllm-status.sh`, `ivllm-cancel.sh`, `ivllm-serve.sh`).

#### Module Mocking

The test intercepts the entire `Backend.ts` module via `mock.module()`:

```typescript
mock.module('../../src/backends/Backend.ts', () => {
    return {
        getBackend: async (creds: Credentials) => {
            return new TestBackend(creds);
        },
    };
});
```

This means `cmdConnect` → `getBackend(config)` → `TestBackend` without any
import changes needed in the production code.

#### Test Cases

| Test | What it verifies |
|------|-----------------|
| `requestStart > calls ivllm-serve.sh with correct args` | Backend calls the right wrapper script with job name and time |
| `requestCancel > graceful` | Calls `ivllm-cancel.sh` without `-f` |
| `requestCancel > force` | Calls `ivllm-cancel.sh` with `-f` |
| `getAllJobStatus > returns lockfile list` | Parses JSON array into `LockfileV3[]` |
| `getAllJobStatus > returns empty list when no jobs` | Returns `[]` for empty JSON |
| `lifecycle helpers > isRunning / isStopped / isStartable / isStarting` | All state checks work |
| `bootstrap > calls checkSSH on first use` | SSH check fires on setup |
| `MockRemoteOps — command recording` | All 5 operation types recorded in order |

---

## Bash Tests

All bash tests run inside the bubblewrap sandbox (`tests/bash/lib/sandbox.sh`)
unless noted otherwise. The sandbox provides real `jq`, `yq`, and mocked
SLURM/vLLM commands (shims in `tests/bash/shims/`).

### `tests/bash/unit/test-utils.sh`

Pure bash logic tests — no sandbox needed. Tests helper functions from
`utils.sh` including `select_closest_version()`, `revSemverSort()`,
`parse_semver()`.

### `tests/bash/sandboxed/test-lockfile.sh`

Tests the **lockfile (status.json) state machine**: creation, status transitions,
and cancel/request logic. All functions work on a real JSON file written to disk
by the sandboxed process tree.

| Test | What it verifies |
|------|-----------------|
| `create_pending_basic` | Port valid, timestamps ISO-8601, fields round-trip |
| `create_pending_duplicate` | Duplicate creation fails (idempotency guard) |
| `create_pending_default_timeout` | Default timeout is 30 minutes |
| `update_initialise_basic` | vllmPid, slurmJobId, hostname set |
| `update_initialise_worker_only` | Worker ignored, lockfile stays `pending` |
| `update_running` | Status transition works |
| `update_running_worker_only` | Worker ignored |
| `clean_shutdown` | Full `running` → `stopped` lifecycle, exit code 0 |
| `unclean_shutdown` | `running` → `failed` with reason + exit code |
| `request_cancel` | Status transitions to cancel |
| `request_cancel_no_lockfile` | Fails gracefully |
| `request_cancel_from_worker` | Cancel works from any node |
| `is_status` | Status checks in all states |
| `is_status_missing_lockfile` | Returns false for ghost job |
| `update_reason` | Reason field stored without changing status |
| `get_job_status_setting` | Fields round-trip correctly |
| `full_lifecycle` | Happy path: pending → initialise → running → stopped |
| `lifecycle_cancel` | Cancel + reason + stop |
| `lifecycle_fail_during_startup` | Failed during startup (exit code in status.json) |

### `tests/bash/sandboxed/test-cache.sh`

Tests **JIT cache save/restore** logic: directory creation, permissions, node
gating (only head node saves), and cleanup.

| Test | What it verifies |
|------|-----------------|
| `cache_save_restore` | Files copied to correct location, permissions 0750 |
| `cache_restore_missing` | No errors when cache dir is empty |
| `cache_save_empty` | Graceful handling of empty directory |
| `cache_permissions` | Directory 0750, files 0640 |
| `cache_worker_node_does_not_save` | Worker skips cache save |

### `tests/bash/sandboxed/test-config.sh`

Tests **vllm.yaml config reading** against the real `yq 3.4.1` binary.

| Test | What it verifies |
|------|-----------------|
| `get_job_config_setting_model` | `.model` path resolves correctly |
| `get_job_config_setting_idle_timeout` | Integer value preserved |
| `get_job_config_setting_tensor_parallel` | Integer value preserved |
| `resolve_stripped_job_config_strips_env_and_metadata` | `yq d` removes both blocks |
| `get_job_config_exports_produces_export_lines` | env block → `export KEY=VALUE` lines |
| `get_job_config_exports_empty_env_block` | No export lines for empty env |

### `tests/bash/sandboxed/test-vllm-env.sh`

Tests **common-env.sh** (NVHPC/CUDA/compiler setup) and **vllm-env.sh**
(NCCL/Slingshot/vLLM tuning) sourcing.

| Test | What it verifies |
|------|-----------------|
| `common_env_sources` | No errors on source |
| `common_env_vars_set` | NVHPC_ROOT, CUDA paths, CC/CXX, LD_LIBRARY_PATH |
| `common_env_missing_nvhpc_falls_through_empty` | resolve_nvhpc_root() writes to stderr |
| `vllm_env_sources_and_sets_vars` | NCCL_CROSS_NIC, FI_PROVIDER, VLLM vars |

### `tests/bash/sandboxed/test-monitor-head.sh`

Tests **`monitor_head()`** — the background monitor loop. Uses real background
processes (fake vLLM pid, fake parent pid) so that `kill -0` liveness checks
exercise real subprocess semantics.

| Test | What it verifies |
|------|-----------------|
| `monitor_head_detects_cancel` | status → cancel, reason "user cancel", parent killed |
| `monitor_head_detects_vllm_process_death` | status → failed, reason "lost contact with vLLM" |
| `monitor_head_detects_lockfile_deletion` | Monitor exits, localdir cleaned |
| `monitor_head_idle_timeout_shuts_down` | status → failed, reason "idle timeout" |
| `monitor_head_active_traffic_prevents_idle_timeout` | Monitor does NOT shut down with active traffic |

### `tests/bash/sandboxed/test-monitor-startup.sh`

Tests **`monitor_startup()`** — the foreground monitor. Starts a real mock vLLM
HTTP server (same as `tests/bash/shims/vllm`) that handles `/health` and
`/v1/chat/completions`.

| Test | What it verifies |
|------|-----------------|
| `startup_sends_health_and_warms_up` | Lockfile transitions to `running` |
| `startup_health_then_succeeds` | Monitor polls and succeeds after delay |
| `startup_warmup_fails` | Lockfile → `failed`, reason "vLLM warmup failed" |
| `startup_non_head_node` | Worker returns immediately, status unchanged |
| `startup_wrong_status` | Returns 1 when already `running` |

### `tests/bash/sandboxed/test-monitor-worker.sh`

Tests **`monitor_worker()`** — the background monitor on worker nodes.

| Test | What it verifies |
|------|-----------------|
| `worker_monitor_rejects_head_node` | Returns 1 for head node (SLURM_NODEID=0) |
| `worker_monitor_missing_lockfile` | Returns 1, kills worker pid |
| `worker_monitor_stays_alive_running` | Monitor keeps running |
| `worker_monitor_shuts_down_on_cancel` | Worker killed, localdir cleaned |
| `worker_monitor_shuts_down_on_failed` | Worker killed, localdir cleaned |

### `tests/bash/sandboxed/test-exit-trap.sh`

Tests **`tidy_up()`** — the exit-trap handler. Creates a lockfile with
appropriate status, starts a real background process as the fake vLLM pid,
then calls `tidy_up()` directly. Uses real process signals and shims.

| Test | What it verifies |
|------|-----------------|
| `tidy_up_200_triggers_slurm_timeout` | status → stopped, reason "SLURM timeout" |
| `tidy_up_201_triggers_user_cancel` | status → stopped, reason "user cancel or idle timeout" |
| `tidy_up_0_normal_shutdown` | Status unchanged, vLLM killed if alive |
| `tidy_up_nonzero_crash_runtime` | status → failed, reason "crashed during inference" |
| `tidy_up_nonzero_crash_startup` | status → failed, reason "failed to start" |
| `tidy_up_kills_vllm_process` | Sends SIGTERM, SIGKILL after 2s |
| `tidy_up_cancels_slurm_job` | scancel shim called with correct job id |
| `tidy_up_no_slurm_cancel` | scancel NOT called when slurmJobId is null |

### `tests/bash/sandboxed/test-login-handoff.sh`

Tests the **login-node wrapper scripts** in the `login` profile sandbox.
Scripts run end-to-end calling shimmed `sbatch`/`srun`/`scancel`/`squeue`.
After each script exits, the test reads `/work/calls.log` and asserts the
correct commands were called.

| Test | What it verifies |
|------|-----------------|
| `login_serves_with_minimal_config` | sbatch called with correct job name, partition, gpus, mem |
| `login_cancels_existing` | request_cancel sets status to cancel, scancel NOT called |
| `login_cancels_missing_job` | Fails with error, scancel NOT called |
| `login_shows_status` | Returns JSON with correct jobName |
| `login_setup_runs` | srun called with setup script, version flag |
| `login_force_cancel` | scancel called with correct slurmJobId |

---

## Test Evolution Summary

### What changed in recent commits

| Change | Impact |
|--------|--------|
| `semver.ts` deleted | Removed 60 TypeScript assertions (semver lives in bash `utils.sh` only) |
| `RemoteOps.mock.test.ts` deleted | Mock moved into `CLI.lifecycle.test.ts` as `TestRemoteOps` |
| `CLI.lifecycle.test.ts` rewritten | Uses `mock.module()` to intercept `getBackend()`, `TestBackend` + `TestRemoteOps` pattern |
| `bash-integration.test.ts` restructured | Full test suite removed; sandboxed tests now skip-by-default (`RUN_LONG_TESTS`) |
| `local-ops.test.ts` cleaned up | Removed duplicate `require('http')` calls, added optional chaining |
| All bash tests | Added `# shellcheck disable=SC2016` / `SC2317` comments |
| `test-login-handoff.sh` | Removed skip comment for `ivllm-serve.sh` (syntax error fixed) |
| `Backend.ts` | Added factory pattern (`getBackend()`) with async lazy import |
| `src/index.ts` | All handler functions use `await getBackend(config)` |
