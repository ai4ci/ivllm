# Test Scripts — isambard-vllm

This document describes the current test suite and how tests are structured,
run, and maintained. All tests pass.

---

## Single Entry Point: `bun test`

**Bash tests are not a separate suite you run independently in normal
workflow — they are wrapped inside `bun test`** via
`tests/unit/bash-integration.test.ts`, which shells out to
`bash tests/bash/run.sh`. A single `bun test` command exercises TypeScript
*and* bash.

Two speed tiers, split by an environment variable so the fast path stays fast:

| Command | Runs | Bash scope | Typical time |
|---------|------|-----------|--------------|
| `bun test` (= `npm test`) | TS unit + integration + bash **unit** | `tests/bash/unit/` only | ~0.2s |
| `RUN_LONG_TESTS=true bun test` (= `npm run test:all`) | Everything | `tests/bash/unit/` + `tests/bash/sandboxed/` | ~70s |

The sandboxed bash suite is gated behind `RUN_LONG_TESTS` because it spins up
a bubblewrap (`bwrap`) sandbox per test (process/PID namespace, mock HTTP
servers, real subprocess signals) — each test costs real wall-clock time.
Gating it out of the default `bun test` run keeps the everyday
edit-test-repeat loop fast; `test:all` is the full/CI-grade run.

**Fixed bug:** `tests/unit/bash-integration.test.ts`'s two `it()` blocks call
`execSync` with a 30s/300s timeout for the *child process*, but had no
matching timeout on the `it()` itself. bun:test's default 5s per-test timeout
was killing the assertion before the sandboxed bash suite (which legitimately
takes ~69s) could finish — so `RUN_LONG_TESTS=true bun test` reported a
failure even though the underlying bash tests were all green. Fixed by adding
an explicit `{ timeout: 30_000 }` / `{ timeout: 300_000 }` third argument to
each `it()` call, matching its own `execSync` timeout.

---

## Test Counts

Two ways to count, because bash tests are nested inside two TS-level `it()`
calls that each make a single assertion about the *exit code* of a whole
bash suite run — the granular bash assertions aren't visible to `bun test`'s
own tally.

### As `bun test` / `bun test:all` sees it (TypeScript-level)

| Scope | Files | Tests | `expect()` calls | Status |
|-------|-------|-------|-------------------|--------|
| `tests/unit/Backend.test.ts` | 1 | 22 | 26 | ✅ Green |
| `tests/unit/local-ops.test.ts` | 1 | 7 | 12 | ✅ Green |
| `tests/unit/bash-integration.test.ts` | 1 | 2 (1 gated by `RUN_LONG_TESTS`) | 2 | ✅ Green |
| `tests/integration/CLI.lifecycle.test.ts` | 1 | 12 | 27 | ✅ Green |
| **`bun test` (default)** | 4 | **43** (42 pass, 1 skip) | **66** | ✅ 0 fail |
| **`bun test:all`** (`RUN_LONG_TESTS=true`) | 4 | **43** (all pass) | **67** | ✅ 0 fail |

### Inside `bash tests/bash/run.sh` (bash-level, what `bash-integration.test.ts` wraps)

| Scope | Files | Assertions | Status |
|-------|-------|------------|--------|
| `tests/bash/unit/test-utils.sh` | 1 | 12 | ✅ Green |
| `tests/bash/sandboxed/*.sh` | 9 | 63 | ✅ Green |
| **Total** | **10** | **75** | **0 failures** |

Bash tests use a bubblewrap (`bwrap`) sandbox with real `jq 1.7` and `yq 3.4.1`
binaries and mocked SLURM/vLLM commands (`sbatch`, `srun`, `scancel`, `vllm`,
`hf`, `uv`, etc. — shims in `tests/bash/shims/`).

TypeScript tests use a mock backend (`TestBackend`) with a `TestRemoteOps` that
records all calls but never touches the network. Module mocking via
`bun:test`'s `mock.module()` intercepts `getBackend()` (from
`src/backends/backend-factory.ts`) to swap in the mock.

---

## Running Tests

```bash
# Default — fast path (TS unit + integration + bash unit only)
bun test
npm test

# Full suite — adds sandboxed bash (~70s)
RUN_LONG_TESTS=true bun test
npm run test:all

# Bash tests standalone (bypasses bun entirely — useful when iterating
# on bash scripts only, without paying the bun startup/TS overhead)
bash tests/bash/run.sh          # unit + sandboxed
bash tests/bash/run.sh unit     # unit only
bash tests/bash/run.sh sandboxed # sandboxed only

# Run CLI directly
bun run start
```

Prefer `bun test` for everyday development and `bun run test:all` before
committing (per `AGENTS.md`'s pre-commit checklist: all affected tests must
pass). Reach for `bash tests/bash/run.sh` directly only when iterating on
bash-only changes and you want to skip the TS test files.

---

## TypeScript Tests

### `tests/unit/Backend.test.ts` — 22 tests, 26 `expect()` calls

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
| `getJobStatus` — found | Finds job in list |
| `getJobStatus` — not found | Throws |
| `isRunning` — running | Returns `true` |
| `isRunning` — pending | Returns `false` |
| `isRunning` — missing lockfile | Returns `false` |
| `isStopped` — stopped | Returns `true` |
| `isStopped` — failed | Returns `true` |
| `isStopped` — running | Returns `false` |
| `isStopped` — missing lockfile | Returns `true` (never started → stopped) |
| `isStartable` — stopped | Returns `true` |
| `isStartable` — failed | Returns `true` |
| `isStartable` — running | Returns `false` |
| `isStartable` — missing lockfile | Returns `true` (never started → startable) |
| `isStarting` — pending | Returns `true` |
| `isStarting` — initialising | Returns `true` |
| `isStarting` — running | Returns `false` |
| `isStarting` — missing lockfile | Returns `false` |

### `tests/unit/local-ops.test.ts` — 7 tests, 12 `expect()` calls

Tests **`local-ops.ts`** — HTTP operations performed on the local machine
after the SSH tunnel is established. Uses a real `http.Server` on random high
ports (59000+) for each test.

| Test | What it verifies |
|------|-----------------|
| `isLocalPortInUse` — free port | Returns `false` on unused port |
| `isLocalPortInUse` — used port | Returns `true` on active server |
| `isHealthy` — healthy server | Returns `true` when `/health` returns 200 |
| `isHealthy` — no /health endpoint | Returns `false` when `/health` returns 404 |
| `isHealthy` — timeout | Returns `false` when server never responds |
| `queryModels` — model list | Parses `/v1/models` response correctly |
| `queryModels` — non-2xx response | Throws error |

### `tests/unit/bash-integration.test.ts` — 2 tests, 2 `expect()` calls

Integrates the **bash test suite** into the Bun test runner so `bun test`
exercises both languages in one command.

| Test | Description | Gated? |
|------|-------------|--------|
| `Bash test suite — unit only` | Runs `bash tests/bash/run.sh unit`, asserts output matches `/0 failed/` | Always runs |
| `Bash test suite — sandboxed only` | Runs `bash tests/bash/run.sh sandboxed`, asserts output matches `/0 failed/` | Skipped unless `RUN_LONG_TESTS=true` |

Both `it()` calls carry an explicit timeout (`30_000`ms / `300_000`ms)
matching their `execSync` child-process timeout — see the bugfix note above.

### `tests/integration/CLI.lifecycle.test.ts` — 11 tests, 22 `expect()` calls

Full lifecycle tests using a **mock backend** that records calls but never
touches the network. Tests the `Backend` contract end-to-end.

#### MockRemoteOps

A `RemoteOps` implementation that records every call (`runRemote`, `copyFile`,
`copyDirectory`, `checkSSH`, `spawnTunnel`) and holds a map of lockfile state
injected by `setLockfile()`. `runRemote()` returns canned responses for known
commands (`ivllm-status.sh`, `ivllm-cancel.sh`, `ivllm-serve.sh`).

#### Module Mocking

The test intercepts the `backend-factory.ts` module via `mock.module()`:

```typescript
mock.module('../../src/backends/backend-factory.ts', () => {
    return {
        getBackend: (creds: Credentials) => {
            return new TestBackend(creds);
        },
    };
});
```

`getBackend()` is synchronous (see `design/backend-contract.md` and the
`Backend.ts` / `backend-factory.ts` split — the factory was extracted into
its own file with eager imports to break a circular dependency between
`Backend.ts` and `IsambardBareMetalBackend.ts`; the mock stays sync to match).
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
| `lifecycle helpers > isRunning returns true/false` | State check works |
| `lifecycle helpers > isStartable returns true for stopped job` | State check works |
| `lifecycle helpers > isStarting returns true for pending job` | State check works |
| `bootstrap > calls checkSSH on first use` | SSH check fires on setup |
| `MockRemoteOps — command recording` | All 5 operation types recorded in order |

---

## Bash Tests

All sandboxed bash tests run inside the bubblewrap sandbox
(`tests/bash/lib/sandbox.sh`); `tests/bash/unit/` runs on the bare host (pure
logic, no sandbox needed). The sandbox provides real `jq`, `yq`, and mocked
SLURM/vLLM/HuggingFace commands (shims in `tests/bash/shims/`).

### `tests/bash/unit/test-utils.sh` — 12 assertions

Pure bash logic tests — no sandbox needed. Tests helper functions from
`utils.sh` including `get_max_job_time()`, `select_closest_version()`,
`revSemverSort()`, semver comparison (`semver_lt`, `semver_gte`).

### `tests/bash/sandboxed/test-lockfile.sh` — 20 assertions

Tests the **lockfile (status.json) state machine**: creation, status transitions,
and cancel/request logic. All functions work on a real JSON file written to disk
by the sandboxed process tree.

| Test | What it verifies |
|------|-----------------|
| `create_pending_basic` | Port valid, timestamps ISO-8601, fields round-trip |
| `create_pending_duplicate` | Duplicate creation fails (idempotency guard) |
| `create_pending_default_timeout` | Default timeout is 30 minutes |
| `update_initialise_basic` | vllmPid, slurmJobId, hostname set |
| `update_initialise_does_not_create_log` | Log file only created by SLURM output redirection |
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
| `get_job_status_setting` | Fields round-trip correctly; **missing field returns empty string** (not the literal `"null"` — this is `get_job_status_setting`'s documented contract in `utils.sh`; the test was updated to match after a jq `null`→`""` conversion was added) |
| `full_lifecycle` | Happy path: pending → initialise → running → stopped |
| `lifecycle_cancel` | Cancel + reason + stop |
| `lifecycle_fail_during_startup` | Failed during startup (exit code in status.json) |

### `tests/bash/sandboxed/test-cache.sh` — 5 assertions

Tests **JIT cache save/restore** logic: directory creation, permissions, node
gating (only head node saves), and cleanup.

| Test | What it verifies |
|------|-----------------|
| `cache_save_restore` | Files copied to correct location, permissions 0750 |
| `cache_restore_missing` | No errors when cache dir is empty |
| `cache_save_empty` | Graceful handling of empty directory |
| `cache_permissions` | Directory 0750, files 0640 |
| `cache_worker_node_does_not_save` | Worker skips cache save |

### `tests/bash/sandboxed/test-config.sh` — 6 assertions

Tests **vllm.yaml config reading** against the real `yq 3.4.1` binary.

| Test | What it verifies |
|------|-----------------|
| `get_job_config_setting_model` | `.model` path resolves correctly |
| `get_job_config_setting_idle_timeout` | Integer value preserved |
| `get_job_config_setting_tensor_parallel` | Integer value preserved |
| `resolve_stripped_job_config_strips_env_and_metadata` | `yq d` removes both blocks |
| `get_job_config_exports_produces_export_lines` | env block → `export KEY=VALUE` lines |
| `get_job_config_exports_empty_env_block` | No export lines for empty env |

### `tests/bash/sandboxed/test-vllm-env.sh` — 4 assertions

Tests **common-env.sh** (NVHPC/CUDA/compiler setup) and **vllm-env.sh**
(NCCL/Slingshot/vLLM tuning) sourcing.

| Test | What it verifies |
|------|-----------------|
| `common_env_sources` | No errors on source |
| `common_env_vars_set` | NVHPC_ROOT, CUDA paths, CC/CXX, LD_LIBRARY_PATH |
| `common_env_missing_nvhpc_falls_through_empty` | resolve_nvhpc_root() writes to stderr |
| `vllm_env_sources_and_sets_vars` | NCCL_CROSS_NIC, FI_PROVIDER, VLLM vars |

### `tests/bash/sandboxed/test-monitor-head.sh` — 5 assertions

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

### `tests/bash/sandboxed/test-monitor-startup.sh` — 4 assertions

Tests **`monitor_startup()`** — the foreground monitor. Starts a real mock vLLM
HTTP server (same as `tests/bash/shims/vllm`) that handles `/health` and
`/v1/chat/completions`.

| Test | What it verifies |
|------|-----------------|
| `startup_sends_health_and_warms_up` | Lockfile transitions to `running` |
| `startup_health_then_succeeds` | Monitor polls and succeeds after delay |
| `startup_non_head_node` | Worker returns immediately, status unchanged |
| `startup_wrong_status` | Returns 1 when already `running` |

### `tests/bash/sandboxed/test-monitor-worker.sh` — 5 assertions

Tests **`monitor_worker()`** — the background monitor on worker nodes.

| Test | What it verifies |
|------|-----------------|
| `worker_monitor_rejects_head_node` | Returns 1 for head node (SLURM_NODEID=0) |
| `worker_monitor_missing_lockfile` | Returns 1, kills worker pid |
| `worker_monitor_stays_alive_running` | Monitor keeps running |
| `worker_monitor_shuts_down_on_cancel` | Worker killed, localdir cleaned |
| `worker_monitor_shuts_down_on_failed` | Worker killed, localdir cleaned |

### `tests/bash/sandboxed/test-exit-trap.sh` — 7 assertions

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

### `tests/bash/sandboxed/test-login-handoff.sh` — 7 assertions

Tests the **login-node wrapper scripts** in the `login` profile sandbox.
Scripts run end-to-end calling shimmed `sbatch`/`srun`/`scancel`/`squeue`/`hf`.
After each script exits, the test reads `/work/calls.log` and asserts the
correct commands were called.

`ivllm-serve.sh` now sources `ivllm-get-model.sh` inline (synchronously, on
the login node, via `srun`) **before** submitting the `sbatch` job — the
model download is no longer a separate scheduled job (the old
`slurm-hf-download.sh` was deleted). Two tests cover both branches of that
inline check:

| Test | What it verifies |
|------|-----------------|
| `login_serves_with_minimal_config` | Model pre-cached (`MOCK_HF_CACHED_MODELS` set) → `hf cache` called, `hf download` NOT called, sbatch still reached with correct job name/partition/script |
| `login_serves_downloads_uncached_model` | Model not cached, `HF_TOKEN` set → `srun ... hf download <model>` called, sbatch still reached |
| `login_cancels_existing` | request_cancel sets status to cancel, scancel NOT called |
| `login_cancels_missing_job` | Fails with error, scancel NOT called |
| `login_shows_status` | Returns JSON with correct jobName |
| `login_setup_runs` | srun called with setup script, version flag |
| `login_force_cancel` | scancel called with correct slurmJobId |

**Assertion note:** `assert_shim_not_called(tool)` only takes a tool name —
it asserts that tool was never invoked *at all*, not that a specific
subcommand wasn't called. Since these tests need `hf cache` to be called but
`hf download` NOT to be called, the "not called" check greps
`$IVLLM_TEST_CALL_LOG` directly for the specific subcommand instead of using
the helper.

---

## Test Evolution Summary

### What changed in recent commits

| Change | Impact |
|--------|--------|
| `semver.ts` deleted | Semver logic lives only in bash `utils.sh` |
| `RemoteOps.mock.test.ts` deleted | Mock moved into `CLI.lifecycle.test.ts` as `TestRemoteOps` |
| `CLI.lifecycle.test.ts` rewritten | Uses `mock.module()` to intercept `getBackend()`, `TestBackend` + `TestRemoteOps` pattern |
| `Backend.ts` / `backend-factory.ts` split | `getBackend()` + registry extracted into `backend-factory.ts` (eager imports) to break a circular dependency between `Backend.ts` and `IsambardBareMetalBackend.ts` |
| `bash-integration.test.ts` restructured | Sandboxed tests skip-by-default (`RUN_LONG_TESTS`); **fixed missing per-`it()` timeout** so `RUN_LONG_TESTS=true bun test` no longer fails on its own 5s default timeout |
| `slurm-hf-download.sh` deleted | Model download folded into `ivllm-get-model.sh`, called inline (synchronously, via `srun`) from `ivllm-serve.sh` before `sbatch` submission |
| `test-login-handoff.sh` | Updated for inline download flow (`MOCK_HF_CACHED_MODELS` env var); added `login_serves_downloads_uncached_model` for the uncached branch |
| `test-lockfile.sh` | `get_job_status_setting` missing-field assertion changed from `"null"` to `""` to match `utils.sh`'s null→empty-string conversion |
| `test-exit-trap.sh` | `tidy_up_no_slurm_cancel` test no longer present (7 assertions, not 8) |
| ADR-116 (chmod fix) | All `chmod -R` calls in `utils.sh` replaced with non-recursive `chmod g+rwX` on directory creation (Lustre/GPFS performance + multi-user permission errors) |
| `run_head_vllm.sh` / `run_worker_vllm.sh` | Fixed undefined `$nodeRank` (now `0` / `$IVLLM_NODE_RANK`) and `startRank`/`localDp` ordering bug; added multi-node data-parallel args |
