# Bash Test Scripts — isambard-vllm

This document describes what each bash test script does, what it verifies,
and how it exercises the codebase. All scripts run inside the bwrap sandbox
(`tests/bash/lib/sandbox.sh`) unless noted otherwise.

---

## Layer 1: Lockfile State Machine

### `tests/bash/sandboxed/test-lockfile.sh`

Tests the **lockfile (status.json) state machine**: creation, status transitions,
and cancel/request logic. All functions work on a real JSON file written to disk
by the sandboxed process tree — no mock JSON, no in-memory fakes.

| Test | What it does | What it verifies |
|------|-------------|-----------------|
| `create_pending_basic` | Creates a `pending` job with model + timeout | Port is valid, timestamps are ISO-8601, fields round-trip |
| `create_pending_duplicate` | Creates then recreates same job | Duplicate creation fails (idempotency guard) |
| `create_pending_default_timeout` | Creates job without explicit timeout | Default timeout is 30 minutes |
| `update_initialise_basic` | `pending` → `initialising` with pid | vllmPid, slurmJobId, hostname set |
| `update_initialise_does_not_create_log` | Only updates lockfile, not log file | Log file only created by SLURM output redirection |
| `update_initialise_worker_only` | Worker node (SLURM_NODEID=1) calls update | Worker is ignored, lockfile stays `pending` |
| `update_running` | `initialising` → `running` | Status transition works |
| `update_running_worker_only` | Worker calls running update | Worker ignored, lockfile stays `pending` |
| `clean_shutdown` | Full `running` → `stopped` lifecycle | Exit code 0, stopTime timestamp |
| `unclean_shutdown` | `running` → `failed` with reason+code | Reason + exit code captured |
| `request_cancel` | `running` → `cancel` | Status transitions to cancel |
| `request_cancel_no_lockfile` | Cancel non-existent job | Fails gracefully |
| `request_cancel_from_worker` | Cancel from worker node | Cancel works from any node |
| `is_status` | Checks status in multiple states | Pending → initialising → running → cancel |
| `is_status_missing_lockfile` | is_status on ghost job | Returns false for missing lockfile |
| `update_reason` | Sets reason on any status | Reason field stored without changing status |
| `get_job_status_setting` | Reads various fields via `.` path | Fields round-trip correctly |
| `full_lifecycle` | pending → initialise → running → stopped | Full happy-path lifecycle |
| `lifecycle_cancel` | pending → initialise → running → cancel → stopped | Cancel + reason + stop |
| `lifecycle_fail_during_startup` | pending → initialise → failed | Failed during startup (exit code in status.json) |

---

## Layer 1: Cache Operations

### `tests/bash/sandboxed/test-cache.sh`

Tests **JIT cache save/restore** logic: directory creation, permissions, node
gating (only head node saves), and cleanup.

| Test | What it does | What it verifies |
|------|-------------|-----------------|
| `cache_save_restore` | Fake model files → save → restore | Files copied to correct location, permissions 0750 |
| `cache_restore_missing` | Restore when cache dir is empty | No errors, model files regenerated |
| `cache_save_empty` | Save with no model files | Graceful handling of empty directory |
| `cache_permissions` | Save creates correct permissions | Directory 0750, files 0640 |
| `cache_worker_node_does_not_save` | Worker node (SLURM_NODEID=1) calls save | Worker skips cache save |

---

## Layer 1: Configuration Parsing (yq v3)

### `tests/bash/sandboxed/test-config.sh`

Tests **vllm.yaml config reading** against the real `yq 3.4.1` binary
(installed on the HPC). Previously caught issues 7–9 (argument order, v3 vs
v4 syntax). All tests now green.

| Test | What it does | What it verifies |
|------|-------------|-----------------|
| `get_job_config_setting_model` | Reads `.model` from `minimal.yaml` | Path resolves correctly |
| `get_job_config_setting_idle_timeout` | Reads `.idle-timeout` | Integer value preserved |
| `get_job_config_setting_tensor_parallel` | Reads `.tensor-parallel-size` | Integer value preserved |
| `resolve_stripped_job_config_strips_env_and_metadata` | Creates config with env/metadata | `yq d` removes both blocks |
| `get_job_config_exports_produces_export_lines` | env block → `export KEY=VALUE` lines | Correct shell export syntax |
| `get_job_config_exports_empty_env_block` | No env block → no export lines | Graceful empty output |

---

## Layer 1: Environment Preamble

### `tests/bash/sandboxed/test-vllm-env.sh`

Tests **common-env.sh** (NVHPC/CUDA/compiler setup) and **vllm-env.sh**
(NCCL/Slingshot/vLLM tuning) sourcing.

| Test | What it does | What it verifies |
|------|-------------|-----------------|
| `common_env_sources` | Sources common-env.sh with NVHPC fixture | No errors on source |
| `common_env_vars_set` | Sources and checks key env vars | NVHPC_ROOT, CUDA paths, CC/CXX, LD_LIBRARY_PATH |
| `common_env_missing_nvhpc_falls_through_empty` | No NVHPC directory → NVHPC_ROOT empty | resolve_nvhpc_root() writes to stderr (not stdout) |
| `vllm_env_sources_and_sets_vars` | Sources vllm-env.sh | NCCL_CROSS_NIC, FI_PROVIDER, VLLM_ENGINE_ITERATION_TIMEOUT_S |

---

## NEW: Monitor Head (Background)

### `tests/bash/sandboxed/test-monitor-head.sh`

Tests **`monitor_head()`** — the background monitor loop that runs on the head
node for the entire job lifetime. Uses real background processes (fake vLLM
pid, fake parent pid) so that `kill -0` liveness checks exercise real
subprocess semantics.

| Test | What it does | What it verifies |
|------|-------------|-----------------|
| `monitor_head_detects_cancel` | Background monitor + fake vLLM → cancel request | status → cancel, reason "user cancel", parent killed |
| `monitor_head_detects_vllm_process_death` | Kill fake vLLM mid-run | status → failed, reason "lost contact with vLLM process" |
| `monitor_head_detects_lockfile_deletion` | Delete lockfile mid-run | Monitor exits, parent killed, localdir cleaned |
| `monitor_head_idle_timeout_shuts_down` | Idle log → no recent requests | status → failed, reason "idle timeout" |
| `monitor_head_active_traffic_prevents_idle_timeout` | Log with recent API request | Monitor does NOT shut down (job stays running) |

---

## NEW: Monitor Startup (Foreground)

### `tests/bash/sandboxed/test-monitor-startup.sh`

Tests **`monitor_startup()`** — the foreground monitor that runs on the head
node during initialisation. Blocks until vLLM responds to `/health`, sends a
warmup request, saves the JIT cache, and transitions status to `running`.

| Test | What it does | What it verifies |
|------|-------------|-----------------|
| `startup_sends_health_and_warms_up` | Mock vLLM HTTP server responds immediately to `/health` and warmup request | Lockfile transitions `pending` → `initialising` → `running` |
| `startup_health_then_succeeds` | Mock vLLM server starts after 1s delay | Monitor polls and eventually transitions to `running` |
| `startup_warmup_fails` | Mock vLLM `/health` works but warmup `/v1/chat/completions` fails | Lockfile status → `failed`, reason "vLLM warmup failed", return code 1 |
| `startup_non_head_node` | SLURM_NODEID=1 (worker) calls monitor_startup | Returns immediately, lockfile status unchanged |
| `startup_wrong_status` | Lockfile already in `running` status | Returns 1, no state change |

**How it works:** The test starts a real mock vLLM HTTP server (same as
`tests/bash/shims/vllm`) in the background, sources `utils.sh`, calls
`monitor_startup()`, then checks the lockfile. The mock server handles
`/health` and `/v1/chat/completions` requests and can be configured to fail
after a delay or always fail.

---

## NEW: Monitor Worker (Background)

### `tests/bash/sandboxed/test-monitor-worker.sh`

Tests **`monitor_worker()`** — the background monitor that runs on worker
nodes (node ID > 0) for multi-node jobs. Watches the lockfile and shuts
down the local vLLM process if the job is no longer running.

| Test | What it does | What it verifies |
|------|-------------|-----------------|
| `worker_monitor_rejects_head_node` | SLURM_NODEID=0 calls monitor_worker | Returns 1, kills worker pid, logs error |
| `worker_monitor_missing_lockfile` | No lockfile exists on startup | Returns 1, kills worker pid |
| `worker_monitor_stays_alive_running` | Lockfile exists, status=running | Monitor keeps running without error |
| `worker_monitor_shuts_down_on_cancel` | Monitor running → cancel status set | Worker process killed, localdir cleaned, status logged |
| `worker_monitor_shuts_down_on_failed` | Monitor running → failed status set | Worker process killed, localdir cleaned |

**How it works:** The test sources `utils.sh`, sets SLURM_NODEID (0 for head
reject, 1+ for worker), creates/doesn't create lockfile with appropriate
status, backgrounds a fake worker pid, calls `monitor_worker()`, then
checks the result. Uses real background processes so `kill -0` checks
exercise real subprocess semantics.

---

## NEW: Exit Traps and Signal Handling

### `tests/bash/sandboxed/test-exit-trap.sh`

Tests **`tidy_up()`** — the exit-trap handler that runs on all nodes when a
job shuts down. Kills the vLLM process (if alive), updates the lockfile
based on exit code, and cancels the SLURM job.

| Test | What it does | What it verifies |
|------|-------------|-----------------|
| `tidy_up_200_triggers_slurm_timeout` | tidies up with exit_code=200 (SIGUSR1) | status → stopped, reason "SLURM timeout", lockfile updated |
| `tidy_up_201_triggers_user_cancel` | tidies up with exit_code=201 (SIGUSR2) | status → stopped, reason "user cancel or idle timeout" |
| `tidy_up_0_normal_shutdown` | tidies up with exit_code=0 (clean exit) | status unchanged, vLLM process killed if alive |
| `tidy_up_nonzero_crash_runtime` | tidies up with exit_code=42 while running | status → failed, reason "crashed during inference" |
| `tidy_up_nonzero_crash_startup` | tidies up with exit_code=1 while initialising | status → failed, reason "failed to start" |
| `tidy_up_kills_vllm_process` | vllmPid exists and process alive | Sends SIGTERM, then SIGKILL after 2s if still alive |
| `tidy_up_cancels_slurm_job` | slurmJobId not null | scancel shim called with correct job id |
| `tidy_up_no_slurm_cancel` | slurmJobId = "null" | scancel NOT called |

**How it works:** Creates a lockfile with appropriate status, starts a real
background process as the fake vLLM pid, then calls `tidy_up()` directly.
Uses real process signals and shims so the exit-trap semantics are tested
end-to-end.

---

## NEW: Login-Node Handoff

### `tests/bash/sandboxed/test-login-handoff.sh`

Tests the **login-node wrapper scripts** (`ivllm-serve.sh`, `ivllm-cancel.sh`,
`ivllm-status.sh`) in the `login` profile sandbox. These scripts run on the
login node and call `sbatch`/`srun`/`scancel`/`squeue` — all of which are
shims that record calls to `/work/calls.log` but do not execute the wrapped
commands. The test assertions check the calls.log for correct arguments.

| Test | What it does | What it verifies |
|------|-------------|------------------|
| `login_serves_with_minimal_config` | `ivllm-serve.sh -j serve-job` with a stopped lockfile | sbatch called with correct job name, partition, gpus, mem |
| `login_cancels_existing` | `ivllm-cancel.sh -j cancel-job` (non-force) | request_cancel sets status to cancel, scancel NOT called |
| `login_cancels_missing_job` | `ivllm-cancel.sh -j nonexistent` | Fails with error, scancel NOT called |
| `login_shows_status` | `ivllm-status.sh -j status-job` | Returns JSON with correct jobName |
| `login_setup_runs` | `ivllm-setup.sh -v 0.8.0` | srun called with setup script, version flag |
| `login_force_cancel` | `ivllm-cancel.sh -j fcancel-job -f` | scancel called with correct slurmJobId |

**How it works:** Each test runs inside a bwrap `login` profile sandbox
(no SLURM_* env vars). The login wrapper scripts execute end-to-end, calling
the shimmed `sbatch`/`srun`/`scancel`/`squeue`. After the script exits, the
test reads `/work/calls.log` and asserts the correct commands were called
with the correct arguments.
