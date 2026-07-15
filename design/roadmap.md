# Migration Roadmap — isambard-vllm v3

This roadmap describes the phased migration from the v2 session-owner
architecture to the v3 self-managed architecture.

Each phase is a standalone milestone with its own test criteria. Phases
are designed to be delivered incrementally — the system remains functional
at every stage.

**Current version**: 2.9.0  
**Target version**: 3.0.0

---

## Phase M1: Bash Framework Foundation

**Goal**: Create the self-contained bash library that will manage vLLM jobs
on the HPC side. No CLI changes yet — the existing v2 commands continue
to work unchanged. The bash library is deployed alongside existing code.

**Why first**: Everything else depends on this. It's the lowest-risk phase
because it adds files without modifying any existing code. The prototype
already provides a starting point (`design/prototype/prototype.sh`).

### Files to create

| File | Source | Description |
|------|--------|-------------|
| `src/templates/lib/utils.sh` | `design/prototype/prototype.sh` + `templates/inference.ts` | Lockfile management, cache functions, monitor triad, exit trap, diagnostics |
| `src/templates/lib/preamble.sh` | `templates/inference.ts:renderNVHPCPreamble()` | NVHPC/NCCL/Slingshot environment variables |
| `src/templates/lib/hf.sh` | New | Model download via `srun` on interactive partition |
| `src/templates/lib/vllm_logs.json` | `design/prototype/vllm_logs.json` | vLLM logging config for timestamped access logs |
| `src/templates/lib/README.md` | New | Documentation for the bash framework |
| `tests/templates/lib/test-utils.sh` | `design/prototype/test-vllm.sh` | Mock `srun`, `scancel`, `vllm` for testing bash functions |
| `tests/templates/lib/` | New | Test scripts for each bash module |

### What to extract from inference.ts

The current `src/templates/inference.ts` (1,176 lines) contains:

- `renderNVHPCPreamble()` → `lib/preamble.sh` — 120 lines of hard-won NCCL/Slingshot tuning
- `renderExitDiagnostics()` → `lib/utils.sh` — SLURM accounting persist
- `renderWorkDirSetup()` → `lib/utils.sh` — JIT cache restore, workdir creation
- `renderMonitor()` → `lib/utils.sh` — memory/cache monitoring
- `renderExitTrap()` → `lib/utils.sh` — `tidy_up` and signal handling
- `renderHealthCheckAndWait()` → `lib/utils.sh` — health check + cache save
- `renderSingleNodePayload()` / `renderMultiNodePayload()` → thin SLURM scripts that `source` libs

### From prototype.sh

The `design/prototype/prototype.sh` already has:

- Lockfile functions: `create_status_pending`, `update_status_initialise`,
  `update_status_running`, `update_status_clean_shutdown`,
  `update_status_unclean_shutdown`, `request_cancel`, `is_status`
- Monitor functions: `monitor_startup`, `monitor_head`, `monitor_worker`
- Cache functions: `restore_cache`, `save_cache`
- Shutdown: `tidy_up` (exit trap), `setup_traps`
- Path helpers: `resolve_localdir`, `resolve_location`, `resolve_lockfile`,
  `resolve_logfile`, `resolve_setting`, `resolve_cachetar`
- Utility: `report_memory`, `clear_localdir`

### Test criteria

- [ ] Each bash function has a corresponding test in the mock harness
- [ ] `monitor_startup` works with mock vLLM: detects `/health`, runs warmup, saves cache
- [ ] `monitor_head` detects `cancel` in lockfile and triggers shutdown
- [ ] `monitor_head` detects idle timeout by scanning mock access logs
- [ ] `monitor_worker` detects non-running lockfile status and shuts down
- [ ] `tidy_up` handles all exit codes correctly (0, 200=SIGUSR1, 201=SIGUSR2, other)
- [ ] Cache save/restore round-trips correctly
- [ ] `preamble.sh` sourced without errors on GH200

### What remains unchanged

All TypeScript code. All existing tests continue to pass.

### Done when

- [-] `bash tests/templates/lib/` passes all tests
- [ ] The bash framework is deployed on Isambard alongside `ivllm setup`
- [ ] Manual test: SSH to login node, `source lib/preamble.sh && source lib/utils.sh`,
      run through a mock lifecycle end-to-end
- [ ] Committed, version bumped

---

## Phase M2: New Lockfile + Command Interface

**Goal**: Add `ivllm connect` and `ivllm cancel` commands. Introduce the
new `status.json` lockfile schema. Remove old `ivllm start`/`interactive`/`stop`.

**Why this phase**: The new commands exist but still drive the old
sbatch/srun infrastructure. This is a CLI-only change — the compute-side
behaviour is unchanged. It lets users start using the new interface before
the backend switches over.

### Files to create

| File | Source | Description |
|------|--------|-------------|
| `src/commands/connect.ts` | New | Unified connect command (replaces start + interactive) |
| `src/commands/cancel.ts` | New | Cancel command (replaces stop) |

### Files to modify

| File | Changes |
|------|---------|
| `src/index.ts` | Add `connect` and `cancel` command routes; remove `start`, `interactive`, `stop` |
| `src/job.ts` | Add `makeV3Paths()` for new `$PROJECTDIR/engine/jobs/<job>/` structure |
| `src/job.ts` | Add `parseV3Lockfile()` for new `status.json` schema |
| `src/types.ts` | Add `LockfileState` type, `JobStatusV3` union, `LockfileV3` interface |
| `src/vllm-config.ts` | Add `idleTimeout` field, metadata block (`version`, `author`, `lifecycle`, `targetVllmVersion` replacing `minVllmVersion`). Metadata is stripped before job runs but preserved in the job directory. |
| `src/commands/list.ts` | Read new `status.json` format, display richer info |
| `src/commands/status.ts` | Update for new lockfile path |

### vllm.yaml metadata format

The vllm.yaml config file gains an optional `metadata:` block at the top
level. This block is stripped before being passed to vLLM but is preserved
in the job directory for debugging and provenance.

```yaml
metadata:
  version: "1.0"
  author: "alice@example.com"
  lifecycle: maturing        # experimental | maturing | stable | deprecated
  target-vllm-version: "0.22.0"  # replaces min-vllm-version
  description: "Qwen3.6 35B A3B FP8 on 4 GH200 nodes"

model: Qwen/Qwen3.6-35B-A3B-FP8
tensor-parallel-size: 4
pipeline-parallel-size: 4
# ... remaining vLLM args
idle-timeout: 30
```

**Stripping**: The metadata block is parsed by `parseVllmConfig()` and
removed from the dict passed to vLLM. The original (raw) config file is
always uploaded to the job directory alongside the stripped version.

**Storage**: Both `vllm.yaml` (raw, with metadata) and `vllm.stripped.yaml`
(without metadata) are stored in `$PROJECTDIR/engine/jobs/<job>/`.

**Alternatives considered**:
- YAML comments for metadata: harder to parse in both TypeScript and bash.
  A `metadata:` key is cleaner and survives YAML round-trips.
- Metadata-only file (`job.meta.yaml`): adds a file to manage. Inline
  metadata in the config is simpler and stays with the config.

### Files to remove

| File | Replacement |
|------|-------------|
| `src/commands/start.ts` | `src/commands/connect.ts` |
| `src/commands/interactive.ts` | `src/commands/connect.ts` |
| `src/commands/stop.ts` | `src/commands/cancel.ts` |
| `src/monitors.ts` | Logic absorbed into `connect.ts` + bash |

### Test criteria

- [ ] `ivllm connect <job> --config <yaml>` creates `status.json` with `pending`
- [ ] `ivllm connect <job>` reads existing `status.json`, shows current state
- [ ] `ivllm connect <job>` with running job → establishes tunnel
- [ ] `ivllm cancel <job>` writes `cancel` to `status.json`
- [ ] `ivllm cancel <job> --force` runs `scancel` + updates lockfile
- [ ] `ivllm list` shows all jobs with status, model, age
- [ ] All v3-only tests pass in `--mock` mode
- [ ] Old `--dry-run` mode still works for `connect`
- [ ] All existing v2 tests for config, semver, remote-ops still pass

### What remains unchanged

bash framework (Phase M1), `ivllm setup`, `ivllm agent`, `ivllm config`.

### Done when

- [ ] `ivllm connect --help` shows proper usage
- [ ] `ivllm cancel --help` shows proper usage
- [ ] `ivllm connect qwen2 --config examples/qwen2.5-instruct.yaml --dry-run` works
- [ ] End-to-end: `ivllm connect qwen2 --config examples/qwen2.5-instruct.yaml --mock`
      runs through full lifecycle on mock vLLM
- [ ] All tests pass: `bun test`
- [ ] `ivllm start`, `ivllm interactive`, `ivllm stop` removed from CLI
- [ ] Committed, version bumped

---

## Phase M3: Self-Managed Lifecycle

**Goal**: The SLURM scripts now use the bash framework (from Phase M1)
instead of the old TypeScript templates. The compute node manages its own
lifecycle. The CLI is a thin monitor that watches the lockfile.

**Why this phase**: This is the core architectural change. After this phase,
the compute node can run without the client.

### Files to modify

| File | Changes |
|------|---------|
| `src/templates/inference.ts` (1,176 lines) | **Drastically reduced** — generate thin wrappers that `source lib/utils.sh` and `source lib/preamble.sh` |
| `src/session-helper.ts` (~500 lines) | **Removed** — logic moves to `connect.ts` + bash |
| `src/commands/connect.ts` | Replace `runInferenceSession()` with lockfile-monitoring loop |
| `src/slurm.ts` | Simplify `submitJob`/`runInteractive` — no longer need to pass monitors |
| `src/job.ts` | Update `makePaths` to new `$PROJECTDIR/engine/` structure |

### How connect.ts now works

Instead of the old 10-step `runInferenceSession`:

1. SSH pre-flight check
2. Read/parse YAML config
3. Create `status.json` with `pending`
4. Upload config + generate thin `slurm.sh` (sources libs)
5. Download model (via `hf.sh` or inline)
6. `sbatch slurm.sh` (or `srun` for `--interactive`)
7. Poll `status.json`:
   - `pending` → print queue status
   - `initialising` → tail logs
   - `running` → establish tunnel, print endpoint
   - `failed`/`stopped` → print diagnostics, exit
8. Stay in foreground (optional with `--detach` flag)
9. On Ctrl+C/exit → **do NOT kill the job**. Just close the tunnel and exit.

### What the new SLURM script looks like

**Single model per script** — each model gets its own `sbatch` job. The
nested subshell pattern in the prototype was for running multiple models
within one SLURM allocation, but independent jobs are simpler, more
resilient, and align with ADR-113 (each model is an independent job).

**Multi-node** — two approaches coexist for now (see Phase M7.5):

- **MP** (interactive default): script runs once per node via `srun`,
  `SLURM_NODEID` determines head vs worker. Simple, no extra deps.
- **Ray** (batch default): script runs once on head node, uses `srun`
  to start workers. More resilient to node failure.

**Log routing** — each node writes its own log file (`vllm.<NODEID>.log`).
The head node log is the one `monitor_head` reads; worker logs are for
diagnostics. This works for both MP and Ray backends.

```bash
#!/bin/bash
#SBATCH --job-name=qwen2
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=4:00:00

source $PROJECTDIR/engine/lib/preamble.sh
source $PROJECTDIR/engine/lib/utils.sh

WORK_DIR="$PROJECTDIR/engine/jobs/qwen2"
cd "$WORK_DIR"

source $PROJECTDIR/engine/vllm/0.22.0/bin/activate

# Lockfile already created by CLI with "pending"
# Overwrite with SLURM details
update_status_initialise "$JOBNAME" "$VLLM_PID"

srun --export=ALL --cpu-bind=cores vllm serve \
  --config vllm.yaml \
  --host 0.0.0.0 --port 8000 &
VLLM_PID=$!

setup_traps "$JOBNAME"
restore_cache "$JOBNAME"
monitor_startup "$JOBNAME" "$VLLM_PID"
monitor_head "$JOBNAME" "$VLLM_PID" &
wait $VLLM_PID
tidy_up "$JOBNAME" $?
```

This is ~30 lines instead of the current 300+ line template.

### Test criteria

- [ ] New SLURM script template results in <50 lines per script
- [ ] `--dry-run` with new template generates valid bash (check with `bash -n`)
- [ ] `ivllm connect --mock` runs through full lifecycle with new scripts
- [ ] Old `--interactive` flag still works (uses `srun` wrapper)
- [ ] vLLM starts, becomes healthy, status transitions work
- [ ] Ctrl+C on connect exits without killing the SLURM job
- [ ] `ivllm list` shows job still running after disconnect
- [ ] `ivllm cancel` triggers graceful shutdown (lockfile → `cancel` → `stopped`)
- [ ] `ivllm cancel --force` hard-kills via scancel
- [ ] All tests pass

### What remains unchanged

`ivllm setup`, `ivllm agent`, `ivllm config`

### Multi-node approach (both backends kept)

Both multi-node backends are retained for now:
- **MP** (interactive default): used when `--nnodes` is set without
  `--distributed-executor-backend`. Script runs once per node via `srun`,
  `SLURM_NODEID` determines head vs worker.
- **Ray** (batch default): used with explicit `--distributed-executor-backend ray`.
  Script runs once on head node, uses `srun` to start workers.

A systematic comparison (MP vs Ray on Slingshot) is deferred to Phase M7.5
at the end of the roadmap. The comparative evaluation is expensive and risks
a tweak-startup-crash loop. Both approaches work — keep them both for now.

Clean-up tasks that can be done now without evaluation:
- [ ] Interactive template: remove dead Ray env vars (`RAY_PORT`,
      `RAY_OBJECT_STORE_MEMORY`, etc.) that are set but never consumed
- [ ] Each node writes its own log (`vllm.<NODEID>.log`) for both backends

### Done when

- [ ] End-to-end test on Isambard with a real multi-node model (GLM-5.2 and QWEN3.5-397):
      connect → monitor → cancel, verify clean shutdown
- [ ] Disconnect test: connect → Ctrl+C → reconnect → verify tunnel works
- [ ] Multi-node test: connect with a 2-node config → verify Ray starts
- [ ] All old session-owner code removed
- [ ] Committed, version bumped

---

## Phase M4: Detach Mode + Background Monitoring

**Goal**: The CLI can truly detach — it exits cleanly and the job keeps
running. The job can be reconnected to from any client. Add optional
background monitoring via a lightweight daemon.

**Why this phase**: This is the key UX improvement. Users can start a job,
close their laptop, come back later, and reconnect.

### Files to modify

| File | Changes |
|------|---------|
| `src/commands/connect.ts` | Add `--detach` flag: establish tunnel, print endpoint, exit immediately |
| `src/commands/connect.ts` | Default behaviour: stay in foreground, but don't kill job on exit |
| `src/commands/connect.ts` | Monitor mode: optional `--monitor` flag that tails logs and shows status updates |
| `src/commands/list.ts` | Show time remaining (SLURM end time) if running |

### Behaviour matrix

| User action | Job state | What happens |
|-------------|-----------|-------------|
| `connect <job>` | Running | Establish tunnel, print endpoint, stay in foreground |
| `connect <job>` | Stopped | Restart job, monitor startup, establish tunnel |
| `connect <job>` | Initialising | Tail logs, wait for running, establish tunnel |
| `connect <job> --detach` | Any | Submit/attach, print endpoint, exit immediately |
| `connect <job> --monitor` | Any | Stay in foreground, show periodic status + log updates |
| Ctrl+C | Attached | Close tunnel, exit. Job keeps running. |
| Ctrl+C, then reconnect | Running | New tunnel, same job. |

### Test criteria

- [ ] `--detach` exits immediately after printing endpoint
- [ ] Reconnecting after detach shows correct job state
- [ ] Multiple clients can connect simultaneously (multi-tunnel test)
- [ ] Job survives client crash (SIGKILL the connect process)
- [ ] `--monitor` shows periodic cache size and log tail
- [ ] All tests pass

### Done when

- [ ] Manual test: connect with `--detach`, close terminal, reopen, reconnect
- [ ] Manual test: connect on machine A, connect on machine B (same user)
- [ ] Idle timeout kills job after inactivity (if configured — relies on Phase M5)
- [ ] Committed, version bumped

---

## Phase M5: Idle Timeout

**Goal**: Jobs shut themselves down after a configurable period of inactivity.
This prevents abandoned jobs from wasting GPU hours.

**Why this phase**: With detach/reattach working (Phase M4), idle timeout
becomes essential — users will inevitably forget about running jobs.

### Files to modify

| File | Changes |
|------|---------|
| `src/templates/lib/utils.sh` | `monitor_head` already has idle timeout logic — needs hooking up |
| `src/templates/lib/preamble.sh` | Set `VLLM_LOGGING_CONFIG_PATH` to `vllm_logs.json` |
| `src/templates/lib/vllm_logs.json` | Deploy alongside libs (verify format with latest vLLM) |
| `src/vllm-config.ts` | Add `idleTimeout` field (default 30, -1 = never) |
| `src/job.ts` | Pass `idleTimeout` into lockfile on creation |
| `src/types.ts` | Add `idleTimeout` to relevant interfaces |

### How it works

1. `idleTimeout` is set in `vllm.yaml`:
   ```yaml
   model: Qwen/Qwen2.5-7B-Instruct
   idle-timeout: 30  # minutes, default 30
   ```
2. Value is stored in `status.json` when the lockfile is created
3. `monitor_head` reads the timeout from the lockfile
4. Every 10s, it scans the last `idleTimeout` minutes of the vLLM access log
   for API request patterns: `/v1/chat/completions`, `/v1/models`, etc.
5. If no requests within the window → writes `reason: "idle timeout"`
   and sends SIGUSR2 to the vLLM parent → `tidy_up` transitions to `stopped`

The vLLM logging config (`vllm_logs.json`) adds timestamps to the uvicorn
access log format, which the monitor uses for time-window matching.

### Test criteria

- [ ] `idleTimeout: 1` causes job to shut down after 1 minute of inactivity
- [ ] Active API requests reset the idle timer
- [ ] `idleTimeout: -1` keeps job running indefinitely (no timeout)
- [ ] Default is 30 minutes
- [ ] Lockfile shows `reason: "idle timeout"` on timeout shutdown
- [ ] Monitor correctly parses vLLM log timestamps
- [ ] All tests pass

### Done when

- [ ] Manual test: start job, wait for idle timeout, verify `status.json` shows `stopped` with reason
- [ ] Manual test: start job, make API requests, verify job stays alive past idle timeout
- [ ] Manual test: `idleTimeout: -1` job never times out
- [ ] Committed, version bumped

---

## Phase M6: Multi-User Hardening

**Goal**: Multiple project members can share running jobs. Permissions,
diagnostics, and documentation are correct for team use.

**Why this phase**: The architecture supports multi-user from Phase M3
onward, but proper hardening is needed before it's reliable for a team.

### Files to modify

| File | Changes |
|------|---------|
| `src/commands/connect.ts` | `ivllm list --all` shows jobs from all users |
| `src/templates/lib/utils.sh` | Verify all `umask 0002` and `chmod g+w` calls are correct |
| `src/templates/lib/utils.sh` | Wrap `scancel` in `--uid` fallback for non-owners |
| `src/templates/lib/preamble.sh` | Verify `umask 0002` at start of every SLURM script |
| `src/templates/lib/hf.sh` | Verify shared HF cache permissions |
| `src/templates/setup.ts` | Verify `$PROJECTDIR/engine/` created with group write |

### Permissions checklist

- [ ] `$PROJECTDIR/engine/` is group-writable (`chmod -R g+w`)
- [ ] `$PROJECTDIR/engine/jobs/` is group-readable
- [ ] all `status.json` files are group-writable
- [ ] JIT cache tar.gz files are group-readable
- [ ] HF cache at `$PROJECTDIR/hf/` is group-readable
- [ ] vLLM venv at `$PROJECTDIR/engine/vllm/<version>/` is group-readable
- [ ] Log files are group-readable
- [ ] SLURM scripts are group-readable

### Documentation

- [ ] README updated with multi-user workflow
- [ ] Example: Alice starts a job, Bob connects to it
- [ ] Document that only the job owner can `scancel` (unless using `--force`)
- [ ] Document `HF_TOKEN` sharing (each user configures their own)
- [ ] Document diagnostics directory
- [ ] AGENTS.md updated

### Test criteria

- [ ] Two users can `ivllm list` and see the same jobs
- [ ] Two users can `ivllm connect <job>` to the same running job
- [ ] One user can `ivllm cancel <job>` on another user's job (graceful)
- [ ] Permissions audit script passes
- [ ] All tests pass

### Done when

- [ ] Manual test with two accounts on Isambard
- [ ] All new docs written
- [ ] Committed, version bumped

---

## Phase M7: Clean Up and Finalize

**Goal**: Remove all dead code from v2. Update all documentation. Ship v3.0.0.

### Files to remove

| File | Size | Notes |
|------|------|-------|
| `src/templates/inference.ts` | 1,176 lines | Replaced by `lib/*.sh` + thin wrappers |
| `src/templates/mock-inference.ts` | 114 lines | Replaced by `test-vllm.sh` mock harness |
| `src/session-helper.ts` | ~500 lines | Logic absorbed into connect.ts + bash |
| `src/monitors.ts` | ~200 lines | Monitoring moved to bash `monitor_*` |
| `src/commands/start.ts` | 61 lines | Replaced by connect.ts |
| `src/commands/interactive.ts` | 58 lines | Replaced by connect.ts |
| `src/commands/stop.ts` | 88 lines | Replaced by cancel.ts |

### Files to update

| File | Changes |
|------|---------|
| `README.md` | Remove `start`/`interactive`/`stop` references. Add `connect`/`cancel`. |
| `design/*.md` | All new docs from this roadmap |
| `src/types.ts` | Remove old types no longer used |
| `src/job.ts` | Simplify — remove old path resolution |
| `package.json` | Bump to 3.0.0 |

### Test changes

| File | Changes |
|------|---------|
| `tests/start.test.ts` | Remove or update for connect |
| `tests/status.test.ts` | Update for new lockfile |
| `tests/stop.test.ts` | Remove or update for cancel |
| `tests/inference.test.ts` | Update for new templates |
| `tests/mock-inference.test.ts` | Replace with bash-level mock tests |
| `tests/interactive.test.ts` | Remove or update for connect |

### Test criteria

- [ ] `bun test` passes with all tests updated for v3
- [ ] No references to removed commands in codebase
- [ ] No `job_details.json` references anywhere
- [ ] No `session-helper` imports

### Done when

- [ ] All old code removed
- [ ] Codebase size: ~3,000 lines TypeScript + ~500 lines bash
- [ ] All documentation updated
- [ ] Tagged as v3.0.0
- [ ] Committed

---

## Phase M7.5 — Multi-node backend evaluation (deferred)

**Why deferred**: This was originally Phase M3.5 but is moved to the end
of the roadmap. Comparative benchmarking of MP vs Ray on Slingshot is
expensive — it risks a tweak-startup-crash loop where each change requires
a full multi-node job submission, hours of queue time, and careful analysis.

Both backends work today. Keep both until the v3 migration is stable.

**Prerequisites**: All of M1–M7 (v3 migration complete, codebase clean).

**Scope**:
- [ ] Benchmark MP vs Ray on Slingshot: throughput, latency, NCCL bus bandwidth
- [ ] Does MP handle all parallelism modes (TP, PP, DP, EP) correctly?
- [ ] Does MP handle node failure gracefully?
- [ ] Decide: can batch template use MP instead of Ray?
- [ ] If yes: remove Ray dependency, clean up both templates
- [ ] If no: document which to use when

---

## Phase M7.6 — Benchmarking as a backend capability

**Goal**: A fire-and-forget batch benchmarking workflow. Set up multiple
jobs with different names/configs, trigger benchmarking, come back later
and review results.

**Why this phase**: With the v3 migration complete and both multi-node
backends working, we need a systematic way to measure performance.
Benchmarking is a natural backend capability — each backend knows how to
benchmark its own models.

**How it works**:

```bash
# Set up a model for benchmarking
ivllm connect qwen36 --config qwen36.yaml --benchmark

# Or benchmark an already-configured job
ivllm benchmark qwen36 --output results/qwen36/

# Fire-and-forget: submit multiple, check results later
for model in qwen36 gemma4 llama4; do
  ivllm connect $model --config ${model}.yaml --benchmark
done
```

The benchmark:
1. Starts vLLM (or uses an already-running instance)
2. Runs a configurable benchmark suite (latency, throughput, TTFT, ITL)
3. Writes results to the job directory and a shared results location
4. Shuts down (if `--benchmark` was the only purpose)

**Implementation**:
- Add `benchmark` method to the `Backend` interface (ADR-111):
  ```typescript
  interface Backend {
    benchmark?(jobName: string, options: BenchmarkOptions): Promise<BenchmarkResult>;
  }
  ```
- For Isambard: the benchmark runs inside the SLURM job after vLLM is healthy,
  using tools like `vllm benchmark` or custom benchmark scripts
- Results are saved as JSON in the job directory and a shared results index
- The `--benchmark` flag on `ivllm connect` auto-runs benchmark after startup
- A `ivllm benchmark <job>` command re-runs benchmark on an existing instance

**Test criteria**:
- [ ] `ivllm connect --benchmark` with mock vLLM returns synthetic results
- [ ] Real run on Isambard produces latency and throughput numbers
- [ ] Results are reproducible within acceptable variance
- [ ] Multiple benchmark jobs can run concurrently
- [ ] All tests pass

## Quick Reference

| Phase | TypeScript | Bash | Tests | Risk |
|-------|-----------|------|-------|------|
| M1: Bash Framework | No changes | ~5 files created | New bash tests | Low (additive) |
| M2: New Commands | ~3 files created, ~3 removed | No changes | Updated TS tests | Low (CLI only) |
| M3: Self-Managed Lifecycle | ~4 files modified, ~3 removed | Templates use libs | Updated tests | Medium (core change) |
| M4: Detach Mode | ~2 files modified | No changes | Updated tests | Low |
| M5: Idle Timeout | ~3 files modified | Monitor hooked up | Updated tests | Low |
| M6: Multi-User | ~2 files modified | Permission hardening | Updated tests | Low |
| M7: Clean Up | ~10 files removed | No changes | Tests removed/updated | Low |
| M7.5: MP vs Ray eval | No changes | Benchmark scripts | Benchmark tests | High (deferred) |
| M7.6: Benchmarking | Backend interface + CLI | Benchmark scripts | New tests | Low |

**Total**: ~6,500 lines TypeScript → ~3,000 lines TypeScript + ~500 lines bash

---

## Beyond v3.0.0 — Future Phases

These phases are post-v3.0.0. They build on the architecture established
in M1–M7 and are documented here to inform current design decisions.

### Phase F3: Model Router (local HTTP proxy)

**Goal**: An OpenAI-compatible HTTP server running on LOCAL that manages
multiple models across one or more backends. Agents connect to one endpoint
and the router dispatches requests by model name.

**Prerequisites**: M4 (detach), M5 (idle timeout), M6 (multi-user).

| Requirement | v3 feature it builds on |
|-------------|------------------------|
| Multiple models running simultaneously | ADR-113 (independent jobs) |
| Lazy startup on first request | Lockfile protocol + `backend.connect()` |
| Auto-shutdown after inactivity | Idle timeout (M5) |
| Port pool for model discovery | ADR-112 (port pool) |
| Backend-agnostic dispatch | ADR-111 (Backend interface) |

Also for consideration model request / response healing (particularly
unsupported use of different user types for specific models)

**Implementation sketch**:

```typescript
class ModelRouter {
  private backends: Map<string, Backend>;
  private portPool = new PortPool(11435, 11534);

  async handleChatCompletion(request: ChatRequest): Promise<Response> {
    const model = request.model;
    const job = this.registry.findByModel(model);
    
    if (!job || job.status === 'stopped') {
      // Lazy startup
      const result = await this.backends.get(job.backend)!.connect(job);
      // Wait for running, then proxy
    }
    
    // Proxy to running instance
    return this.proxy(model, request);
  }
}
```

**Files to create**:
- `src/router/server.ts` — HTTP server (Hono or built-in Bun)
- `src/router/handler.ts` — Request routing, model discovery
- `src/router/registry.ts` — Model registry (maps model name ↔ job config)
- `src/commands/router.ts` — `ivllm router` command
- `src/port-pool.ts` — Port allocation tracking

**Files to modify**:
- `src/index.ts` — add `router` command

### Phase F4: Backend Abstraction

**Goal**: Extract the Isambard-specific logic into a `Backend` implementation
(ADR-111) and support alternative backends.

**Prerequisites**: M7 (clean up).

**Implementation steps**:

1. Define `src/backends/interface.ts` — the `Backend` interface
2. Create `src/backends/isambard-vllm.ts` — wraps existing `remote-ops.ts`,
   lockfile management, and bash framework invocation into a `Backend`
3. Create `src/backends/ollama.ts` — local backend using `ollama run`
4. Add `--backend` flag to `ivllm connect`
5. Create `src/backends/registry.ts` — maps backend name → implementation

```
src/backends/
├── interface.ts         ← Backend interface definition
├── registry.ts          ← Backend registry (name → implementation)
├── isambard-vllm.ts     ← Isambard + vLLM backend
├── ollama.ts            ← Local Ollama backend (optional)
└── ssh.ts               ← Shared SSH utilities (extracted from remote-ops.ts)
```

**Candidate backends** (in priority order):

| Backend | Where it runs | Requires |
|---------|--------------|----------|
| `isambard-vllm` | Isambard AI | SLURM, SSH |
| `ollama` | Local machine | `ollama` binary |
| `isambard-container` | Isambard AI via Apptainer | ADR-010 (container support) |
| `other-hpc` | Another SLURM HPC | Different preamble, possibly different scheduler |

**Note on container backend**: The [UKGovernmentBEIS/isambard_containers](https://github.com/UKGovernmentBEIS/isambard_containers)
project already maintains pre-built vLLM Apptainer images for Isambard GH200.
An `isambard-container` backend could consume these directly via `sifter pull`,
eliminating the need for bare-metal pip install. See
`design/architecture.md` (Cross-Project Learnings) for details.

### Phase F5: Multiple Models per Node

**Goal**: Run multiple small models on a single Isambard node, each with
its own GPU allocation.

**Prerequisites**: M3 (self-managed lifecycle), F3 (model router).

**Key changes**:

1. Add `resources` field to lockfile `backendConfig`:
   ```json
   {
     "resources": {
       "gpus": [0, 1],
       "memoryPerGpuGb": 40,
       "cpuCores": 16
     }
   }
   ```
2. SLURM script uses `CUDA_VISIBLE_DEVICES` to restrict each vLLM
   instance to its allocated GPUs
3. A new `salloc` mode: allocate a multi-GPU node once, then `srun`
   individual model processes with `CUDA_VISIBLE_DEVICES`
4. Router (F3) manages GPU allocation and prevents oversubscription

**Multi-model on interactive partition** (simpler path):
- Each model gets its own `srun` on the interactive partition
- GPU affinity: `srun --gpus-per-node=2` with `CUDA_VISIBLE_DEVICES=0,1`
- Models share the node but are isolated at the GPU level
- Independent lockfiles, monitors, and idle timeouts

### Phase F6: Router Across Backends

**Goal**: The model router (F3) dispatches requests across multiple
backends transparently. An agent can use both Isambard models and local
Ollama models from the same endpoint.

**Prerequisites**: F3 (router), F4 (backend abstraction).

```
Agent → http://localhost:11434/v1  →  Router
                                        │
                          ┌─────────────┼──────────────┐
                          ▼             ▼              ▼
                   Isambard vLLM    Isambard vLLM   Ollama (local)
                   (Qwen3.6)        (Gemma4)        (Llama-3B)
```

The router selects the backend based on the model name in the request.
Model registry entries specify which backend to use:

```json
{
  "qwen36": {
    "backend": "isambard-vllm",
    "model": "Qwen/Qwen3.6-35B-A3B-FP8",
    "configFile": "/home/user/qwen36.yaml"
  },
  "gemma4": {
    "backend": "isambard-vllm",
    "model": "google/gemma-4-4b-it",
    "configFile": "/home/user/gemma4.yaml"
  },
  "llama3": {
    "backend": "ollama",
    "model": "llama3.2:3b"
  }
}
```

### Phase summary

| Phase | Name | Depends on | Deliverable |
|-------|------|-----------|-------------|
| F3 | Model Router | M4, M5, M6 | `ivllm router` — local HTTP proxy |
| F4 | Backend Abstraction | M7 | `Backend` interface + `ollama` backend |
| F5 | Multi-model per node | M3, F3 | GPU-partitioned models on one node |
| F6 | Cross-backend Router | F3, F4 | Router dispatches across backends |

These phases are direction only — they are not scheduled and will be
evolved through the same ADR + roadmap process as M1–M7.
