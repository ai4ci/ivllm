# Implementation Plan — Day 3

This plan covers **Phase M3: Self-Managed Lifecycle** — the core architectural
change where SLURM scripts switch from the old TypeScript-generated templates
to the new bash framework, and `ivllm connect` performs real SSH operations.

---

## What we have so far

- **Bash framework**: `src/templates/lib/utils.sh`, `vllm-env.sh`, `vllm_logs.json`
- **Bash tests**: 27 tests — all passing
- **New commands**: `ivllm connect` and `ivllm cancel` scaffolded with `--dry-run`
- **New types**: `LockfileV3`, `EnginePathsV3`, `VllmConfigMetadata`, `parseV3Lockfile()`
- **Config metadata**: `idleTimeout`, `metadata:` block parsed and stripped
- **CLI routing**: `connect`/`cancel` replace `start`/`interactive`/`stop`
- **TypeScript**: 354 tests passing (28 pre-existing failures in stale v2 template tests)

---

## Phase M3 overview

This is the biggest phase. It makes the compute node self-sufficient by
having SLURM scripts source the bash framework instead of embedding bash
inside TypeScript template strings.

### Prerequisite: refactor mock infrastructure

The existing `makeRemoteOps(config, dryRun: boolean)` mock is inadequate
for testing v3 connect/cancel. It returns canned text responses (e.g.
`cat → 'lockfile'`) rather than simulating real filesystem operations.
A `mock` mode with a local filesystem sandbox is needed before we can
test the new SSH operations.

See `design/testing.md` (Mock Remote Ops section) for the full design.

### Key changes

| Before (v2) | After (v3) |
|-------------|-----------|
| `src/templates/inference.ts` generates 300+ line bash as a string | `slurm.sh` is a thin 30-line wrapper that sources `lib/utils.sh` + `lib/vllm-env.sh` |
| `session-helper.ts` orchestrates the 10-step lifecycle | `connect.ts` does pre-flight, lockfile, upload, sbatch, monitor, tunnel |
| `monitors.ts` polls lockfile from LOCAL | `monitor_head`/`monitor_worker` in bash run on COMPUTE |
| Job starts via `sbatch` with environment-embedded config | Job starts via `sbatch` with `--export` and config in job directory |
| v2 lockfile: `job_details.json` in `~/<job>/` | v3 lockfile: `status.json` in `$PROJECTDIR/engine/jobs/<job>/` |

---

## Step 0: Refactor mock infrastructure

Before adding real SSH ops, refactor `makeRemoteOps` to support three
modes and add a mock filesystem sandbox for testing.

### Files to modify

| File | Changes |
|------|---------|
| `src/remote-ops.ts` | Accept `OpsMode` (`'real' | 'mock' | 'dry-run'`) instead of `boolean` |
| `src/remote-ops.ts` | Add `MockRemoteFs` class for local filesystem simulation |
| `src/types.ts` | Update `RemoteOps` interface if needed |
| `tests/remote-ops.test.ts` | Update existing tests for new API |

### OpsMode refactor

```typescript
type OpsMode = 'real' | 'mock' | 'dry-run';

// Old signature — update callers
function makeRemoteOps(config: Credentials, dryRun: boolean): RemoteOps;
// New signature
function makeRemoteOps(config: Credentials, mode: OpsMode): RemoteOps;
```

Update all callers (there aren't many — just `connect.ts`, `cancel.ts`,
`stop.ts`, and tests).

### MockRemoteFs behaviour

In `mock` mode:
- `runRemote` with `mkdir`, `cat`, `jq`, `set -C` — executes against a
  local temp directory so files are really created and can be read back
- `runRemote` with `sbatch` — returns `Submitted batch job 123456`
- `runRemote` with `scancel` — no-op
- `copyFile` — copies to temp dir (same as dry-run)
- `spawnTunnel` — returns mock emitter (same as dry-run)
- `checkSSH` — returns true

### Test criteria

- [ ] Existing callers updated: `makeRemoteOps(config, true)` → `makeRemoteOps(config, 'dry-run')`
- [ ] Existing tests still pass
- [ ] `mock` mode creates real files in a temp directory when `runRemote`
      processes `mkdir`, `cat`, `jq` commands
- [ ] `mock` mode lockfile can be written and read back via `runRemote`
- [ ] `mock` mode `sbatch` returns a fake job ID
- [ ] Committed

---

## Step 1: Real lockfile creation via SSH

`ivllm connect` currently only does `--dry-run`. Add real SSH operations
for the first step of the lifecycle: creating the lockfile on the HPC.

### Files to modify

| File | Changes |
|------|---------|
| `src/commands/connect.ts` | Add SSH-based lockfile creation using existing `remote-ops.ts` |

### What to add

```typescript
// After dry-run check, before placeholder:
const ops = makeRemoteOps(config, false);

// 1. SSH pre-flight
await ops.checkSSH();

// 2. Create engine directory structure
await ops.runRemote(`mkdir -p ${v3paths.engineDir}/jobs/${jobName}`);

// 3. Create lockfile atomically
const lockfileJson = JSON.stringify({
  status: 'pending',
  jobName: jobName,
  model: parsedConfig.model,
  serverPort: generateRandomHighPort(),
  requestedTime: new Date().toISOString(),
  idleTimeout: parsedConfig.idleTimeout,
} as LockfileV3);
await ops.runRemote(`set -C; cat > ${v3paths.statusFile} << 'JSONEOF'
${lockfileJson}
JSONEOF`);
```

### Test criteria

This step is hard to unit-test without SSH. Use `--dry-run` for
development. Mark with `// @integration` for future mock-SSH tests.

- [ ] `ivllm connect test-job --config test.yaml` attempts SSH and creates lockfile
- [ ] Error handling: SSH failure, lockfile already exists, bad config
- [ ] Pre-existing `ivllm list`, `ivllm status` still work
- [ ] Committed

---

## Step 2: Upload config and generate SLURM script

Upload the vllm.yaml config to the job directory on the HPC and generate
the SLURM script using the bash framework.

### Files to modify

| File | Changes |
|------|---------|
| `src/commands/connect.ts` | Add SCP upload, script generation |
| `src/templates/connect-template.sh` | **New** — static SLURM script template that sources libs |

### The SLURM script template

Create a minimal template at `src/templates/connect-template.sh`:

```bash
#!/bin/bash
#SBATCH --job-name={{JOB_NAME}}
#SBATCH --nodes=1
#SBATCH --gpus={{GPU_COUNT}}
#SBATCH --time={{TIME_LIMIT}}
#SBATCH --output={{LOG_FILE}}

source {{LIB_DIR}}/vllm-env.sh
source {{LIB_DIR}}/utils.sh

JOB_NAME="{{JOB_NAME}}"
CONFIG_FILE="{{CONFIG_FILE}}"

export ENGINE_DIR="{{ENGINE_DIR}}"

# Lockfile already created by CLI with "pending"
# Overwrite with SLURM details
srun --export=ALL --cpu-bind=cores vllm serve \
  --config "$CONFIG_FILE" \
  --host 0.0.0.0 --port {{SERVER_PORT}} &
VLLM_PID=$!

update_status_initialise "$JOB_NAME" "$VLLM_PID"
setup_traps "$JOB_NAME"
restore_cache "$JOB_NAME"
monitor_startup "$JOB_NAME" "$$"
monitor_head "$JOB_NAME" "$$" &
wait $VLLM_PID
tidy_up "$JOB_NAME" $?
```

This is ~30 lines instead of the current 300+ in `inference.ts`.

### What to add to connect.ts

```typescript
import { renderFile } from '../template-utils'; // or simple string replace

// Generate SLURM script
const scriptContent = renderTemplate(templatePath, {
  JOB_NAME: jobName,
  GPU_COUNT: gpuCount,
  TIME_LIMIT: connectArgs.timeLimit,
  LOG_FILE: v3paths.logFileGlob.replace('*', '0'), // node 0 log
  LIB_DIR: v3paths.engineLibDir,
  ENGINE_DIR: v3paths.engineDir,
  CONFIG_FILE: v3paths.vllmConfigFile,
  SERVER_PORT: serverPort,
});

// Upload config and script
await ops.copyFile(localConfigPath, v3paths.vllmConfigFile);
await ops.runRemote(`cat > ${v3paths.scriptFile} << 'SCRIPTEOF'
${scriptContent}
SCRIPTEOF`);
await ops.runRemote(`chmod +x ${v3paths.scriptFile}`);
```

### Test criteria

- [ ] Template renders without errors (simple string substitution)
- [ ] Generated script passes `bash -n` (syntax check)
- [ ] Generated script sources vllm-env.sh and utils.sh
- [ ] Dry-run prints the full generated script
- [ ] Committed

---

## Step 3: Download model and submit sbatch

Complete the connect flow: download model if not cached, submit sbatch,
monitor the lockfile, and establish tunnel when running.

### Files to modify

| File | Changes |
|------|---------|
| `src/commands/connect.ts` | Add model download, sbatch, monitoring, tunnel |
| `src/remote-ops.ts` | May need a `submitSbatch` helper |

### Connect flow (final)

```typescript
// 1. SSH pre-flight
ops.checkSSH();

// 2. Create lockfile (Step 1)
createLockfile(ops, v3paths, lockfileData);

// 3. Upload config + script (Step 2)
await uploadConfig(ops, v3paths, localConfigPath);
await uploadScript(ops, v3paths, renderedScript);

// 4. Download model if not cached
await ensureModelDownloaded(ops, model, hfHome, hfToken);

// 5. Submit sbatch
const jobId = await ops.submitSbatch(v3paths.scriptFile);

// 6. Monitor lockfile loop
await monitorJob(ops, v3paths, jobId, onRunning);

// 7. On running: establish SSH tunnel
function onRunning(details: LockfileV3) {
  const tunnel = ops.spawnTunnel(localPort, details.computeHostname!, details.serverPort);
  console.log(`🚀 http://localhost:${localPort}/v1`);
}
```

### Test criteria

- [ ] Full flow works with `--mock` mode (using existing mock infrastructure)
- [ ] Full flow works end-to-end on Isambard with a real model
- [ ] Ctrl+C during monitoring exits without killing the SLURM job
- [ ] Lockfile polling detects running state correctly
- [ ] Tunnel is established when vLLM is healthy
- [ ] Committed

---

## Step 4: Implement `ivllm cancel` with SSH

Add real SSH operations for graceful and force cancellation.

### Files to modify

| File | Changes |
|------|---------|
| `src/commands/cancel.ts` | Add SSH-based cancel (write cancel, scancel, tail logs) |

### Cancel flow

```typescript
const ops = makeRemoteOps(config, false);

if (cancelArgs.force) {
  // Force cancel: read lockfile for SLURM job ID, scancel it
  const raw = await ops.runRemote(`cat ${v3paths.statusFile}`);
  const lockfile = parseV3Lockfile(raw.stdout);
  if (lockfile?.slurmJobId) {
    await ops.runRemote(`scancel ${lockfile.slurmJobId}`);
  }
  // Update lockfile to stopped
  await ops.runRemote(`jq '.status = "stopped"' ${v3paths.statusFile} > ${v3paths.statusFile}.tmp && mv ${v3paths.statusFile}.tmp ${v3paths.statusFile}`);
} else {
  // Graceful: write "cancel" to lockfile, wait for stopped
  await ops.runRemote(`jq '.status = "cancel"' ${v3paths.statusFile} > ${v3paths.statusFile}.tmp && mv ${v3paths.statusFile}.tmp ${v3paths.statusFile}`);
  // Tail logs until stopped or timeout
  await tailUntilStopped(ops, v3paths);
}
```

### Test criteria

- [ ] `ivllm cancel` writes cancel to lockfile via SSH
- [ ] `ivllm cancel --force` runs scancel via SSH
- [ ] Error handling: job not found, SSH failure
- [ ] `--dry-run` works without SSH
- [ ] Committed

---

## Step 5: Remove old code

Remove the v2 files that are now replaced.

### Files to remove

| File | Replacement |
|------|-------------|
| `src/templates/inference.ts` (1,176 lines) | `connect-template.sh` + `lib/utils.sh` |
| `src/templates/mock-inference.ts` | Mock vLLM in test harness |
| `src/session-helper.ts` | Logic in `connect.ts` + bash framework |
| `src/monitors.ts` | Bash `monitor_head`/`monitor_worker` |

### Files to modify

| File | Changes |
|------|---------|
| `src/index.ts` | Remove unused imports for removed modules |
| `package.json` | Remove test references to removed template tests |

### Test criteria

- [ ] Build still compiles without removed files
- [ ] All remaining tests pass
- [ ] No imports from removed modules anywhere
- [ ] Committed

---

## Quick reference

| Step | What | Files created | Files modified | Risk |
|------|------|--------------|----------------|------|
| 0 | Refactor mock infrastructure | — | `remote-ops.ts`, `types.ts` | Low (mechanical change) |
| 1 | Real lockfile via SSH | — | `connect.ts` | Low (additive) |
| 2 | Config upload + script gen | `connect-template.sh` | `connect.ts` | Medium |
| 3 | Model download + sbatch + monitor | — | `connect.ts` | **High** (core change) |
| 4 | Real cancel with SSH | — | `cancel.ts` | Low |
| 5 | Remove old code | — | — | Medium |

### Running tests

```bash
bash tests/bash/run.sh    # 27 bash tests
bun test                  # TypeScript tests
bun run src/index.ts connect test-job --config test.yaml --dry-run  # verify dry-run
```

### After M3 is done

- `ivllm connect` can start a real job on Isambard
- `ivllm cancel` can stop it gracefully or with force
- The old TypeScript template system is fully removed
- The bash framework is the sole lifecycle manager
