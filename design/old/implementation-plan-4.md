# Implementation Plan — Day 4: MVP

This plan covers the remaining steps to reach a Minimal Viable Product.
MVP means a user can run `ivllm connect <job> --config <yaml>` and get a
working, auto-shutting-down, multi-user vLLM endpoint.

---

## Current state

| Phase | Status | Notes |
|-------|--------|-------|
| M1 Bash framework | ✅ Done | utils.sh, vllm-env.sh, 27 bash tests |
| M2 New CLI commands | ✅ Done | connect/cancel scaffold, v3 types, paths, metadata |
| M3 Step 0 (mock refactor) | ✅ Done | OpsMode, MockRemoteFs |
| M3 Step 1 (SSH lockfile) | ✅ Done | Real SSH: pre-flight, lockfile creation, config upload |
| M3 Steps 2-5 | ❌ Not started | Script gen, sbatch, monitor, tunnel, cancel |
| M4 Detach | ❌ Not started | --detach flag, reconnection |
| M5 Idle timeout | ❌ Not started | Log-based auto-shutdown |
| M6 Multi-user | ❌ Not started | Permissions, diagnostics |
| M7 Cleanup | ❌ Not started | Remove remaining dead code |

**Test count**: 130 tests, 0 failures, 0 tsc errors in source code.

---

## Remaining steps to MVP

### Step 1: Generate SLURM script + submit sbatch

Complete the connect flow: generate a thin SLURM script that sources the
bash framework, upload it, and submit it via sbatch.

**Files to create**:
- `src/templates/connect-template.sh` — Static SLURM script template (~30 lines)

**Files to modify**:
- `src/commands/connect.ts` — Add script generation, sbatch submission

**The SLURM script**:

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

**What to add to connect.ts**:

```typescript
// Template is a static file, parameters substituted via simple string replace
const script = readFileSync(templatePath, 'utf-8')
  .replace(/\{\{JOB_NAME\}\}/g, jobName)
  .replace(/\{\{GPU_COUNT\}\}/g, String(gpuCount))
  .replace(/\{\{TIME_LIMIT\}\}/g, connectArgs.timeLimit)
  .replace(/\{\{LOG_FILE\}\}/g, logFile)
  .replace(/\{\{LIB_DIR\}\}/g, v3paths.engineLibDir)
  .replace(/\{\{ENGINE_DIR\}\}/g, v3paths.engineDir)
  .replace(/\{\{CONFIG_FILE\}\}/g, v3paths.vllmConfigFile)
  .replace(/\{\{SERVER_PORT\}\}/g, String(serverPort));

// Upload script
await ops.runRemote(`cat > ${v3paths.scriptFile} << 'SCRIPTEOF'\n${script}\nSCRIPTEOF`);
await ops.runRemote(`chmod +x ${v3paths.scriptFile}`);

// Submit sbatch
const sbatchResult = await ops.runRemote(`sbatch ${v3paths.scriptFile}`);
```

**Test criteria**:
- [ ] Template renders without errors (simple string substitution)
- [ ] Generated script passes `bash -n` (syntax check)
- [ ] Generated script sources vllm-env.sh and utils.sh
- [ ] `ivllm connect --dry-run` prints the generated script
- [ ] Committed

---

### Step 2: Monitor lockfile + establish tunnel

After sbatch submission, poll the lockfile for status transitions and
establish an SSH tunnel when vLLM is healthy.

**Files to modify**:
- `src/commands/connect.ts` — Add polling loop, tunnel creation

**Monitor loop**:

```typescript
// Poll lockfile every 5 seconds until running or failed
while (true) {
  const { stdout } = await ops.runRemote(`cat ${v3paths.statusFile}`, {
    env: [], silent: true,
  });
  const lockfile = parseV3Lockfile(stdout);
  if (!lockfile) continue;

  console.log(`  [${new Date().toTimeString().slice(0, 8)}] ${lockfile.status}`);

  if (lockfile.status === 'running') {
    // Establish SSH tunnel
    const tunnel = ops.spawnTunnel(
      connectArgs.localPort,
      lockfile.computeHostname!,
      lockfile.serverPort,
    );
    console.log(`\n🚀 OpenAI API endpoint: http://localhost:${connectArgs.localPort}/v1`);
    console.log(`   Model: ${lockfile.model}`);
    console.log(`\nJob is running. Type Ctrl+C to disconnect (job keeps running).\n`);
    break;
  }

  if (lockfile.status === 'failed' || lockfile.status === 'stopped') {
    console.error(`Job ${lockfile.status}: ${lockfile.reason || 'unknown'}`);
    process.exit(1);
  }

  await sleep(5000);
}
```

**Test criteria**:
- [ ] Polling loop detects `running` status and establishes tunnel
- [ ] Polling loop exits on `failed`/`stopped` with error message
- [ ] Ctrl+C exits without killing the SLURM job
- [ ] Tunnel is created with correct port and hostname
- [ ] Committed

---

### Step 3: Implement `ivllm cancel` with SSH

Add real SSH operations for graceful and force cancellation.

**Files to modify**:
- `src/commands/cancel.ts` — Add SSH-based cancel

**Graceful cancel**:
```typescript
// Write "cancel" to lockfile, let the compute-side monitor handle shutdown
await ops.runRemote(
  `jq '.status = "cancel"' ${v3paths.statusFile} > ${v3paths.statusFile}.tmp && mv ${v3paths.statusFile}.tmp ${v3paths.statusFile}`
);
// Tail logs until stopped
while (true) {
  const { stdout } = await ops.runRemote(`cat ${v3paths.statusFile}`, { env: [], silent: true });
  const lockfile = parseV3Lockfile(stdout);
  if (lockfile?.status === 'stopped') {
    console.log(`\n✓ Job '${jobName}' stopped: ${lockfile.reason || 'clean'}`);
    break;
  }
  await sleep(2000);
}
```

**Force cancel**:
```typescript
// Read SLURM job ID from lockfile, scancel directly
const { stdout } = await ops.runRemote(`cat ${v3paths.statusFile}`);
const lockfile = parseV3Lockfile(stdout);
if (lockfile?.slurmJobId) {
  await ops.runRemote(`scancel ${lockfile.slurmJobId}`);
}
// Update lockfile to stopped
await ops.runRemote(
  `jq '.status = "stopped"' ${v3paths.statusFile} > ${v3paths.statusFile}.tmp && mv ${v3paths.statusFile}.tmp ${v3paths.statusFile}`
);
```

**Test criteria**:
- [ ] `ivllm cancel job` writes cancel to lockfile via SSH
- [ ] `ivllm cancel job --force` runs scancel via SSH
- [ ] `--dry-run` works without SSH
- [ ] Error handling: job not found, SSH failure
- [ ] Committed

---

### Step 4: Detach mode

Add `--detach` flag to `ivllm connect` so the CLI exits after submitting
the job, leaving it running. Reconnection reads the lockfile and
establishes a new tunnel.

**Files to modify**:
- `src/commands/connect.ts` — Add `--detach` handling

**Detach behaviour**:

```bash
ivllm connect qwen2 --config qwen2.yaml --detach
# Output:
# Job 'qwen2' submitted (SLURM job 123456)
# To reconnect: ivllm connect qwen2
# To cancel:    ivllm cancel qwen2
```

**Reconnect behaviour**:

```bash
ivllm connect qwen2
# Lockfile exists → check status
# If running → establish tunnel immediately
# If stopped/failed → restart
# If initialising → wait for running
```

**Test criteria**:
- [ ] `--detach` exits after sbatch submission
- [ ] Reconnecting shows correct job state
- [ ] Reconnecting to running job establishes tunnel immediately
- [ ] Committed

---

### Step 5: Idle timeout

Jobs auto-shutdown after a configurable period of inactivity. The
`monitor_head` function in the bash framework already has the logic —
it needs to be wired up with the `vllm_logs.json` config and the
`VLLM_LOGGING_CONFIG_PATH` env var.

**Files to modify**:
- `src/templates/lib/vllm-env.sh` — Already sets `VLLM_LOGGING_CONFIG_PATH`
- `src/templates/lib/utils.sh` — `monitor_head` already has idle timeout logic
- `src/commands/connect.ts` — Pass `idle-timeout` from config into lockfile

**How it works** (already implemented in bash):
1. `idleTimeout` is set in vllm.yaml (`idle-timeout: 30`)
2. Stored in lockfile when created (already done in Step 1)
3. `monitor_head` reads the timeout from the lockfile
4. Every 10s, scans the last N minutes of the vLLM access log for API requests
5. If no requests within the window → shutdown with reason "idle timeout"

**Test criteria**:
- [ ] `idle-timeout: 1` causes job to shut down after 1 minute of inactivity
- [ ] Active API requests reset the idle timer
- [ ] `idle-timeout: -1` keeps job running indefinitely
- [ ] Default is 30 minutes
- [ ] Committed

---

### Step 6: Multi-user hardening

Ensure multiple project members can share running jobs. Fix permissions,
add diagnostics, update documentation.

**Files to modify**:
- `src/templates/lib/vllm-env.sh` — Add `umask 0002`
- `src/templates/setup.ts` — Ensure group-writable engine directory
- Various — Permissions audit

**Permissions checklist**:
- [ ] `$PROJECTDIR/engine/` is group-writable (`chmod -R g+w`)
- [ ] `$PROJECTDIR/engine/jobs/` is group-readable
- [ ] All `status.json` files are group-writable
- [ ] JIT cache tar.gz files are group-readable
- [ ] HF cache at `$PROJECTDIR/hf/` is group-readable
- [ ] vLLM venv at `$PROJECTDIR/engine/vllm/<version>/` is group-readable

**Test criteria**:
- [ ] Two users can `ivllm list` and see the same jobs
- [ ] Two users can `ivllm connect <job>` to the same running job
- [ ] One user can `ivllm cancel <job>` on another user's job
- [ ] Committed

---

### Step 7: Final cleanup

Remove remaining v2 references, update README, bump version.

**Files to remove**:
- `src/commands/status.ts` — If still using v2 lockfile format
- `src/job.ts` — Remove `makePaths`, `parseJobDetails`, `parseStartArgs` (v2)

**Files to update**:
- `README.md` — Remove v2 command references, update for v3
- `package.json` — Bump to 3.0.0

**Test criteria**:
- [ ] No references to `job_details.json` anywhere
- [ ] No references to v2 commands in docs
- [ ] 130+ tests still pass
- [ ] Tagged as v3.0.0
- [ ] Committed

---

## Quick reference

| Step | What | Files created | Files modified | Risk |
|------|------|--------------|----------------|------|
| 1 | SLURM script + sbatch | `connect-template.sh` | `connect.ts` | Medium |
| 2 | Monitor + tunnel | — | `connect.ts` | Medium |
| 3 | Real cancel (SSH) | — | `cancel.ts` | Low |
| 4 | Detach mode | — | `connect.ts` | Low |
| 5 | Idle timeout | — | `vllm-env.sh`, `connect.ts` | Low |
| 6 | Multi-user | — | `vllm-env.sh`, `setup.ts` | Low |
| 7 | Final cleanup | — | Various | Low |

### Running tests

```bash
bash tests/bash/run.sh    # 27 bash tests
bun test                  # 130 TypeScript tests
```

### After MVP

- **F3**: Model router (local HTTP proxy for multiple models)
- **F4**: Backend abstraction (Ollama, container backends)
- **F5**: Multiple models per node
- **F6**: Cross-backend router
