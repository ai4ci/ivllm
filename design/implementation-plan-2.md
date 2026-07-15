# Implementation Plan — Day 2

Pick up here after completing Phase M1 (bash framework). This plan covers
finishing M1 integration and starting Phase M2 (new CLI commands).

---

## What we have so far

- **Bash framework**: `src/templates/lib/utils.sh`, `preamble.sh`, `vllm_logs.json`
- **Bash tests**: 27 tests across 3 files (lockfile, cache, preamble) — all passing
- **Status**: `bash tests/bash/run.sh` passes, `bun test` still shows 336/364 passing (28 pre-existing failures in stale v2 template tests)

---

## Step 1: Finish M1 — Integrate bash tests

Add the bash test runner to `package.json` so `bun test` or `npm test` runs both
TypeScript and bash tests.

### Files to modify

| File | Changes |
|------|---------|
| `package.json` | Add `"test:bash"` script, update `"test"` to run both |

### What to do

```json
{
  "scripts": {
    "test": "bash tests/bash/run.sh && bun test",
    "test:bash": "bash tests/bash/run.sh",
    "test:ts": "bun test"
  }
}
```

### Verify

```bash
bash tests/bash/run.sh    # 27 bash tests pass
bun test                   # 336 TypeScript tests pass
```

### Done when

- [ ] `npm test` runs both bash and TypeScript tests
- [ ] `npm run test:bash` runs only bash tests
- [ ] Both exit 0
- [ ] Committed

---

## Step 2: M2 — New TypeScript types

Add the lockfile schema types to `src/types.ts`. This is additive — no existing
types are removed.

### Files to modify

| File | Changes |
|------|---------|
| `src/types.ts` | Add `LockfileV3`, `LockfileState`, `JobStatusV3`, `VllmConfigMetadata` |

### New types

```typescript
// Lockfile state machine (v3)
export type LockfileState = 
  | 'pending' | 'initialising' | 'running' 
  | 'failed' | 'stopped' | 'cancel';

// Full lockfile schema
export interface LockfileV3 {
  status: LockfileState;
  jobName: string;
  model: string;
  serverPort: number;
  requestedTime: string;
  idleTimeout: number;
  backend?: string;
  backendConfig?: Record<string, unknown>;
  slurmJobId?: string;
  computeHostname?: string;
  startTime?: string;
  stopTime?: string;
  vllmPid?: number;
  reason?: string;
  exitCode?: number;
}

// vllm.yaml metadata block
export interface VllmConfigMetadata {
  version?: string;
  author?: string;
  lifecycle?: 'experimental' | 'maturing' | 'stable' | 'deprecated';
  targetVllmVersion?: string;
  description?: string;
}
```

### Test criteria

- [ ] Types compile without errors (`bun run start --help`)
- [ ] `LockfileV3` can be serialized/deserialized to/from JSON
- [ ] All existing tests still pass
- [ ] Committed

---

## Step 3: M2 — New path resolution

Add path helpers for the new `$PROJECTDIR/engine/` project structure to
`src/job.ts`. The new structure is:

```
$PROJECTDIR/engine/
├── lib/           ← Shared bash framework
├── jobs/<job>/   ← Per-job data
│   ├── status.json
│   ├── jit-cache.tar.gz
│   ├── vllm.<n>.log
│   ├── vllm.yaml      ← Raw (with metadata)
│   ├── vllm.stripped.yaml  ← Stripped (without metadata)
│   └── slurm.sh
├── vllm/<version>/ ← vLLM installation
├── hf/             ← HuggingFace cache
└── diagnostics/    ← Failed job archives
```

### Files to modify

| File | Changes |
|------|---------|
| `src/job.ts` | Add `makeV3Paths()` for new structure |
| `src/job.ts` | Add `parseV3Lockfile()` for new lockfile schema |

### Function signatures

```typescript
interface V3Paths {
  engineDir: string;        // $PROJECTDIR/engine
  jobsDir: string;           // $PROJECTDIR/engine/jobs
  jobDir: string;            // $PROJECTDIR/engine/jobs/<jobName>
  lockfilePath: string;      // $PROJECTDIR/engine/jobs/<jobName>/status.json
  logPath: string;           // $PROJECTDIR/engine/jobs/<jobName>/vllm.<nodeId>.log
  configPath: string;        // $PROJECTDIR/engine/jobs/<jobName>/vllm.yaml
  strippedConfigPath: string; // $PROJECTDIR/engine/jobs/<jobName>/vllm.stripped.yaml
  scriptPath: string;        // $PROJECTDIR/engine/jobs/<jobName>/slurm.sh
  cachePath: string;         // $PROJECTDIR/engine/jobs/<jobName>/jit-cache.tar.gz
  libDir: string;            // $PROJECTDIR/engine/lib
}

function makeV3Paths(
  projectDir: string,
  jobName: string,
): V3Paths;

function parseV3Lockfile(json: string): LockfileV3 | null;
```

### Test criteria

- [ ] `makeV3Paths` returns correct paths for given project dir and job name
- [ ] `parseV3Lockfile` parses valid JSON into `LockfileV3`
- [ ] `parseV3Lockfile` returns `null` for invalid/malformed input
- [ ] Old path resolution (`makePaths`) still works unchanged
- [ ] All existing tests pass
- [ ] Committed

---

## Step 4: M2 — vllm.yaml metadata support

Update `src/vllm-config.ts` to parse the new `metadata:` block from the YAML
config, and provide a function to strip it before passing to vLLM.

### Files to modify

| File | Changes |
|------|---------|
| `src/vllm-config.ts` | Add `parseVllmConfigWithMetadata()`, `stripMetadata()` |
| `src/vllm-config.ts` | Add `VllmConfigWithMetadata` type |

### Behaviour

```yaml
# Input vllm.yaml (raw)
metadata:
  version: "1.0"
  author: "alice"
  lifecycle: maturing
  target-vllm-version: "0.22.0"

model: Qwen/Qwen3.6-35B-A3B-FP8
tensor-parallel-size: 4
idle-timeout: 30
```

After parsing:
- `parseVllmConfigWithMetadata()` returns `{ metadata: {...}, config: ServeOptions }`
- `stripMetadata(raw)` returns the YAML **without** the `metadata:` block
- The metadata block is never sent to vLLM
- `idleTimeout` is read from the config (not metadata)

### Test criteria

- [ ] Config with metadata parses correctly, metadata is separated from config
- [ ] Config without metadata returns `metadata: null`
- [ ] `stripMetadata` removes only the `metadata:` block, preserves everything else
- [ ] Existing configs (without metadata) still parse correctly
- [ ] Existing tests pass
- [ ] Committed

---

## Step 5: M2 — Scaffold connect.ts command

Create the initial `ivllm connect` command. This first version uses `--dry-run`
mode only — it parses args, creates a lockfile, generates the SLURM script,
and prints what it would do without actually SSHing anywhere.

### Files to create

| File | Description |
|------|-------------|
| `src/commands/connect.ts` | Unified connect command |

### Files to modify

| File | Changes |
|------|---------|
| `src/index.ts` | Add `connect` and `cancel` imports and routes; remove `start`, `interactive`, `stop` |

### connect.ts structure (dry-run first)

```typescript
// src/commands/connect.ts
import { Command } from 'commander'; // or use node:util.parseArgs

interface ConnectOptions {
  config?: string;      // Path to vllm.yaml
  localPort?: number;   // Local tunnel port (default 11434)
  batch?: boolean;      // Submit to standard queue (default: interactive reservation)
  dryRun?: boolean;     // Preview only
  detach?: boolean;     // Exit after starting
  noLaunch?: boolean;   // Skip assistant launcher
}

export async function cmdConnect(args: string[]): Promise<void> {
  // 1. Parse args with OptionParser
  // 2. Load credentials
  // 3. Parse vllm.yaml (with metadata support)
  // 4. If dry-run: print what would happen, generate script locally, exit
  // 5. If real: create lockfile via SSH, upload config, submit sbatch, monitor
}
```

### Dry-run behaviour

```bash
ivllm connect qwen2 --config examples/qwen2.5-instruct.yaml --dry-run

# Output:
# === ivllm connect (dry-run) ===
# Job      : qwen2
# Model    : Qwen/Qwen2.5-0.5B-Instruct
# Server   : interactive partition (sbatch)
# Local    : http://localhost:11434/v1
# Lockfile : $PROJECTDIR/engine/jobs/qwen2/status.json
# Script   : $PROJECTDIR/engine/jobs/qwen2/slurm.sh
# Config   : $PROJECTDIR/engine/jobs/qwen2/vllm.yaml
# ── Generated script preview ──
# (contents of slurm.sh)
```

### Test criteria

- [ ] `ivllm connect --help` shows proper usage
- [ ] `ivllm connect job --config file.yaml --dry-run` prints details and exits 0
- [ ] Dry-run generates a valid SLURM script (sources utils.sh + preamble.sh)
- [ ] Dry-run writes lockfile-format JSON to preview
- [ ] Missing `--config` errors with clear message for first-time use
- [ ] All existing tests still pass
- [ ] Committed

---

## Step 6: M2 — Scaffold cancel.ts command

Create the initial `ivllm cancel` command. Dry-run mode prints what it would do.

### Files to create

| File | Description |
|------|-------------|
| `src/commands/cancel.ts` | Cancel command |

### cancel.ts structure

```typescript
interface CancelOptions {
  force?: boolean;   // Use scancel directly instead of graceful cancel
  dryRun?: boolean;
}

export async function cmdCancel(args: string[]): Promise<void> {
  // 1. Parse args
  // 2. If dry-run: print what would happen
  // 3. If real:
  //    - Read lockfile, write "cancel" status
  //    - If --force: scancel <slurmJobId>, update lockfile to "stopped"
  //    - Tail logs until status becomes "stopped"
}
```

### Test criteria

- [ ] `ivllm cancel --help` shows proper usage
- [ ] `ivllm cancel job --dry-run` prints what it would do
- [ ] `ivllm cancel job --force --dry-run` prints force-cancel path
- [ ] All existing tests still pass
- [ ] Committed

---

## Step 7: M2 — Update index.ts (CLI routing)

Wire the new commands into the CLI entry point and remove the old ones.

### Files to modify

| File | Changes |
|------|---------|
| `src/index.ts` | Replace `start`/`interactive`/`stop` with `connect`/`cancel` |

### New USAGE output

```
Usage: ivllm <command> [options]

Commands:
  setup <version>         Install vLLM <version> on the HPC (one-off)
  connect <job>           Start or connect to an inference session
  cancel <job>            Cancel a running job
  list                    List stored vLLM job configs
  status [job]            Show status of a job (or all jobs)
  config                  Show or set configuration
  agent                   Launch AI assistant connected to local vLLM server

Options:
  --version, -v           Show version
```

### Test criteria

- [ ] `ivllm connect --help` works
- [ ] `ivllm cancel --help` works  
- [ ] `ivllm start` shows "unknown command" error
- [ ] `ivllm interactive` shows "unknown command" error
- [ ] `ivllm stop` shows "unknown command" error
- [ ] All existing tests for remaining commands (`setup`, `list`, `status`, `config`, `agent`) still pass
- [ ] Committed

---

## Quick reference

| Step | Files created | Files modified | Tests |
|------|--------------|----------------|-------|
| 1. Finish M1 | — | `package.json` | Manual verify |
| 2. New types | — | `src/types.ts` | New TS tests |
| 3. New paths | — | `src/job.ts` | New TS tests |
| 4. vllm.yaml metadata | — | `src/vllm-config.ts` | New TS tests |
| 5. connect.ts | `src/commands/connect.ts` | `src/index.ts` | New TS tests |
| 6. cancel.ts | `src/commands/cancel.ts` | `src/index.ts` | New TS tests |
| 7. CLI routing | — | `src/index.ts` | Update existing |

### Running tests

```bash
bash tests/bash/run.sh    # bash tests (27, should stay green)
bun test                  # TypeScript tests (should stay 336 green)
```

### Quick commands

```bash
# Test single file during development
bun test tests/vllm-config.test.ts

# Run new connect tests
bun test tests/connect.test.ts

# Dry-run the new command
bun run src/index.ts connect test-job --config examples/qwen2.5-instruct.yaml --dry-run
```
