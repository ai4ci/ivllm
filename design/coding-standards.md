# Coding Standards — isambard-vllm

This document defines coding conventions and patterns for the isambard-vllm
codebase, covering both TypeScript (CLI layer) and Bash (HPC runtime layer).

---

## TypeScript Standards

### Argument Parsing: Use Commander.js

All CLI commands use [commander.js](https://github.com/tj/commander.js)
in `src/index.ts`. Commands are defined as `.command()` chains with
`.option()` and `.action()`.

```typescript
import { program } from 'commander';

program
  .name('ivllm')
  .version(version)
  .command('connect <job>')
  .option('--config <path>', 'vLLM config YAML (required for first use)')
  .option('--local-port [port]', 'Local port for API', '11434')
  .option('--dry-run', 'Preview without connecting to HPC', false)
  .action(cmdConnect);

program
  .command('cancel <jobName>')
  .option('--force', 'use slurm scancel directly instead of graceful cancel', false)
  .option('--abort', 'abort the job capturing diagnostics', false)
  .action(cmdCancel);
```

Command groups follow the same `.command().option().action()` shape whether
they're a flat command (`connect`, `cancel`, `status`) or have subcommands —
`bench` is defined as `program.command('bench')` once, then
`bench.command('submit')`/`.command('status')`/`.command('results')` are
chained off that returned `Command` object, each with their own
`.argument()`/`.option()`/`.action()`.

See `src/index.ts` for the full command layout.

### Object-Oriented Patterns: Prefer Classes over Interface Proliferation

**Prefer classes with methods** over bags-of-interfaces when the data has
behaviour. Use inheritance for lifecycle state.

```typescript
// GOOD — class with behaviour
class JobConfig {
  readonly jobName: string;
  readonly model: string;
  readonly serverPort: number;
  readonly idleTimeout: number;
  private yaml: Record<string, unknown>;

  constructor(jobName: string, yamlPath: string) {
    this.jobName = jobName;
    this.yaml = parseYaml(yamlPath);
    this.model = this.yaml.model;
    this.serverPort = generateRandomHighPort();
    this.idleTimeout = this.yaml.idleTimeout ?? 30;
  }

  toLockfile(): LockfileV3 {
    return {
      status: 'pending',
      jobName: this.jobName,
      model: this.model,
      serverPort: this.serverPort,
      idleTimeout: this.idleTimeout,
      requestedTime: new Date().toISOString(),
    };
  }
}

// BAD — replicated interfaces per data view
interface LockfileSchema { status: string; jobName: string; model: string; ... }
interface JobDisplayInfo { name: string; model: string; statusText: string; ... }
interface ApiEndpointInfo { model: string; port: number; hostname: string; ... }
```

**When to use a class**:
- The data has derived/computed fields (e.g. `Paths` with `makePaths`)
- The data has behaviour (e.g. `JobConfig.toLockfile()`)
- Multiple views of the same data exist (screen display, API, serialisation)
- The data has lifecycle (e.g. `ProcessState`, `SessionState`)

**The Backend class pattern**

The v3 architecture defines an abstract `Backend` class (see `src/backends/Backend.ts`)
for abstracting different inference runtimes. The `IsambardBareMetalBackend` is the
first (and only current) implementation.

```typescript
// src/backends/Backend.ts — abstract base class
export abstract class Backend {
  abstract bootstrap(): Promise<void>;
  abstract setup(version: string, force?: boolean): Promise<void>;
  abstract connect(job: string, localPort: number): Promise<CloseableEventEmitter>;
  abstract requestCancel(job: string, force: boolean, abort: boolean): Promise<void>;
  abstract requestStart(job: string, maxTime: string, batch: boolean, config?: string): Promise<void>;
  abstract getAllJobStatus(): Promise<LockfileV3[]>;
  abstract watchLog(job: string, node?: string, start?: boolean): Promise<CloseableEventEmitter>;
  abstract fetchDiagnostics(job: string, localDest?: string): Promise<string>;

  // Convenience methods
  getJobStatus(job: string): Promise<LockfileV3>;
  isCancelling(job: string): Promise<boolean>;
  isRunning(job: string): Promise<boolean>;
  isStopped(job: string): Promise<boolean>;
  isStartable(job: string): Promise<boolean>;
  isStarting(job: string): Promise<boolean>;

  // Non-abstract, default-throws — only overridden by backends that support
  // `ivllm bench` (scope.md §1):
  requestBenchmark(comparison: string, configs: string[], time?: string): Promise<void>;
  getBenchmarkStatus(comparison: string): Promise<BenchmarkStatus>;
  fetchBenchmarkResults(comparison: string, localDest: string): Promise<
    | { ready: true; path: string }
    | { ready: false; status: BenchmarkStatus }
  >;
}

// src/backends/IsambardBareMetalBackend.ts — concrete implementation
export class IsambardBareMetalBackend extends Backend {
  // Wraps SshRemoteOps + bash framework on HPC
}
```

The `Backend` class extends beyond the original interface design (ADR-111)
with `bootstrap()`, `requestStart()`, state helpers, and a unified
`requestCancel(job, force, abort)` instead of separate cancel/forceCancel/
abort methods. The benchmark methods are deliberately non-abstract
(default-throws) rather than abstract, since not every backend needs to
support `ivllm bench` — this is the pattern to follow for any future
optional/backend-specific capability, rather than forcing every backend to
implement a method it can't meaningfully support.

**When to use an interface**:
- Simple data transfer objects with no behaviour
- External API request/response shapes
- Configuration blobs that are parsed once and consumed as-is
- Service contracts (like `Backend`) that have multiple implementations

### Naming Conventions

| Construct | Convention | Example |
|-----------|-----------|---------|
| Files | `kebab-case.ts` | `config.ts`, `local-ops.ts`, `semver.ts`, `utils.ts` |
| Directories | `kebab-case/` | `backends/`, `ops/` |
| Classes | `PascalCase` | `Backend`, `IsambardBareMetalBackend`, `SshRemoteOps` |
| Interfaces | `PascalCase` | `Credentials`, `LockfileV3`, `RunRemoteOptions` |
| Types | `PascalCase` | `LockfileState`, `V1ModelsResponse` |
| Functions | `camelCase` | `loadCredentials`, `formatJobRow` |
| Constants | `UPPER_SNAKE_CASE` | `HEALTH_CHECK_TIMEOUT` |
| Private fields | `camelCase` (no `_` prefix) | `private ops: RemoteOps` |
| Bash scripts | `snake_case.sh` | `utils.sh`, `vllm-env.sh`, `slurm-vllm-serve.sh` |
| Bash wrappers | `snake_case.sh` | `ivllm-serve.sh`, `ivllm-status.sh` |

### Error Handling

- Throw `Error` with descriptive messages for expected failure paths
- Catch at the command boundary (the exported function for each command)
- Use `process.exit(1)` only at the top-level command handler
- Do not print stack traces to the user (use `(e as Error).message`)
- For debug info, use `console.error` not `console.log`

```typescript
// GOOD
export async function cmdConnect(args: string[]): Promise<void> {
  try {
    // ... command logic
  } catch (e) {
    console.error('Error:', (e as Error).message);
    process.exit(1);
  }
}
```

### Testing (TypeScript)

- Test files live in `tests/unit/` and `tests/integration/`
- Use `bun:test` — `bun test` runs all TypeScript tests
- Mock `RemoteOps` via `MockRemoteOps` class (records calls, no SSH needed)
- Use real `http.Server` for local-ops tests (port/health/model query)
- Integration tests use `TestBackend` + `TestRemoteOps` for full lifecycle
- All tests must pass before commit: `bun test`

```
tests/
├── unit/
│   ├── Backend.test.ts          — Lockfile parsing, state checks
│   ├── RemoteOps.mock.test.ts   — MockRemoteOps records all calls
│   ├── local-ops.test.ts        — Port/health/model queries
│   ├── semver.test.ts           — Version comparison/sorting
│   └── bash-integration.test.ts — Wraps bash suite via bun:test
└── integration/
    └── CLI.lifecycle.test.ts    — Full Backend lifecycle
```

---

## Bash Standards

### Function-Oriented Design

Bash code MUST use functions with `local` variables. Avoid global variables
except for truly global constants.

```bash
# GOOD
resolve_lockfile() {
    local job=$1
    echo "$ENGINE_DIR/jobs/$job/status.json"
}

update_status_running() {
    local lockfile
    lockfile=$(resolve_lockfile "$1")
    
    if ((SLURM_NODEID == 0)); then
        jq '.status = "running"' "$lockfile" > "$lockfile.tmp" &&
        mv "$lockfile.tmp" "$lockfile"
    fi
}

# BAD — sequence-dependent global variables
JOBNAME=$1
LOCKFILE="$ENGINE_DIR/jobs/$JOBNAME/status.json"
if [[ -f "$LOCKFILE" ]]; then ...
```

## CLI tools use optarg

Bash CLI tools are mostly backend and not truly user facing. For consistency
use optarg for CLI processing:

GOOD:
```bash
#!/bin/bash

usage() {
    echo "Usage: $0 [-v] [-o output] [-n count] file..."
    echo ""
    echo "Options:"
    echo "  -v          Enable verbose output"
    echo "  -o output   Write results to output file"
    echo "  -n count    Number of lines to process"
    echo "  -h          Show this help message"
    exit 1
}

VERBOSE=false
OUTPUT=""
COUNT=0

while getopts ":vo:n:h" opt; do
    case $opt in
        v) VERBOSE=true ;;
        o) OUTPUT="$OPTARG" ;;
        n) COUNT="$OPTARG" ;;
        h) usage ;;
        \?) echo "Error: Invalid option -$OPTARG" >&2; usage ;;
        :)  echo "Error: Option -$OPTARG requires an argument" >&2; usage ;;
    esac
done

shift $((OPTIND - 1))

if [ $# -eq 0 ]; then
    echo "Error: No input files specified" >&2
    usage
fi

if [ "$VERBOSE" = true ]; then
    echo "Verbose: ON"
    echo "Output: ${OUTPUT:-stdout}"
    echo "Count: ${COUNT:-all}"
    echo "Files: $@"
    echo ""
fi

for file in "$@"; do
    if [ ! -f "$file" ]; then
        echo "Warning: '$file' not found, skipping" >&2
        continue
    fi

    if [ -n "$OUTPUT" ]; then
        if [ "$COUNT" -gt 0 ] 2>/dev/null; then
            head -n "$COUNT" "$file" >> "$OUTPUT"
        else
            cat "$file" >> "$OUTPUT"
        fi
    else
        if [ "$COUNT" -gt 0 ] 2>/dev/null; then
            head -n "$COUNT" "$file"
        else
            cat "$file"
        fi
    fi
done
```

### Variable Scope Rules

1. **Always use `local`** inside functions — even for accumulators and loop vars
2. **Use `readonly`** for constants at file scope
3. **Use `declare`** for associative arrays and integer attributes
4. **Export only what subprocesses need** — prefer passing as arguments

```bash
readonly CHECK_INTERVAL_SECS=10
readonly TARGET_ENDPOINTS=("/v1/chat/completions" "/v1/models")

some_function() {
    local job=$1
    local lockfile
    local status
    
    lockfile=$(resolve_lockfile "$job")
    status=$(jq -r '.status' "$lockfile")
}
```

### Naming Conventions

| Construct | Convention | Example |
|-----------|-----------|---------|
| Functions | `snake_case` | `update_status_running`, `resolve_lockfile`, `monitor_head` |
| File-scope constants | `UPPER_SNAKE_CASE` | `CHECK_INTERVAL_SECS` |
| Local variables | `snake_case` | `local lockfile`, `local vllm_pid` |
| Environment variables | `UPPER_SNAKE_CASE` | `VLLM_PID`, `SLURM_JOB_ID`, `IVLLM_TEST_CALL_LOG` |
| Library scripts | `snake_case.sh` | `utils.sh`, `vllm-env.sh`, `common-env.sh` |
| SLURM scripts | `snake_case.sh` | `slurm-vllm-serve.sh`, `slurm-vllm-setup.sh` |
| Login wrappers | `ivllm-*.sh` | `ivllm-serve.sh`, `ivllm-status.sh`, `ivllm-cancel.sh` |

### Error Handling

- Use `set -euo pipefail` in every SLURM script
- Do NOT `set -e` in library files (`utils.sh`, `vllm-env.sh`) — let the
  caller decide error handling
- Check exit codes explicitly with `||` or `if` rather than relying on `set -e`
- Use `trap` for cleanup — every SLURM script must have an EXIT trap
- Print diagnostic context before exiting

```bash
# GOOD
if ! mkdir -p "$work_dir"; then
    echo "[startup] fatal: could not create $work_dir" >&2
    exit 1
fi

# BAD — relying on set -e
mkdir -p "$work_dir"
```

### Lockfile Safety

Lockfile operations are the most critical bash code. Follow these rules:

1. **Atomic create**: Use `set -C` (noclobber) before creating a new lockfile
2. **Atomic update**: Write to a `.tmp` file, then `mv` over the original
   (`mv` is atomic on the same filesystem)
3. **Always validate**: Check `jq` output exists before `mv`
4. **Parallel filesystem**: Avoid race conditions — do not read-then-write
   the same file from multiple processes

```bash
# GOOD — atomic update
jq '.status = "running"' "$lockfile" > "$lockfile.tmp" &&
mv "$lockfile.tmp" "$lockfile"

# BAD — non-atomic, potential data loss
jq '.status = "running"' "$lockfile" > "$lockfile"
```

### Logging

- Use consistent prefix format: `[component] message`
- Write to stdout for normal operation, stderr for errors
- Include timestamps using `date +%H:%M:%S` in monitor output
- Do not echo raw variables that may contain special characters

```bash
# GOOD
echo "[startup] waiting for vllm /health api"
echo "[head] status file $lockfile is missing: shutting down head" >&2
printf "[%s-node %s] RAM: %s\n" "$(date +%H:%M:%S)" "$SLURM_NODEID" "$mem_summary"
```

### Testing (Bash)

- Run via `bash tests/bash/run.sh` (unit then sandboxed)
- Unit tests run directly on host; sandboxed tests run via bubblewrap
- Mock `srun`, `sbatch`, `scancel`, `squeue`, `scontrol`, `vllm`, `wget`, etc.
  via PATH shims in `tests/bash/shims/`
- Use `sandbox_run` / `sandbox_run_test` helpers from `tests/bash/lib/sandbox.sh`
- Two profiles: `login` (no SLURM vars) and `compute` (full SLURM env)
- Test with `set -x` for debugging, but do not commit with `set -x` enabled
- Each bash function in `lib/` should have corresponding test cases

```
tests/bash/
├── run.sh              — Test runner (unit + sandboxed)
├── lib/
│   ├── sandbox.sh      — bwrap harness
│   ├── assertions.sh   — assert_* helpers
│   └── test-utils.sh   — setup/cleanup for unit tests
├── shims/              — PATH shims for external commands
├── fixtures/           — Sample YAML configs
├── unit/               — Pure bash logic (no mocking)
└── sandboxed/          — Real jq/yq, mocked SLURM/vLLM
```

### When to use Bash vs TypeScript

| Task | Language | Reason |
|------|----------|--------|
| vLLM lifecycle (compute) | Bash | Runs on compute node, needs no runtime |
| Lockfile management | Bash | Atomic filesystem ops, jq integration |
| Monitor triad | Bash | Loop + health check + log parsing |
| Cache save/restore | Bash | tar + filesystem ops |
| CLI user interaction | TypeScript | Rich I/O, error handling, commander.js |
| SSH/SCP operations | TypeScript | Multiplexed SSH via `SshRemoteOps` |
| vLLM config parsing (login) | Bash | yq support, `ivllm-*.sh` wrappers |
| Version comparison | TypeScript | `src/semver.ts` — shared with CLI |
| Local ops (port/health) | TypeScript | `src/local-ops.ts` |

---

## Git and Workflow

### Commit Messages

```
<type>(<scope>): <description>

<optional body>
```

Types: `feat`, `fix`, `refactor`, `test`, `docs`, `chore`
Scope: `cli`, `bash`, `config`, `tunnel`, `monitor`, `docs`

### Branch Naming

- `feature/<phase>-<description>` — e.g. `feature/m1-bash-framework`
- `fix/<issue-number>-<description>` — e.g. `fix/issue-007-triton-patch`

### Pre-commit Checklist

- [ ] `bun test` passes (TypeScript + bash integration)
- [ ] `bash tests/bash/run.sh` passes (unit + sandboxed)
- [ ] New code has tests
- [ ] All tests follow TDD (test fails before implementation)
- [ ] No `console.log` in production code (use `console.error` for diagnostics)
- [ ] No `set -x` in committed bash scripts
- [ ] Version bumped (edit `package.json`)
- [ ] Committed with descriptive message

### Version Bumping

- Minor version for each commit during development
- Major version for v3.0.0 release
- Version is read from `package.json` — update it on each commit

```
# Increment minor
npm version minor  # or manually edit package.json
```
