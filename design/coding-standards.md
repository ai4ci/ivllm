# Coding Standards — isambard-vllm

This document defines coding conventions and patterns for the isambard-vllm
codebase, covering both TypeScript (CLI layer) and Bash (HPC runtime layer).

---

## TypeScript Standards

### Argument Parsing: Use `OptionParser`

Do NOT manually parse `process.argv`. The current code in `src/job.ts`
manually iterates `args` with `boolFlags` and `flags` objects. Replace
with a structured option parser.

**Recommended**: [commander](https://github.com/tj/commander.js) or
[node:util.parseArgs](https://nodejs.org/api/util.html#utilparseargsconfig)
(Node.js 20+ / Bun built-in).

```typescript
// GOOD — using commander
import { Command } from 'commander';

const program = new Command();
program
  .name('ivllm')
  .version(__VERSION__)
  .command('connect <job>')
  .option('--config <path>', 'Path to vLLM YAML config')
  .option('--interactive', 'Run with TTY binding')
  .option('--detach', 'Exit after starting')
  .option('--local-port <n>', 'Local port for SSH tunnel')
  .option('--dry-run', 'Preview without executing')
  .action((job, options) => {
    // options.config, options.interactive, options.detach, etc.
  });

// BAD — manual flag parsing with boolFlags and positional iteration
// See src/job.ts:parseStartArgs for what NOT to do
```

**Migration path**: Each command file (connect.ts, cancel.ts, setup.ts,
config.ts, agent.ts) defines its own `Command` and registers options. The
root `index.ts` is just the program definition with subcommand registration.

### Object-Oriented Patterns: Prefer Classes over Interface Proliferation

The current `src/types.ts` defines many standalone interfaces for data
that travels through different layers (e.g. `Credentials`, `Paths`,
`InferenceJobOptions`, `ServeOptions`, `JobDetails`, `EnvVarEntry`).
These are often replicated or extended across files.

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

**The Backend Interface pattern**

The v3 architecture defines a `Backend` interface (see ADR-111) for
abstracting different inference runtimes. When implementing a backend:

```typescript
// src/backends/interface.ts
export interface Backend {
  readonly name: string;
  connect(job: JobConfig): Promise<HostEndpoint>;
  cancel(jobName: string): Promise<void>;
  forceCancel(jobName: string): Promise<void>;
  status(jobName: string): Promise<LockfileV3 | null>;
  list(): Promise<LockfileV3[]>;
  setup?(version?: string): Promise<void>;
}

// src/backends/isambard-vllm.ts
export class IsambardVllmBackend implements Backend {
  readonly name = 'isambard-vllm';
  // ... implementation wrapping SSH ops + bash framework
}
```

This is the canonical example of the OOP approach: a class implementing a
well-defined interface, encapsulating all lifecycle behaviour.

**When to use an interface**:
- Simple data transfer objects with no behaviour
- External API request/response shapes
- Configuration blobs that are parsed once and consumed as-is
- Service contracts (like `Backend`) that have multiple implementations

### Naming Conventions

| Construct | Convention | Example |
|-----------|-----------|---------|
| Files | `kebab-case.ts` | `remote-ops.ts`, `vllm-config.ts` |
| Classes | `PascalCase` | `JobConfig`, `SessionState` |
| Interfaces | `PascalCase` | `InferenceJobOptions`, `ServeOptions` |
| Types | `PascalCase` | `JobStatus`, `LockfileState` |
| Functions | `camelCase` | `parseStartArgs`, `generateRandomHighPort` |
| Constants | `UPPER_SNAKE_CASE` | `SSH_MUX_OPTS`, `HEARTBEAT_INTERVAL_MS` |
| Private fields | `camelCase` (no `_` prefix) | `private yaml: ...` |
| Command files | `camelCase.ts` | `connect.ts`, `cancel.ts` |

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

### Testing

- One test file per source file (or per command)
- Test files in `tests/` mirror the `src/` structure
- Use the mock infrastructure (`--dry-run`) rather than mocking SSH
- Prefer integration-style tests over isolated unit tests for command files
- All tests must pass before commit

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
| Functions | `snake_case` | `update_status_running`, `resolve_lockfile` |
| File-scope constants | `UPPER_SNAKE_CASE` | `CHECK_INTERVAL_SECS` |
| Local variables | `snake_case` | `local lockfile`, `local vllm_pid` |
| Environment variables | `UPPER_SNAKE_CASE` | `VLLM_PID`, `SLURM_JOB_ID` |
| Script files | `kebab-case.sh` | `utils.sh`, `vllm-env.sh` |

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

- Use `design/prototype/test-vllm.sh` as the template
- Mock `srun`, `scancel`, and `vllm` with local implementations
- Test with `set -x` for debugging, but do not commit with `set -x` enabled
- Each bash function in `lib/` should have a corresponding test function
- Tests should pass with `bash tests/templates/lib/*.sh`

### When to use Bash vs TypeScript

| Task | Language | Reason |
|------|----------|--------|
| vLLM lifecycle | Bash | Runs on compute node, needs no runtime |
| Lockfile management | Bash | Atomic filesystem ops, jq integration |
| Monitor loops | Bash | Simple loop + curl + grep |
| Cache save/restore | Bash | tar + filesystem ops |
| CLI user interaction | TypeScript | Rich I/O, readline, error handling |
| SSH operations | TypeScript | Multiplexing, tunnel lifecycle |
| Config parsing | TypeScript | YAML support, validation |
| Version comparison | TypeScript | Semver parsing, sorting |
| Assistant launcher | TypeScript | Terminal UI, process spawning |

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

- [ ] `bun test` passes
- [ ] New code has tests
- [ ] All tests follow TDD (test fails before implementation)
- [ ] No `console.log` in production code (use `console.error` for diagnostics)
- [ ] No `set -x` in committed bash scripts
- [ ] Version bumped (`src/index.ts` already reads from `package.json`)
- [ ] Committed with descriptive message

### Version Bumping

- Minor version for each commit during development
- Major version for v3.0.0 release
- Version is read from `package.json` — update it on each commit

```
# Increment minor
npm version minor  # or manually edit package.json
```
