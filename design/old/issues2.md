# Implementation Issues — isambard-vllm v3

## Verified Issues

### Issue 1: CLI Commander.js Parameter Mismatch (CRITICAL)

**Location**: `src/index.ts` — all command handler functions  
**Severity**: CRITICAL — affects all CLI commands  
**Status**: Closed - FIXED

**Problem**:
Commander.js passes options as an object to `.action()` handlers, but the function signatures expect individual parameters. When a command like:

```typescript
program.command('cancel')
  .argument('<jobName>', ...)
  .option('--force', ...)
  .action(cmdCancel)
```

is invoked as `ivllm2 cancel qwen36 --force`, Commander.js invokes:
```typescript
cmdCancel('qwen36', { force: true, ... })
```

But the function signature is:
```typescript
async function cmdCancel(jobName: string, force: boolean): Promise<void>
```

This means `force` parameter receives an `OptionValues` object, not a boolean. The object is then passed to `backend.requestCancel(jobName, force)` where a boolean is expected, causing type mismatch and runtime errors.

**Affected Functions**:
- `cmdCancel()` — receives `options: OptionValues` as second param, not `force: boolean`
- `cmdSetup()` — receives `options: OptionValues` as second param, not `force: boolean`
- `cmdConfig()` — receives `options: OptionValues` as single param, not 4 separate params
- `cmdConnect()` — receives `options: OptionValues` as second param, not individual flags

**Fix Required**:
Change all command handlers to receive the correct signature. Example for `cmdCancel`:

```typescript
// Current (wrong):
async function cmdCancel(jobName: string, force: boolean): Promise<void> {

// Correct:
async function cmdCancel(jobName: string, options: { force: boolean }): Promise<void> {
    const backend = new IsambardBareMetalBackend(config);
    await backend.requestCancel(jobName, options.force);
}
```

For `cmdConfig` which has no arguments:
```typescript
// Current (wrong):
async function cmdConfig(host?: string, user?: string, path?: string, token?: string): Promise<void>

// Correct:
async function cmdConfig(options: { loginHost?: string; username?: string; projectDir?: string; hfToken?: string }): Promise<void> {
    const config = loadCredentials();
    if (options.loginHost) config.loginHost = options.loginHost;
    if (options.username) config.username = options.username;
    if (options.projectDir) config.projectDir = options.projectDir;
    if (options.hfToken) config.hfToken = options.hfToken;
    saveConfig(config);
    console.log('Configuration saved.');
}
```

---

### Issue 2: Missing `await` in Command Handlers (HIGH)

**Location**: `src/index.ts` lines 111 and 129  
**Severity**: HIGH — promises are not awaited, commands exit before completion  
**Status**: Closed - FIXED

**Problem**:
`cmdCancel()` and `cmdSetup()` do not await the backend method promises:

```typescript
// Line 111 in cmdCancel — missing await
backend.requestCancel(jobName, force);

// Line 129 in cmdSetup — missing await
backend.setup(vllmVersion, force);
```

This causes the CLI to exit immediately without waiting for the remote operation to complete. The operations run asynchronously in the background (or fail silently).

**Fix**:
Add `await` keyword:
```typescript
await backend.requestCancel(jobName, force);
await backend.setup(vllmVersion, force);
```

---

### Issue 3: Wrong Option Flag in Setup Script Call (HIGH)

**Location**: `src/backends/IsambardBareMetalBackend.ts` line 49  
**Severity**: HIGH — setup script fails because wrong option is passed  
**Status**: Closed

**Problem**:
The TypeScript backend calls `ivllm-setup.sh` with `-j` for the version argument:

```typescript
// IsambardBareMetalBackend.ts line 49:
`${this.remoteEngine}/ivllm-setup.sh -j "${version}"${force ? ' -f' : ''}`,
```

But the bash script expects `-v`:

```bash
# ivllm-setup.sh:
while getopts "v:flh" opt; do
    case $opt in
        v) IVLLM_VERSION="$OPTARG" ;;
```

Calling with `-j` will result in a "no version provided" error because the `-j` flag is not recognized.

**Fix**:
Change the option flag from `-j` to `-v`:
```typescript
`${this.remoteEngine}/ivllm-setup.sh -v "${version}"${force ? ' -f' : ''}`,
```

**Related Note**: The other scripts (`ivllm-serve.sh`, `ivllm-cancel.sh`) correctly use `-j` for job names, so only the setup script has this mismatch.

---

### Issue 4: jq Syntax Error in Lockfile Reading (CRITICAL)

**Location**: `src/engine/lib/utils.sh` line 192  
**Severity**: CRITICAL — function cannot read lockfile values  
**Status**: Closed - FIXED

**Problem**:
The `get_job_status_setting()` function has a typo in the jq command:

```bash
# Current (wrong):
jq r "$2" "$lockfile" 2>/dev/null || echo ""

# jq interprets 'r' as a filter, not the -r flag option
```

Running this command produces:
```
jq: error: r/0 is not defined at <top-level>, line 1:
r
jq: 1 compile error
```

This function is used throughout the bash framework to read values from the lockfile:
- Reading `idleTimeout` in `monitor_head()` (line 642)
- Reading job configuration in various places
- Reading status values

**Impact**: Any code path that calls `get_job_status_setting()` will fail.

**Fix**:
Change `jq r` to `jq -r`:
```bash
jq -r "$2" "$lockfile" 2>/dev/null || echo ""
```

**Verification**:
```bash
# Wrong:
echo '{"test": 123}' | jq r '.test'
# Error: jq: error: r/0 is not defined

# Correct:
echo '{"test": 123}' | jq -r '.test'
# Output: 123
```

---

### Issue 5: Missing `await` for bootstrap() Calls (HIGH)

**Location**: `src/backends/IsambardBareMetalBackend.ts` lines 47, 64, 101, 119, 148, 180  
**Severity**: HIGH — bootstrap operation runs asynchronously without being awaited  
**Status**: Closed - FIXED

**Problem**:
The `bootstrap()` method is declared as `async Promise<void>` (line 32), but it's called without `await` throughout the class:

```typescript
// Line 47 in setup():
async setup(version: string, force?: boolean): Promise<void> {
    this.bootstrap();  // ← missing await
    
    const { stdout, exitCode } = await this.ops.runRemote(
```

The same pattern appears in:
- Line 47: `setup()`
- Line 64: `connect()`
- Line 101: `requestCancel()`
- Line 119: `requestStart()`
- Line 148: `getAllJobStatus()`
- Line 180: `watchLog()`

Since `bootstrap()` copies the entire `engine/` directory to the remote HPC via `copyDirectory()`, not awaiting it means the subsequent operations (like `runRemote()`) may execute before the engine scripts are transferred.

**Fix**:
Add `await` to all `bootstrap()` calls:
```typescript
async setup(version: string, force?: boolean): Promise<void> {
    await this.bootstrap();  // ← add await
    
    const { stdout, exitCode } = await this.ops.runRemote(
```

---

### Issue 6: Missing `await` for checkSSH() Call (HIGH)

**Location**: `src/backends/IsambardBareMetalBackend.ts` line 33  
**Severity**: HIGH — SSH connectivity check not awaited  
**Status**: Closed - FIXED

**Problem**:
The `checkSSH()` method is declared as `async Promise<boolean>` in `SshRemoteOps`, but it's called without `await` in `bootstrap()`:

```typescript
// Line 33 in bootstrap():
async bootstrap(): Promise<void> {
    this.ops.checkSSH();  // ← missing await
    if (!this.bootstrapped) {
        const currentDir = import.meta.dir;
        const enginePath = path.resolve(currentDir, '../engine');
        await this.ops.copyDirectory(
            enginePath,
            this.creds.projectDir,
            'up',
        );
```

Since `checkSSH()` performs async SSH operations (calling `runRemote()` which awaits), not awaiting it means `copyDirectory()` may execute before SSH connectivity is verified.

**Fix**:
Add `await` to the `checkSSH()` call:
```typescript
async bootstrap(): Promise<void> {
    await this.ops.checkSSH();  // ← add await
    if (!this.bootstrapped) {
```

---

## Issues Discovered While Building the Bash Mock Test Environment

The following were found while designing the bubblewrap sandbox and tracing
real external-tool call semantics (real `yq 3.4.1` binary, as confirmed
installed to match the HPC). They were first captured as intentional red
tests (see `tests/bash/sandboxed/test-config.sh`, `test-vllm-env.sh`, and
`test-monitor-head.sh`) and then fixed — all now green, but the tests
remain as regression guard for each category of bug.

### Issue 7: `get_job_config_setting()` passes yq arguments in the wrong order (CRITICAL)

**Location**: `src/engine/lib/utils.sh` — `get_job_config_setting()`  
**Severity**: CRITICAL — every config value read returns empty  
**Status**: Closed

**Problem**:
The HPC has **yq 3.4.1** (mikefarah/yq v3, confirmed against
https://github.com/mikefarah/yq/releases/tag/3.4.1). Its `read`/`r` subcommand
takes the file first, then the path expression:

```
Usage:
  yq read [yaml_file] [path_expression] [flags]
```

But `get_job_config_setting()` calls it with the arguments reversed:

```bash
yq r "$2" "$file" 2>/dev/null || echo ""
#    ^path  ^file   ← swapped
```

Verified directly:
```bash
$ yq r ".model" test.yaml   # wrong order (current code) → exit 1
$ yq r test.yaml ".model"   # correct order → prints the value
```

Because the call is wrapped in `|| echo ""`, the failure is swallowed
silently — every call to `get_job_config_setting` (model, idle-timeout,
min-vllm-version, tensor/pipeline/data-parallel-size, etc.) always returns an
empty string instead of the real value.

**Fix**: swap the argument order to `yq r "$file" "$2"`.

---

### Issue 8: `resolve_stripped_job_config()` uses yq v4 jq-style syntax against yq v3 (CRITICAL)

**Location**: `src/engine/lib/utils.sh` — `resolve_stripped_job_config()`  
**Severity**: CRITICAL — vllm.yaml.clean is never generated  
**Status**: Closed

**Problem**:
```bash
yq 'del(.env, .nnodes, .min-vllm-version, .ivllm, .idle-timeout, .metadata)' "$file" > "$output_file"
```

This bare jq-filter-style invocation (`del(...)` piping) is yq **v4**
syntax. yq v3.4.1 has no bare-filter mode and no multi-path `del`; it only
has a `delete`/`d` subcommand taking exactly one path per invocation:

```
yq delete [yaml_file] [path_expression]
```

Verified directly — the v4-style call fails outright:
```
$ yq 'del(.env, .nnodes, .min-vllm-version, .ivllm, .idle-timeout, .metadata)' test.yaml
Error: unknown command "del(...)" for "yq"
```

and even the single-path form only deletes the first path listed, silently
ignoring the rest:
```
$ yq d test.yaml env nnodes min-vllm-version idle-timeout
# only 'env' is removed; nnodes/min-vllm-version/idle-timeout remain
```

This breaks `vllm.yaml.clean`, which is passed directly to `vllm serve
--config`, meaning `env`, `nnodes`, `min-vllm-version`, `ivllm`,
`idle-timeout`, and `metadata` blocks are never stripped and would reach
`vllm serve` as unrecognised keys.

**Fix** (v3-compatible): chain single-path deletes through stdin, e.g.
```bash
yq d "$file" env | yq d - nnodes | yq d - min-vllm-version | yq d - idle-timeout | yq d - metadata | yq d - ivllm > "$output_file"
```

---

### Issue 9: `get_job_config_exports()` uses yq v4 jq-style syntax against yq v3 (CRITICAL)

**Location**: `src/engine/lib/utils.sh` — `get_job_config_exports()`  
**Severity**: CRITICAL — `env:` block from vllm.yaml is never exported  
**Status**: Closed

**Problem**:
```bash
yq '( .env // {} ) | to_entries | .[] | "export " + .key + "=\"" + .value + "\""' "$file"
```

Same root cause as Issue 8 — `// {}`, `to_entries`, and filter piping are yq
v4 syntax, unsupported by the installed yq v3.4.1:
```
$ yq '( .env // {} ) | to_entries | .[] | ...' test.yaml
Error: unknown command "( .env // {} ) | to_entries | ..." for "yq"
```

This means any user-supplied `env:` block in `vllm.yaml` (used to pass
custom environment variables into the SLURM job) is silently never applied.

**Fix** (v3-compatible): use `-p pv` (path+value) read mode over `env.*` and
build export lines in bash, e.g.
```bash
yq r -p pv "$file" 'env.*' 2>/dev/null | sed -E 's/^env\.([^ ]+) (.*)$/export \1="\2"/'
```

---

### Issue 10: `ivllm-serve.sh` calls `ivllm-get-model.sh` without the required `-m` flag (CRITICAL)

**Location**: `src/engine/ivllm-serve.sh` line 52  
**Severity**: CRITICAL — model download is always skipped/broken  
**Status**: Closed

**Problem**:
```bash
(source $here/ivllm-get-model.sh "$model") || exit 1
```

`$model` is passed as a bare positional argument. But `ivllm-get-model.sh`
only parses its model name via `getopts "m:t:l:h"` and the `-m` flag:

```bash
export HF_MODEL=""
while getopts "m:t:l:h" opt; do
    case $opt in
        m) HF_MODEL="$OPTARG" ;;
```

Since `$model` (e.g. `Qwen/Qwen2.5-7B-Instruct`) does not start with `-`,
`getopts` terminates immediately without consuming it, leaving
`HF_MODEL=""`. The script then proceeds to check/download an **empty**
model name against the HuggingFace cache.

**Fix**:
```bash
(source $here/ivllm-get-model.sh -m "$model") || exit 1
```

---

### Issue 11: `common-env.sh` references `$NVSHMEM_DIR` before it is defined (MEDIUM)

**Location**: `src/engine/lib/common-env.sh` lines 58 and 67  
**Severity**: MEDIUM — latent bug, currently masked because these scripts are sourced without `set -u`  
**Status**: Closed

**Problem**:
```bash
# line 58:
export CMAKE_PREFIX_PATH="$NVSHMEM_DIR/lib/cmake:${CMAKE_PREFIX_PATH:-}"
# ...
# line 67:
export NVSHMEM_DIR="$NVHPC_ROOT/comm_libs/$CUDA_VERSION/nvshmem"
```

`$NVSHMEM_DIR` is read on line 58, nine lines before it is ever assigned on
line 67. In the real system this is silently tolerated (bash treats an
unset variable as empty when `set -u`/`nounset` is not active, so
`CMAKE_PREFIX_PATH` ends up as `/lib/cmake:...` — a malformed, effectively
unusable path, but not a crash). It was caught while building the bash
sandbox test harness (`tests/bash/lib/sandbox.sh`), which runs test bodies
under `set -uo pipefail` — sourcing `common-env.sh` there fails hard with
`NVSHMEM_DIR: unbound variable`.

**Fix**: move the `NVSHMEM_DIR` assignment (line 67) above its first use
(line 58).

**Note**: the sandboxed preamble tests
(`tests/bash/sandboxed/test-vllm-env.sh`) deliberately run with `set +u`
around the `source` calls to match the *actual* invocation context used by
`run_head_vllm.sh`/`run_worker_vllm.sh` (which do not set `-u`), so this
latent bug does not block those tests. It is recorded here rather than
silently worked around.

---

### Issue 12: `resolve_nvhpc_root()` writes its error message to stdout, corrupting the captured value (HIGH)

**Location**: `src/engine/lib/utils.sh` — `resolve_nvhpc_root()` line 89  
**Severity**: HIGH — `NVHPC_ROOT`/`CUDA_HOME`/`PATH`/`LD_LIBRARY_PATH` become garbage instead of empty when the NVHPC SDK isn't installed  
**Status**: Closed

**Problem**:
```bash
resolve_nvhpc_root() {
    local nvhpcDir=$(resolve_nvhpc_dir)
    if [[ ! -d "$nvhpcDir/Linux_aarch64/26.3" ]]; then
      echo "NVHPC SDK version 26.3 is not installed. please run ivllm setup."
      return 1
    fi
    echo "$nvhpcDir/Linux_aarch64/26.3"
}
```

The diagnostic message on line 89 is written with a plain `echo` (stdout),
not `echo ... >&2` (stderr). `common-env.sh` calls this via command
substitution:
```bash
export NVHPC_ROOT=$(resolve_nvhpc_root)
```

Command substitution captures **stdout only** — so when the SDK isn't
installed, `NVHPC_ROOT` does not end up empty; it ends up literally
containing the sentence `"NVHPC SDK version 26.3 is not installed. please
run ivllm setup."`. Every downstream path built from it is then silently
corrupted:
```bash
export CUDA_HOME="$NVHPC_ROOT/cuda/$CUDA_VERSION"
# → "NVHPC SDK version 26.3 is not installed. please run ivllm setup./cuda/12.9"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="...:$NVHPC_ROOT/cuda/$CUDA_VERSION/compat:..."
```

Because none of these library files use `set -e` (by design — see
`design/coding-standards.md`), nothing stops the script here; it proceeds
with a thoroughly broken environment instead of failing fast or leaving
variables genuinely empty.

**Fix**: redirect the diagnostic to stderr:
```bash
echo "NVHPC SDK version 26.3 is not installed. please run ivllm setup." >&2
```

**Verified via the sandboxed test suite** —
`tests/bash/sandboxed/test-vllm-env.sh`'s
`common_env_missing_nvhpc_falls_through_empty` test currently fails,
showing `NVHPC_ROOT` containing the error sentence instead of being empty.
The sandboxed test
`tests/bash/sandboxed/test-vllm-env.sh`'s
`common_env_missing_nvhpc_falls_through_empty` was written first and failed
because of this bug; it now passes after the fix.

---

### Issue 13: `get_job_status_setting()` called without the leading `.` in `tidy_up`, `monitor_startup`, `monitor_head` (CRITICAL — most severe finding)

**Location**: `src/engine/lib/utils.sh` — lines 469, 470, 555, 556, 617, 672  
**Severity**: CRITICAL — the entire monitor triad and shutdown trap are non-functional as written  
**Status**: Closed

**Problem**:
`get_job_status_setting()` is documented as requiring a leading `.` (it is a
jq filter):
```bash
# Usage: local value=$(get_job_status_setting "$job" ".fieldName")
```
and most call sites follow this (`.status` at lines 289 and 789). But
several critical call sites omit the leading dot:

| Line | Call | Function |
|------|------|----------|
| 469 | `get_job_status_setting "$job" "vllmPid"` | `tidy_up()` |
| 470 | `get_job_status_setting "$job" "slurmJobId"` | `tidy_up()` |
| 555 | `get_job_status_setting "$job" "serverPort"` | `monitor_startup()` |
| 556 | `get_job_status_setting "$job" "model"` | `monitor_startup()` |
| 617 | `get_job_status_setting "$job" "status"` | `monitor_startup()` (error branch) |
| 642 | `get_job_status_setting "$job" "idleTimeout"` | `monitor_head()` |
| 672 | `get_job_status_setting "$job" "vllmPid"` | `monitor_head()` |

A bare identifier like `vllmPid` is not valid jq syntax on its own — jq
tries to interpret it as a function call:
```
$ echo '{"vllmPid": 12345}' | jq -r "vllmPid"
jq: error: vllmPid/0 is not defined at <top-level>, line 1:
vllmPid
jq: 1 compile error

$ echo '{"vllmPid": 12345}' | jq -r ".vllmPid"
12345
```

Because `get_job_status_setting` wraps the call in `|| echo ""`, this jq
compile error is silently swallowed and the function returns an **empty
string** for every one of these fields, every time.

**Impact** (each confirmed empirically via the sandboxed test suite —
`tests/bash/sandboxed/test-monitor-head.sh`):

- **`monitor_head()`**: `vllm_pid` is always `""`. The very first loop
  iteration hits `kill -0 ""` (always fails/false), so monitor_head
  **always immediately reports "lost contact with vLLM process" and
  shuts the job down**, regardless of whether vLLM is actually running.
  The real idle-timeout and cancel-detection logic further down the loop
  is never reached in practice because this check fires first, every
  cycle. `idle_timeout` is also always `""`, which would separately break
  the idle-timeout branch's `[ "$idle_timeout" -ge 0 ]` check even if
  execution reached it.
- **`monitor_startup()`**: `server_port` and `model` are always `""`, so
  the `/health` and warmup `/v1/chat/completions` requests target a
  malformed URL (`http://localhost:/health`, no port) and an empty model
  name — startup monitoring cannot work.
- **`tidy_up()`**: `pid` and `slurm_job_id` are always `""`, so the exit
  trap can never `kill` the real vLLM process or `scancel` the real SLURM
  job — cleanup silently does nothing on these two fronts.

**This was found directly through the new bash sandbox test suite** —
writing a real test for `monitor_head` (background a real "vLLM pid"
stand-in process, run `monitor_head` in the background, kill/cancel/wait on
it for real) surfaced this immediately: 4 of 5
`tests/bash/sandboxed/test-monitor-head.sh` tests currently fail with the
same symptom ("vLLM process () has gone away" fires on the very first
check, before the test's actual scenario — cancel, idle timeout, active
traffic — ever gets evaluated). This is exactly the class of bug the
design's mock-harness-based (bash function override) approach could not
have caught, since it never exercised the real jq compile-error path.

**Fix**: add the missing leading `.` at all six call sites, e.g.:
```bash
pid=$(get_job_status_setting "$job" ".vllmPid")
slurm_job_id=$(get_job_status_setting "$job" ".slurmJobId")
...
server_port=$(get_job_status_setting "$job" ".serverPort")
model=$(get_job_status_setting "$job" ".model")
...
idle_timeout=$(get_job_status_setting "$job" ".idleTimeout")
...
vllm_pid=$(get_job_status_setting "$job" ".vllmPid")
```

This was the most severe bug — the entire monitor triad and exit trap
were non-functional. `tests/bash/sandboxed/test-monitor-head.sh` wrote
against real subprocess/signal behaviour caught it: backgrounding the
fake-vllm pid stand-in as a real `sleep` process, then verifying that
`monitor_head` actually reaches the cancel/idle/lockfile branches
instead of prematurely failing on a static integer pid.

---

## Summary Table (updated)

| Issue | File | Line | Severity | Type | Status |
|-------|------|------|----------|------|--------|
| 1. Commander.js param mismatch | src/index.ts | multiple | CRITICAL | Type mismatch | Closed |
| 2. Missing awaits (cmd handlers) | src/index.ts | 111, 129 | HIGH | Async bug | Closed |
| 3. Wrong setup option flag | src/backends/IsambardBareMetalBackend.ts | 49 | HIGH | Option mismatch | Closed |
| 4. jq typo in lockfile reader | src/engine/lib/utils.sh | 192 | CRITICAL | Syntax error | Closed |
| 5. Missing awaits (bootstrap) | src/backends/IsambardBareMetalBackend.ts | 47, 64, 101, 119, 148, 180 | HIGH | Async bug | Closed |
| 6. Missing await (checkSSH) | src/backends/IsambardBareMetalBackend.ts | 33 | HIGH | Async bug | Closed |
| 7. yq arg order reversed | src/engine/lib/utils.sh | `get_job_config_setting` | CRITICAL | Argument order | Closed |
| 8. yq v4 syntax vs v3 binary (del) | src/engine/lib/utils.sh | `resolve_stripped_job_config` | CRITICAL | Version incompatibility | Closed |
| 9. yq v4 syntax vs v3 binary (to_entries) | src/engine/lib/utils.sh | `get_job_config_exports` | CRITICAL | Version incompatibility | Closed |
| 10. Missing `-m` flag calling ivllm-get-model.sh | src/engine/ivllm-serve.sh | 52 | CRITICAL | Missing CLI flag | Closed |
| 11. `$NVSHMEM_DIR` used before defined | src/engine/lib/common-env.sh | 58 (used), 67 (defined) | MEDIUM | Ordering bug | Closed |
| 12. `resolve_nvhpc_root` error goes to stdout | src/engine/lib/utils.sh | 89 | HIGH | Wrong stream | Closed |
| 13. Missing leading `.` in jq filter (tidy_up/monitor_startup/monitor_head) | src/engine/lib/utils.sh | 469,470,555,556,617,642,672 | CRITICAL | Argument/syntax bug | Closed |

Issues 7-13 were all discovered while building the bash sandbox test harness
and writing real tests against it (see `design/testing.md`). The tests that
original flagged each bug are now green — see
`tests/bash/sandboxed/test-config.sh` (issues 7-9 & 11),
`tests/bash/sandboxed/test-vllm-env.sh` (issues 11 & 12), and
`tests/bash/sandboxed/test-monitor-head.sh` (issue 13).


