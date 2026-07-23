# Implementation Issues — isambard-vllm v3

## Verified Issues

### Issue 1: CLI Commander.js Parameter Mismatch (CRITICAL)

**Location**: `src/index.ts` — all command handler functions  
**Severity**: CRITICAL — affects all CLI commands  
**Status**: Verified  

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
**Status**: Verified  

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
**Status**: Verified  

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
**Status**: Verified  

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
**Status**: Verified  

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
**Status**: Verified  

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

## Summary Table

| Issue | File | Line | Severity | Type | Verified |
|-------|------|------|----------|------|----------|
| 1. Commander.js param mismatch | src/index.ts | multiple | CRITICAL | Type mismatch | ✓ |
| 2. Missing awaits (cmd handlers) | src/index.ts | 111, 129 | HIGH | Async bug | ✓ |
| 3. Wrong setup option flag | src/backends/IsambardBareMetalBackend.ts | 49 | HIGH | Option mismatch | ✓ |
| 4. jq typo in lockfile reader | src/engine/lib/utils.sh | 192 | CRITICAL | Syntax error | ✓ |
| 5. Missing awaits (bootstrap) | src/backends/IsambardBareMetalBackend.ts | 47, 64, 101, 119, 148, 180 | HIGH | Async bug | ✓ |
| 6. Missing await (checkSSH) | src/backends/IsambardBareMetalBackend.ts | 33 | HIGH | Async bug | ✓ |

---

## Impact Analysis

### Immediate Blockers

**Issue 4 (jq typo)** blocks any operation that tries to read lockfile values, making the entire system non-functional for:
- Idle timeout detection (cannot read `idleTimeout` from lockfile)
- Status checking
- Configuration reading
- Job monitoring

**Issue 3 (setup option)** blocks the `ivllm setup` command completely.

**Issue 1 (Commander params)** blocks all CLI commands from working correctly because options are received as objects instead of scalar values.

**Issue 2 & 5 (missing awaits)** cause operations to exit or proceed before completing their work:
- Issue 2: `setup` and `cancel` commands exit before remote operation finishes
- Issue 5: Engine scripts may not be copied before being executed on remote

### Execution Order Fix

If fixing these issues, resolve in this order:
1. **Issue 4** — Fix jq syntax (prerequisite for all bash-based operations)
2. **Issue 5, 6** — Add awaits to bootstrap() and checkSSH() (prerequisite for remote operations)
3. **Issue 1** — Fix Commander.js parameter handling (prerequisite for CLI)
4. **Issue 2** — Add awaits to cmd handlers (refinement of CLI)
5. **Issue 3** — Fix setup script option (fixes specific command)

---

## Notes

- All issues are implementation bugs, not architectural problems
- No redesign is needed; fixes are straightforward
- The underlying logic and architecture remain sound
- These issues prevent current code from functioning but don't invalidate the v3 design
