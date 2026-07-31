# Active Issues

This document tracks known, pre-existing bugs in the `isambard-vllm` repository.

## Known Issues

### 1. `tail -f` defaults to last 10 lines, silently dropping console history on every attach

**File:** `src/engine/ivllm-show-log.sh` (lines 69 and 73)

**Status:** already fixed in the working tree (uncommitted) — `tail -n +1 -f` is now used in both branches. Kept here for the record.

Before:
```bash
exec stdbuf -oL tail -f "${files[@]}"
...
coproc TAIL { stdbuf -oL tail -f "${files[@]}" 2>&1; }
```
After (current working tree):
```bash
exec stdbuf -oL tail -n +1 -f "${files[@]}"
...
coproc TAIL { stdbuf -oL tail -n +1 -f "${files[@]}" 2>&1; }
```

---

### 2. Raw `\r`-based progress output (vLLM/HF download bars) is passed through verbatim

**File:** `src/engine/ivllm-show-log.sh`, lines 68–79 (root cause: `src/engine/lib/slurm-vllm-serve.sh` line 12 — the whole SLURM job's stdout/stderr is redirected into `vllm.<node>.log` via `sbatch --output`/`--error`, so `vllm serve`'s raw `tqdm` progress output ends up in the log verbatim)

`vllm serve` / HuggingFace downloads emit progress updates terminated with `\r` (carriage return), not `\n`. A real terminal renders `\r` by jumping back to column 0 and overwriting, so a human watching `ivllm connect` only ever sees the *last* redraw of each burst — every earlier tick was genuinely shown, then erased in place. The bytes are intact in `vllm.<node>.log`; this is a rendering artifact, but it looks exactly like missing console lines, and it recurs throughout the run (download, shard loading, warmup), not just at startup.

**Current code (lines 67–79):**
```bash
# No marker — plain follow, runs until the client disconnects.
if [[ -z "$IVLLM_MATCH" ]]; then
    exec stdbuf -oL tail -n +1 -f "${files[@]}"
fi

# Marker mode — stream until the marker appears, then stop the tail.
coproc TAIL { stdbuf -oL tail -n +1 -f "${files[@]}" 2>&1; }
trap 'kill "$TAIL_PID" 2>/dev/null' EXIT INT TERM

awk -v target="$IVLLM_MATCH" '
    { print; fflush() }
    index($0, target) { exit 0 }
' <&"${TAIL[0]}"
```

**Fixed code** — insert a `\r`→`\n` normalizing filter into both pipelines, so every progress tick becomes its own real line instead of being overwritten in place (nothing is silently discarded from the console anymore — if the flood of ticks is considered too noisy, see the alternative below):

```bash
# No marker — plain follow, runs until the client disconnects.
if [[ -z "$IVLLM_MATCH" ]]; then
    exec stdbuf -oL tail -n +1 -f "${files[@]}" | stdbuf -oL tr '\r' '\n'
fi

# Marker mode — stream until the marker appears, then stop the tail.
coproc TAIL { stdbuf -oL tail -n +1 -f "${files[@]}" 2>&1 | stdbuf -oL tr '\r' '\n'; }
trap 'kill "$TAIL_PID" 2>/dev/null' EXIT INT TERM

awk -v target="$IVLLM_MATCH" '
    { print; fflush() }
    index($0, target) { exit 0 }
' <&"${TAIL[0]}"
```

**Alternative** (collapse each `\r`-delimited burst down to only its final tick, closer to what a live terminal shows, instead of expanding every tick into its own line) — replace `tr '\r' '\n'` in both places with:
```bash
sed -u 's/.*\r/\r/' | tr '\r' '\n'
```
`sed -u 's/.*\r/\r/'` strips everything up to and including the *last* `\r` on each input record, keeping only the final segment; piping that through `tr '\r' '\n'` then turns the one remaining leading `\r` into a normal line break. Pick whichever behaviour you want (full history vs. final-tick-only); the first (`tr '\r' '\n'` alone) is the minimal, non-lossy change.

---

### 3. `runRemote()` echoes raw stdout chunks instead of lines, and races with the next tail session's `inherit` stdio

**File:** `src/ops/SshRemoteOps.ts`, lines 61–89 (`runRemote`); handoff triggered from `src/backends/IsambardBareMetalBackend.ts`, lines 143–146 (`requestStart`)

**Current code — `src/ops/SshRemoteOps.ts` lines 61–89:**
```ts
async runRemote(
    command: string,
    options: RunRemoteOptions = { env: [], silent: true },
): Promise<RunRemoteResult> {
    return new Promise((resolve, reject) => {
        const target = `${this.config.username}@${this.config.loginHost}`;
        const fullCommand = this.makeFullCommand(command, options.env);

        const proc = spawn(
            'ssh',
            [...SSH_MUX_OPTS, '-o', 'BatchMode=yes', target, fullCommand],
            {
                stdio: ['ignore', 'pipe', 'inherit'],
            },
        );

        let stdout = '';

        proc.stdout?.on('data', (chunk: Buffer) => {
            stdout += chunk.toString();
            if (!options.silent) console.log(chunk.toString());
        });

        proc.on('error', reject);
        proc.on('close', (code) =>
            resolve({ exitCode: code ?? 0, stdout: stdout.trim() }),
        );
    });
}
```

**Problem (a):** `chunk.toString()` is echoed to `console.log` as-is, chunk-by-chunk, not line-by-line. `console.log` always appends its own `\n`, so a chunk boundary landing mid-line tears that line into two fragments, each getting a spurious extra newline.

**Fix (a) — buffer partial lines, only log complete ones, flush the remainder on close:**
```ts
async runRemote(
    command: string,
    options: RunRemoteOptions = { env: [], silent: true },
): Promise<RunRemoteResult> {
    return new Promise((resolve, reject) => {
        const target = `${this.config.username}@${this.config.loginHost}`;
        const fullCommand = this.makeFullCommand(command, options.env);

        const proc = spawn(
            'ssh',
            [...SSH_MUX_OPTS, '-o', 'BatchMode=yes', target, fullCommand],
            {
                stdio: ['ignore', 'pipe', 'inherit'],
            },
        );

        let stdout = '';
        let lineBuffer = '';

        proc.stdout?.on('data', (chunk: Buffer) => {
            const text = chunk.toString();
            stdout += text;

            if (options.silent) return;

            lineBuffer += text;
            const lines = lineBuffer.split('\n');
            lineBuffer = lines.pop() ?? ''; // keep the trailing partial line buffered
            for (const line of lines) console.log(line);
        });

        proc.on('error', reject);
        proc.on('close', (code) => {
            if (!options.silent && lineBuffer.length > 0) {
                console.log(lineBuffer); // flush any unterminated trailing line
            }
            resolve({ exitCode: code ?? 0, stdout: stdout.trim() });
        });
    });
}
```

**Problem (b):** `requestStart()`'s monitor call uses this `runRemote` (`stdio:'pipe'` + manual `console.log`), then `cmdConnect` immediately opens a *second*, unrelated tail via `watchLog()` → `runRemoteSync` (`stdio:'inherit'`), writing straight to the same fd. When stdout isn't a TTY, `console.log`'s writes can still be queued when the second child starts writing directly — risking interleaved/out-of-order console output, on top of restarting `tail -f` a second time on the same run.

**Current code — `src/backends/IsambardBareMetalBackend.ts` lines 140–146:**
```ts
if (monitor)
    await this.ops.runRemote(
        `${this.remoteEngine}/ivllm-show-log.sh -j "${job}" -m "[startup] Startup complete"`,
        { env: this.envs, silent: false },
    );
```

**Fix (b) — use the same `runRemoteSync`/`inherit` mechanism as `watchLog`, so there's no second writer racing on the fd:**
```ts
if (monitor) {
    const monitorProc = this.ops.runRemoteSync(
        `${this.remoteEngine}/ivllm-show-log.sh -j "${job}" -m "[startup] Startup complete"`,
        this.envs,
    );
    await new Promise<void>((resolve) => monitorProc.once('close', () => resolve()));
}
```
This drops the `pipe`+`console.log` re-echo for this call entirely (its return value/`stdout` capture was never used by the caller), so there's only ever one writer on the inherited fd for the whole `connect` startup sequence — fix (a) above still applies to every other `runRemote(..., { silent: false })` caller (e.g. `setup`, `requestCancel`).

---
