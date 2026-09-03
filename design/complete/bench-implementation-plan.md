# `ivllm bench` — Implementation Plan

**Status, updated 2026-09-03**: Implemented, lifecycle `experimental`, end-to-end
testing still outstanding (confirmed with Rob). The plan below has been carried
out — `design/prototype/ivllm-bench.sh` was promoted to `src/engine/ivllm-bench.sh`,
and `requestBenchmark`/`getBenchmarkStatus`/`fetchBenchmarkResults` are wired
into `IsambardBareMetalBackend.ts` and the real CLI (`bench submit|status|results`
in `src/index.ts`). Kept for historical context on the design intent rather
than rewritten — the remaining real work is E2E testing, not more design.

## 1. What the prototype already proves

`design/prototype/ivllm-bench.sh` is a self-contained, standalone login-node
bash script (no existing production file changes beyond one already-applied
change to `capture_job_diagnostics()` — see its own header) that:

- Takes a directory of `*.yaml`/`*.yml` configs, one per candidate.
- Builds a "benchmark" shadow project directory as a subfolder of that
  comparison directory, with the expensive shared subdirectories
  (`engine/vllm`, `engine/nvhpc`, `engine/rdma`, `model`) symlinked back to
  the real project directory.
- Prefetches every unique `.model` across configs once, sequentially, before
  submitting anything (avoids a concurrent-download race across jobs sharing
  a model — the common case for a tuning comparison, not an edge case).
- Submits every config as an ordinary batch-partition job via the real,
  unmodified `ivllm-serve.sh -b`.
- Waits for each to reach `running`, benchmarks it in place via
  `srun --overlap --jobid=<slurmJobId>` (co-located `vllm bench serve`, no
  network hop), then gracefully cancels it and archives full diagnostics
  (`capture_job_diagnostics()`, which now sweeps everything in the job
  directory, including `bench.json` and `IVLLM_DEBUG_LEVEL>=2`'s
  `debug/pyspy-*.log` dumps).
- Writes a periodically-refreshed `benchmarking_status.json` (schema below)
  and is designed to be launched **detached** — a caller polls the status
  file rather than waiting on the script or SSH session.
- Prints (and saves to `results/summary.txt`) a table with one row **per job
  per historical run**, not just the latest — rerunning a comparison after
  tweaking a config is the expected workflow.

This document is about exposing that script through `ivllm bench
submit|status|results`, not about changing what it does.

## 2. Bash side: promote the prototype

- `design/prototype/ivllm-bench.sh` → `src/engine/ivllm-bench.sh`, sitting
  alongside `ivllm-serve.sh`/`ivllm-cancel.sh`/`ivllm-get-model.sh` (same
  per-user deployment location, `~/.local/bin/`, via the existing
  `Backend.bootstrap()` → `copyDirectory(enginePath, remoteEngine, 'up')`
  path — no new deployment mechanism needed).
- No further `utils.sh` changes anticipated — the one production change the
  prototype needed (`capture_job_diagnostics()` sweeping the whole job dir)
  is already applied.
- Per AGENTS.md: tests for the promoted script should exist and pass
  *before* this is considered done (see §4) — this is a "move + wire up",
  not a rewrite, so most of the validation is "does it still do what the
  prototype already demonstrated," not new logic design.

## 3. TypeScript surface

### 3.1 New type (`src/types.ts`)

```typescript
/** Per-job status inside a BenchmarkStatus snapshot. */
export interface BenchmarkJobStatus {
    status: string;           // LockfileV3.status values, or "unknown"
    reason: string | null;
}

/** Parsed benchmarking_status.json — see ivllm-bench.sh's own doc comment
 *  for the authoritative schema; this type must stay in sync with it. */
export interface BenchmarkStatus {
    pid: number;
    updated: string;          // ISO8601 UTC
    complete: boolean;        // true once EVERY job has a terminal status
    counts: {
        pending: number;
        initialising: number;
        running: number;
        stopped: number;
        failed: number;
    };
    jobs: Record<string, BenchmarkJobStatus>;
}
```

### 3.2 `Backend` abstract class — three new methods

No new `Credentials` field is needed — `ivllm-bench.sh` computes its shadow
project directory relative to the comparison directory argument itself
(`$COMPARISON_DIR/benchmark`), not from a separate top-level project
setting, so the existing `projectDir` credential is sufficient (passed as
`IVLLM_PROJECTDIR`, same as every other command).

```typescript
/**
 * Submit a benchmark comparison — upload configs, launch ivllm-bench.sh
 * detached (fire-and-forget; does NOT wait for jobs to reach "running",
 * let alone complete).
 * @param comparison — Comparison name; becomes the remote directory name
 *   under `${projectDir}/engine/comparisons/<comparison>/`
 * @param configs — Local paths to one or more vLLM YAML configs
 * @param options.time — SLURM `--time` per job (passed through to
 *   ivllm-bench.sh's own `IVLLM_BENCH_TIME`), default matches ADR-118 (2h)
 */
abstract requestBenchmark(
    comparison: string,
    configs: string[],
    options?: { time?: string },
): Promise<void>;

/**
 * Poll a benchmark comparison's progress — cheap, safe to call often.
 * @param comparison — Comparison name
 * @returns Parsed benchmarking_status.json
 * @throws Error if the comparison directory / status file doesn't exist
 *   (e.g. wrong name, or submit failed before the first write)
 */
abstract getBenchmarkStatus(comparison: string): Promise<BenchmarkStatus>;

/**
 * Fetch a benchmark comparison's results, only if complete.
 * @param comparison — Comparison name
 * @param localDest — Local destination directory
 * @returns A discriminated result — `ready: false` carries the current
 *   status snapshot instead of throwing, since "still running" is an
 *   expected state for a fire-and-forget job, not an exceptional one
 */
abstract fetchBenchmarkResults(
    comparison: string,
    localDest: string,
): Promise<
    | { ready: true; path: string }
    | { ready: false; status: BenchmarkStatus }
>;
```

### 3.3 `IsambardBareMetalBackend` — implementation sketch

```typescript
private remoteComparisonDir(comparison: string): string {
    return `${this.creds.projectDir}/engine/comparisons/${comparison}`;
}

async requestBenchmark(
    comparison: string,
    configs: string[],
    options: { time?: string } = {},
): Promise<void> {
    await this.bootstrap();
    const remoteEngine = await this.getRemoteEngine();
    const remoteDir = this.remoteComparisonDir(comparison);

    // Upload every config — same copyFile call requestStart() already uses
    // for a single config, just once per file here.
    for (const config of configs) {
        if (!fs.existsSync(config)) {
            throw new Error(`no configuration file found at: ${config}`);
        }
        const remoteConfig = `${remoteDir}/${path.basename(config)}`;
        await this.ops.copyFile(config, remoteConfig);
    }

    // Detached launch — see ivllm-bench.sh's own "Fire-and-forget client
    // contract" comment block for why (no GPU work of its own, can run for
    // hours, must survive this SSH command returning).
    const timeEnv = options.time
        ? ` IVLLM_BENCH_TIME="${options.time}"`
        : '';
    const launch =
        `nohup${timeEnv} ${remoteEngine}/ivllm-bench.sh -c "${remoteDir}" ` +
        `> "${remoteDir}/orchestrator.log" 2>&1 < /dev/null & disown; echo started`;

    const { stdout, exitCode } = await this.ops.runRemote(launch, {
        env: this.envs,
        silent: false,
    });

    if (exitCode !== 0 || !stdout.includes('started')) {
        throw new Error(
            `benchmark submit failed for '${comparison}' (exit ${exitCode}): ${stdout}`,
        );
    }
}

async getBenchmarkStatus(comparison: string): Promise<BenchmarkStatus> {
    await this.bootstrap();
    const remoteDir = this.remoteComparisonDir(comparison);
    const { stdout, exitCode } = await this.ops.runRemote(
        `cat "${remoteDir}/benchmarking_status.json"`,
        { env: this.envs, silent: true },
    );
    if (exitCode !== 0) {
        throw new Error(
            `no status found for comparison '${comparison}' — check the name, or that submit succeeded`,
        );
    }
    return JSON.parse(stdout) as BenchmarkStatus;
}

async fetchBenchmarkResults(
    comparison: string,
    localDest: string,
): Promise<{ ready: true; path: string } | { ready: false; status: BenchmarkStatus }> {
    const status = await this.getBenchmarkStatus(comparison);
    if (!status.complete) {
        return { ready: false, status };
    }
    const remoteDir = `${this.remoteComparisonDir(comparison)}/results`;
    await this.ops.copyDirectory(localDest, remoteDir, 'down');
    return { ready: true, path: localDest };
}
```

Error-handling convention matches the rest of the file (throw with exit
code + stdout on non-zero `runRemote` results); `getBenchmarkStatus`'s "file
doesn't exist" case is deliberately a thrown `Error` (an actually-wrong
comparison name/never-submitted state), whereas "not complete yet" inside
`fetchBenchmarkResults` is deliberately *not* an error (see abstract method
doc above).

### 3.4 CLI wiring (`src/index.ts`)

Nested subcommand via Commander, matching the requested 3-verb shape:

```typescript
const bench = program
    .command('bench')
    .description('Benchmark and compare vLLM configurations');

bench
    .command('submit')
    .description('Submit a set of vLLM configs as a benchmark comparison')
    .argument('<comparison>', 'name for this comparison run')
    .argument('<configs...>', 'one or more vLLM YAML config files')
    .option('--time <duration>', 'SLURM time limit per job as <hh:mm:ss>')
    .action(cmdBenchSubmit);

bench
    .command('status')
    .description('Check progress of a benchmark comparison')
    .argument('<comparison>', 'comparison name, from `ivllm bench submit`')
    .action(cmdBenchStatus);

bench
    .command('results')
    .description('Fetch results of a completed benchmark comparison')
    .argument('<comparison>', 'comparison name')
    .argument('<outDir>', 'local directory to copy results into')
    .action(cmdBenchResults);
```

`cmdBenchStatus` always exits 0 and prints the counts/per-job table (mirrors
`cmdStatus`'s existing formatting approach — a new `formatBenchStatus()`
alongside the existing `formatJobTable`/`formatJobRow` in `utils.ts`).

`cmdBenchResults` is where the "soft error if in-progress" from the
original ask lives — **kept as a separate command from `status`, not
merged**, matching this codebase's existing precedent (`status` and
`diagnostics` are already separate top-level commands despite similar
overlap). Merging them would make one command do two different jobs
depending on hidden state; keeping them separate keeps "check progress" and
"fetch when done" each doing exactly one thing. The soft-error behaviour is
just `fetchBenchmarkResults`'s own discriminated result:

```typescript
async function cmdBenchResults(comparison: string, outDir: string): Promise<void> {
    const config = loadCredentials();
    assertConfigured(config);
    const backend = getBackend(config);
    const result = await backend.fetchBenchmarkResults(comparison, outDir);
    if (!result.ready) {
        console.log(`Comparison '${comparison}' is not finished yet:`);
        console.log(formatBenchStatus(result.status));
        process.exitCode = 1; // distinguishable from a real error for scripting
        return;
    }
    console.log(`✓ Results saved to: ${result.path}`);
}
```

## 4. Testing strategy

**TypeScript mock framework (`TestBackend`/`TestRemoteOps`,
`tests/integration/CLI.lifecycle.test.ts`)** — directly extensible, same
shallow-but-real level of coverage the existing `requestStart`/
`requestCancel` tests already get: `TestRemoteOps.runRemote()` pattern-matches
on the command string (`cmd.includes('ivllm-bench.sh')` /
`cmd.includes('benchmarking_status.json')`) and returns canned
responses/lockfile-like state. This can verify:
- `requestBenchmark` uploads the right number of configs to the right
  remote paths and constructs the detached-launch command correctly
  (including the `nohup`/`disown`/`IVLLM_BENCH_TIME` pieces).
- `getBenchmarkStatus` parses a canned `benchmarking_status.json` correctly,
  and throws when the mock reports non-zero exit (comparison not found).
- `fetchBenchmarkResults` returns `{ready: false, status}` vs. calling
  `copyDirectory` and returning `{ready: true}`, based on injected
  `complete` state.
- `cmdBenchResults`'s exit-code/soft-error behaviour.

What it **cannot** test: any of `ivllm-bench.sh`'s actual bash logic — the
mock never executes real bash, so the model-prefetch dedup, the
`write_status_summary()` JSON construction, the `srun --overlap` mechanics,
etc. are entirely out of its reach. It only proves the TypeScript layer
calls the right remote commands with the right arguments.

**Bash sandbox harness (`tests/bash/lib/sandbox.sh`, real bwrap +
mocked SLURM + real jq/yq)** — this is where the actual logic gets covered,
and most of it fits the harness's existing model well: `write_status_summary()`
and the model-dedup/prefetch loop are ordinary bash functions operating on
real `status.json`/`vllm.yaml` files that the existing
`create_status_pending`/`update_status_*` helpers can set up directly, same
as every other sandboxed test. One gap: the harness currently only binds
`src/engine`'s contents into the sandbox (see `sandbox_run`'s per-entry
`--ro-bind` loop) — since the script doesn't live there yet, a test needs an
explicit extra bind for `design/prototype/ivllm-bench.sh` until it's
promoted to `src/engine/` (§2), at which point it starts working through the
existing bind loop for free.

Not testable in the bash sandbox: the `srun --overlap --jobid=<id>`
cross-job-attach mechanism itself (SLURM behaviour, not something the
`srun` shim can meaningfully fake beyond "was it called with these
arguments"), and anything about real network reachability (already
partially validated this session via `design/prototype/nccl-probe.sh`-style
direct-on-Isambard smoke testing, which remains the right tool for that
class of question, not a unit test).

**Net**: the TS mock and bash sandbox cover two genuinely different, mostly
non-overlapping layers here (does the CLI call the right thing / does the
bash logic do the right thing) — both are usable now, with one small harness
extension needed for the bash side while the script is still a prototype.

## 5. Rollout checklist

- [ ] Extend `tests/bash/lib/sandbox.sh` to bind the prototype script (§4) —
      unblocks writing bash tests before the move in step below.
- [ ] Write failing bash tests for `write_status_summary()` and the
      model-dedup prefetch loop (AGENTS.md: tests before implementation —
      "implementation" here being the *promotion*, since the logic already
      exists in the prototype).
- [ ] Move `design/prototype/ivllm-bench.sh` → `src/engine/ivllm-bench.sh`;
      confirm the bash tests now pass unmodified (or with only the bind
      path updated).
- [ ] Add `BenchmarkStatus`/`BenchmarkJobStatus` to `src/types.ts`.
- [ ] Add the three abstract methods to `Backend.ts`.
- [ ] Implement them in `IsambardBareMetalBackend.ts` (§3.3).
- [ ] Extend `TestBackend`/`TestRemoteOps` and add TS tests (§4).
- [ ] Add `formatBenchStatus()` to `utils.ts`.
- [ ] Wire the `bench submit|status|results` subcommands into `index.ts`.
- [ ] Update `design/roadmap.md`'s benchmarking section to check off
      completed steps as this lands, per its existing step table.
- [ ] Bump patch version per AGENTS.md's pre-commit checklist once merged.
