import type { RemoteOps, EnvVarEntry, LocalOps } from './types.ts';
import { SessionState, ProcessState } from './types.ts';
import { makePaths } from './job.ts';
import type { InferenceJobOptions } from './types';
import { monitorSession, detachSession } from './monitors.ts';

import { writeStrippedConfig } from './vllm-config.ts';
import { submitJob, runInteractive } from './slurm.ts';
import { renderInferenceScript } from './templates/inference.ts';
import { renderMockInferenceScript } from './templates/mock-inference.ts';
import { writeFileSync, existsSync, unlinkSync } from 'fs';
import { tmpdir } from 'os';
import { join, basename } from 'path';

// ── Shared utilities ──────────────────────────────────────────────────────

/**
 *
 * @param startArgs
 * @param ops
 * @param config
 * @param localOps
 */
export async function preFlight(
  startArgs: InferenceJobOptions,
  localOps: LocalOps,
): Promise<void> {
  const localPort = startArgs.localPort;
  if (!startArgs.dryRun) {
    console.log(`Checking local port ${localPort}...`);
    const portInUse = await localOps.isLocalPortInUse(localPort);
    if (portInUse) {
      console.error(
        `Error: Local port ${localPort} is already in use by ${portInUse.process} (pid ${portInUse.pid}).`,
      );
      process.exit(1);
    }
    console.log(`✓ Port ${localPort} is free`);
  }
}

/**
 *
 * @param ops
 * @param config
 * @param startArgs
 * @param hfHome
 * @param model
 * @param vllmVersion
 * @param ss
 */
export async function ensureModelDownloaded(ss: SessionState): Promise<void> {
  const hfHome = ss.paths.remoteProjectHfDir;

  debugger;

  const model = ss.startArgs.configYaml.model;
  const cachePath = ss.paths.remoteProjectHfModelDir;
  const hfToken = ss.startArgs.credentials.hfToken ?? process.env['HF_TOKEN'];
  const hfEnv: EnvVarEntry[] = hfToken
    ? [
        { key: 'HF_HOME', value: hfHome },
        { key: 'HF_TOKEN', value: hfToken },
      ]
    : [{ key: 'HF_HOME', value: hfHome }];

  const ops = ss.ops;

  if (ss.startArgs.mock) {
    console.log(`[mock] Model download skipped`);
  } else {
    const { exitCode: cacheCheck } = await ops.runRemote(
      `test -d ${cachePath}`,
      { env: [], silent: true },
    );
    if (cacheCheck === 0 && !ss.startArgs.dryRun) {
      console.log(`✓ Model cached at ${cachePath}`);
    } else {
      console.log(`Downloading model ${model} to ${hfHome} on compute node...`);
      await ops.runRemote(
        `mkdir -p ${hfHome} && chmod g+w ${hfHome} 2>/dev/null || true`,
        { env: [], silent: true },
      );

      console.log(`activate venv: ${ss.paths.remoteProjectVllmVenvActivate}`);
      await ops.runRemote(
        `umask 0002 && source ${ss.paths.remoteProjectVllmVenvActivate}`,
      );

      // srun on the interactve partition to download model. Seems to be blocked on login nodes.
      const remoteSrun = `srun \\
--job-name="${ss.startArgs.jobName}_download" \\
--nodes=1 \\
--ntasks-per-node=1 \\
--cpus-per-task=4 \\
--gpus-per-node=0 \\
--partition=interactive \\
--reservation=interactive \\
--interactive \\
--mem=16G \\
--time=00:30:00 \\
--export=ALL \\
/bin/bash -c "umask 0002 && source ${ss.paths.remoteProjectVllmVenvActivate} && hf download '${model}'"
`;
      const { exitCode: dlCode } = await ops.runRemote(remoteSrun, {
        env: hfEnv,
        silent: false,
      });

      if (dlCode !== 0) {
        console.error('Error: Model download failed.');
        process.exit(1);
      }
      console.log('✓ Model downloaded');
    }
  }
}

/**
 *
 * @param ops
 * @param jobName
 * @param remoteWorkDir
 * @param remoteJobDetails
 * @param dryRun
 * @param sessionState
 */
export async function createJobLockfile(
  sessionState: SessionState,
): Promise<void> {
  const ops = sessionState.ops;
  const jobName = sessionState.startArgs.jobName;
  const remoteWorkDir = sessionState.paths.remoteJobDir;
  const remoteLockFile = sessionState.paths.remoteJobLockFile;

  console.log('Creating job working directory and lockfile...');
  await ops.runRemote(`mkdir -p ${remoteWorkDir}`);
  const pendingJson = JSON.stringify({
    status: 'pending',
    job_name: jobName,
  });
  const { exitCode: lockCode } = await ops.runRemote(
    `set -C; echo '${pendingJson}' > ${remoteLockFile}`,
  );
  if (lockCode !== 0) {
    console.error(
      `Error: Job '${jobName}' already exists (lockfile present). Use 'ivllm stop ${jobName}' to clean up.`,
    );
    process.exit(1);
  }
  console.log('✓ Lockfile created');
}

// ── Shutdown & diagnostics ────────────────────────────────────────────────

/**
 *
 * @param state
 */
export async function printSlurmLog(state: SessionState): Promise<void> {
  console.error('\n--- SLURM log ---');
  const { stdout } = await state.ops.runRemote(
    `tail -50 ${state.paths.remoteJobLogFile} 2>/dev/null`,
  );
  if (stdout) console.error(stdout);
  console.error('--- end log ---\n');
}

// export async function printCrashDiagnostics(
//   state: SessionState,
// ): Promise<void> {
//   if (state.crashDiagnosticsPrinted || !state.slurmJobId) return;
//   state.crashDiagnosticsPrinted = true;
//
//   console.error('\n--- SLURM accounting ---');
//   let stdout = '';
//   for (let attempt = 0; attempt < 5; attempt++) {
//     const result = await state.ops.runRemote(
//       `${buildSacctDiagnosticsCommand(state.slurmJobId)} 2>/dev/null || true`
//     );
//     stdout = result.stdout;
//     if (stdout.trim() && sacctDiagnosticsSettled(stdout, state.slurmJobId))
//       break;
//     if (attempt < 4) await sleep(2000);
//   }
//   if (stdout.trim()) {
//     console.error(stdout);
//     const localAccountingTmp = join(
//       tmpdir(),
//       `ivllm-${state.jobName}-slurm-accounting.txt`,
//     );
//     writeFileSync(
//       localAccountingTmp,
//       stdout.endsWith('\n') ? stdout : `${stdout}\n`,
//       'utf-8',
//     );
//     try {
//       await state.ops.copyFile(
//         localAccountingTmp,
//         `${state.remoteWorkDir}/${state.jobName}.slurm-accounting.txt`,
//       );
//     } catch (error) {
//       console.error(
//         `(failed to save SLURM accounting snapshot: ${(error as Error).message})`,
//       );
//     } finally {
//       if (existsSync(localAccountingTmp)) unlinkSync(localAccountingTmp);
//     }
//   } else {
//     console.error('(no sacct output available)');
//   }
//   console.error('--- end accounting ---\n');
//   console.error(`Remote work dir : ~/${state.jobName}`);
//   console.error(
//     `Remote script   : ~/${state.jobName}/${state.jobName}.slurm.sh`,
//   );
//   console.error(
//     `Remote log      : ~/${state.jobName}/${state.jobName}.slurm.log`,
//   );
//   console.error(`Remote ray logs : ~/${state.jobName}/ray-logs/`);
//   console.error(`Remote sacct    : ~/${state.jobName}/slurm-accounting.txt`);
//   console.error(`Remote details  : ~/${state.jobName}/job_details.json\n`);
// }

export async function shutdown(
  state: ProcessState,
  reason: string,
  exitCode?: number,
): Promise<void>;

export async function shutdown(
  state: SessionState,
  reason: string,
  exitCode?: number,
): Promise<void>;

/**
 *
 * @param processState
 * @param state
 * @param reason
 * @param exitCode
 */
export async function shutdown(
  state: ProcessState | SessionState,
  reason: string,
  exitCode = 0,
): Promise<void> {
  if (state.shuttingDown) return;
  state.shuttingDown = true;
  console.log(`\n⏹  Shutting down: ${reason}`);

  // If target is hold a live srun connection via ssh we need to kill it.
  // this will cancel the job.
  if (state.process) {
    process.stdout.write('  Closing interactive ssh connection...');
    state.process.kill();
    console.log(' done');
  }

  if (state.heartbeatTimer) clearInterval(state.heartbeatTimer);

  let cancelString: string;
  if (state.slurmJobId) {
    cancelString = state.slurmJobId;
  } else {
    cancelString = `--name ${state.sessionName}`;
  }
  process.stdout.write(`  Cancelling SLURM job (${cancelString})...`);
  await state.ops
    .runRemote(`scancel ${cancelString}}`, {
      env: [],
      silent: true,
    })
    .catch(() => {});
  console.log(' done');

  if (state.tunnel) {
    process.stdout.write('  Closing SSH tunnel...');
    state.tunnel.kill();
    console.log(' done');
  }

  if (state instanceof SessionState) {
    process.stdout.write('  Removing lockfile...');
    await state.ops
      .runRemote(`rm -f "${state.paths.remoteJobLockFile}"`)
      .catch(() => {});
    console.log(' done');
  }

  console.log('✓ Session shut down');

  // Best-effort: terminate any local orphaned tunnel for the default port
  // const localPort = state.config.defaultLocalPort ?? 11434;
  // await cleanupLocalTunnel(localPort);

  process.exit(exitCode);
}

// ── Shutdown helper ─────────────────────────────────────────────────────

/**
 * Attempt to terminate any local SSH forward-tunnel processes for the given port.
 * @param localPort
 */
// function cleanupLocalTunnel(localPort: number): Promise<void> {
//   return new Promise((resolve) => {
//     const pattern = `ssh.*-L.*${localPort}:`;
//     const proc = spawn('pkill', ['-f', pattern], { stdio: 'ignore' });
//     proc.on('close', (code) => {
//       if (code === 0)
//         console.log(`  Terminated local SSH tunnel on port ${localPort}`);
//       // exit code 1 means no process matched — that's fine
//       resolve();
//     });
//     proc.on('error', () => resolve()); // pkill not available — ignore
//   });
// }

/**
 *
 */
export function timestamp(): string {
  return new Date().toTimeString().slice(0, 8);
}

/**
 *
 * @param ms
 */
export function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// ── Unified session pipeline ──────────────────────────────────────────────

/**
 *
 * @param config
 * @param startArgs
 * @param isInteractive
 * @param ops
 * @param localOps
 */
export async function runInferenceSession(
  startArgs: InferenceJobOptions,
  ops: RemoteOps,
  localOps: LocalOps,
): Promise<void> {
  const config = startArgs.credentials;
  const { jobName, serverPort } = startArgs;
  const localPort = startArgs.localPort;

  const isInteractive = startArgs.isInteractive;
  const { model } = startArgs.configYaml;

  // ── 1. Pre-flight ──────────────────────────────────────────────────────
  ops.checkSSH();
  await preFlight(startArgs, localOps);

  const yaml = startArgs.configYaml;

  const vllmVersion = await ops.matchVllmVersion(yaml.minVllmVersion);

  const paths = makePaths(
    config,
    startArgs.jobName,
    yaml.model,
    startArgs.cacheKey,
    vllmVersion,
  );

  // ── 3. Build session state ────────────────────────────────────────────
  const sessionState = new SessionState({
    sessionName: jobName,
    ops,
    localOps,
    paths: paths,
    startArgs,
    vllmVersion,
  });

  // ── 4. Signal handlers ────────────────────────────────────────────────
  process.on('SIGINT', () => shutdown(sessionState, 'interrupted (Ctrl+C)'));
  process.on('SIGTERM', () => shutdown(sessionState, 'terminated'));

  // ── 5. Startup info ───────────────────────────────────────────────────

  const { configFile, gpuCount } = startArgs;
  const nodeCount = Math.ceil(gpuCount / 4);

  console.log(
    `=== ivllm ${isInteractive ? 'interactive' : 'start'} (v${__VERSION__}) ===`,
  );
  console.log(`Job        : ${jobName}`);
  console.log(`Model      : ${model}`);
  console.log(`Cache      : ${startArgs.cacheKey}`);
  console.log(`Config     : ${paths.localCacheVllmConfigFile}`);
  console.log(
    `GPUs       : ${gpuCount}${nodeCount > 1 ? ` (${nodeCount} nodes × ${gpuCount / nodeCount} GPUs each)` : ''}`,
  );
  if (nodeCount > 1) {
    console.log(`⚠ Multi-node job: ${nodeCount} nodes requested`);
  }
  console.log(`Local port : ${localPort}  |  Server port: ${serverPort}`);
  console.log('');

  // ── 6. Venv check ─────────────────────────────────────────────────────
  if (!startArgs.dryRun) {
    console.log('Checking venv...');
    const venvDir = paths.remoteProjectVllmDir;
    const { exitCode: venvCheck } = await ops.runRemote(
      `test -f ${paths.remoteProjectVllmVenvActivate}`,
    );
    if (venvCheck !== 0) {
      console.error(
        `Error: vLLM venv not found at ${venvDir}. Run 'ivllm setup' first.`,
      );
      await shutdown(sessionState, 'venv not found', 1);
    }
    console.log('✓ Venv check passed');
  }

  // ── 7. Model download ─────────────────────────────────────────────────
  await ensureModelDownloaded(sessionState);

  // ── 8. Lockfile ───────────────────────────────────────────────────────
  await createJobLockfile(sessionState);

  // ── 9. Generate script ────────────────────────────────────────────────
  const remoteConfigFile = paths.remoteJobVllmConfigFile;
  const remoteScriptPath = paths.remoteJobScriptFile;

  const script = startArgs.mock
    ? renderMockInferenceScript(sessionState)
    : renderInferenceScript(sessionState);

  const localScriptTmp = join(tmpdir(), `ivllm-${jobName}.slurm.sh`);
  writeFileSync(localScriptTmp, script, 'utf-8');

  // ── 10. Copy & submit ─────────────────────────────────────────────────

  try {
    console.log('Copying files to login node...');
    if (remoteConfigFile && configFile) {
      const strippedConfigTmp = writeStrippedConfig(configFile);
      try {
        await ops.copyFile(strippedConfigTmp, remoteConfigFile);
      } finally {
        unlinkSync(strippedConfigTmp);
      }
    }
    await ops.copyFile(localScriptTmp, remoteScriptPath);
    console.log('✓ Files copied');

    if (startArgs.dryRun) {
      console.log(`\n=== Dry-run complete ===`);
      console.log(`Generated files saved to: ${basename(remoteScriptPath)}`);
      if (remoteConfigFile && configFile) {
        console.log(`  ${configFile}  →  ${remoteConfigFile}`);
      }
      console.log(`  ${jobName}.slurm.sh  →  ${remoteScriptPath}`);
      console.log(`\nTo run for real, omit --dry-run.`);
      return;
    }

    if (isInteractive || startArgs.mock) {
      console.log('Running interactive job...');
      await runInteractive(ops, remoteScriptPath, sessionState, monitorSession);
    } else {
      console.log('Submitting SLURM job...');
      await submitJob(
        remoteScriptPath,
        sessionState,
        sessionState.startArgs.preCache ? detachSession : monitorSession,
      );
    }
  } finally {
    if (existsSync(localScriptTmp)) unlinkSync(localScriptTmp);
  }
}
