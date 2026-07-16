import type { Credentials, LockfileV3, OpsMode } from '../types.ts';
import { loadCredentials, assertConfigured } from '../config.ts';
import { makeJobV3Paths } from '../job.ts';
import { parseVllmConfig } from '../vllm-config.ts';
import { makeRemoteOps } from '../remote-ops.ts';
import crypto from 'crypto';

/**
 * Run the dry-run preview for a connect command.
 * Prints what would happen without connecting to the HPC.
 */
function dryRunPreview(
  jobName: string,
  configFile: string | undefined,
  localPort: number,
  batch: boolean,
  timeLimit: string,
  config: Credentials,
): void {
  const v3paths = makeJobV3Paths(config.projectDir, jobName);

  console.log(`\n=== ivllm connect (dry-run) ===`);
  console.log(`Job      : ${jobName}`);
  console.log(`Port     : ${localPort}`);
  console.log(`Partition: ${batch ? 'standard' : 'interactive'}`);
  console.log(`Time     : ${timeLimit}`);
  console.log(`\nRemote paths:`);
  console.log(`  Lockfile : ${v3paths.statusFile}`);
  console.log(`  Config   : ${v3paths.vllmConfigFile}`);
  console.log(`  Script   : ${v3paths.scriptFile}`);
  console.log(
    `  Log      : ${v3paths.logFileGlob} (node-specific, one per node)`,
  );

  if (configFile) {
    try {
      const cfg = parseVllmConfig(configFile);
      console.log(`\nModel config:`);
      console.log(`  Model      : ${cfg.model}`);
      console.log(`  Max len    : ${cfg.maxModelLen}`);
      console.log(
        `  TP:PP:DP   : ${cfg.tensorParallelSize}:${cfg.pipelineParallelSize}:${cfg.dataParallelSize}`,
      );
      console.log(`  Idle tmout : ${cfg.idleTimeout}`);
      console.log(`  Min vLLM   : ${cfg.minVllmVersion}`);
    } catch (e) {
      console.log(`\n  (config parse error: ${(e as Error).message})`);
    }
  }

  console.log(`\nGenerated SLURM script would be written to:`);
  console.log(`  ${v3paths.scriptFile}`);
}

/** Generate a random high port for the vLLM server. */
function generateRandomHighPort(): number {
  const MIN = 49152,
    MAX = 65535;
  return crypto.randomInt(0, MAX - MIN + 1) + MIN;
}

/**
 * The `ivllm connect` command handler.
 */
export async function cmdConnect(
  jobName: string,
  configFile: string | undefined,
  port: string,
  batch: boolean,
  dryRun: boolean,
  timeLimit: string,
): Promise<void> {
  // Dry-run: preview without connecting (no credentials needed)

  const localPort = parseInt(port, 10);

  if (dryRun) {
    // Use a placeholder config if credentials aren't configured
    const placeholderConfig: Credentials = {
      loginHost: '<login-host>',
      username: '<username>',
      projectDir: '/projects/<XXXX>',
      defaultLocalPort: 11434,
    };
    dryRunPreview(
      jobName,
      configFile,
      localPort,
      batch,
      timeLimit,
      placeholderConfig,
    );
    return;
  }

  // Load and validate credentials for real operations
  const config = loadCredentials();
  try {
    assertConfigured(config);
  } catch (e) {
    console.error('Error:', (e as Error).message);
    process.exit(1);
  }

  // Resolve config file — required for first use, cached for subsequent
  if (!configFile) {
    console.error('Error: --config <file> is required for first use.');
    console.error(
      '  ivllm connect my-job --config examples/qwen2.5-instruct.yaml',
    );
    process.exit(1);
  }

  // Parse config
  let parsedConfig;
  try {
    parsedConfig = parseVllmConfig(configFile);
  } catch (e) {
    console.error('Error parsing config:', (e as Error).message);
    process.exit(1);
  }

  // Create remote ops
  const mode: OpsMode = 'real';
  const ops = makeRemoteOps(config, mode);

  // Stage 1: SSH pre-flight
  console.log(`\n=== ivllm connect: ${jobName} ===`);
  console.log(`Model    : ${parsedConfig.model}`);
  console.log(
    `Server   : ${batch ? 'standard partition' : 'interactive reservation'}`,
  );

  console.log('\nChecking SSH connectivity...');
  await ops.checkSSH();
  console.log('  ✓ SSH connection OK');

  // Stage 2: Create job directory and lockfile
  const v3paths = makeJobV3Paths(config.projectDir, jobName);
  const serverPort = generateRandomHighPort();

  console.log(`\nCreating job directory...`);
  const mkdirResult = await ops.runRemote(`mkdir -p ${v3paths.jobDir}`);
  if (mkdirResult.exitCode !== 0) {
    console.error('Error: could not create job directory');
    process.exit(1);
  }
  console.log(`  ✓ ${v3paths.jobDir}`);

  console.log('Creating lockfile...');
  const lockfile: LockfileV3 = {
    status: 'pending',
    jobName: jobName,
    model: parsedConfig.model,
    serverPort,
    requestedTime: new Date().toISOString(),
    idleTimeout: parsedConfig.idleTimeout,
  };

  // Use set -C (noclobber) for atomic lockfile creation
  const lockfileJson = JSON.stringify(lockfile);
  const createResult = await ops.runRemote(
    `set -C; cat > ${v3paths.statusFile} << 'LOCKFILE_EOF'\n${lockfileJson}\nLOCKFILE_EOF`,
    { env: [], silent: true },
  );

  if (createResult.exitCode !== 0) {
    // Check if lockfile already exists
    const existingResult = await ops.runRemote(
      `cat ${v3paths.statusFile} 2>/dev/null`,
      {
        env: [],
        silent: true,
      },
    );
    if (existingResult.stdout) {
      console.error(`Error: Job '${jobName}' already exists.`);
      console.error(`  Use 'ivllm cancel ${jobName}' to stop it, or`);
      console.error(`  use 'ivllm cancel ${jobName} --force' to force-cancel.`);
    } else {
      console.error('Error: could not create lockfile');
    }
    process.exit(1);
  }
  console.log(`  ✓ ${v3paths.statusFile} (port ${serverPort})`);

  // Stage 3: Upload config
  console.log('Uploading config...');
  await ops.copyFile(configFile, v3paths.vllmConfigFile);
  console.log(`  ✓ ${v3paths.vllmConfigFile}`);

  console.log(`\nJob '${jobName}' is set up and ready.`);
  console.log(`Lockfile : ${v3paths.statusFile}`);
  console.log(`Config   : ${v3paths.vllmConfigFile}`);
  console.log(`Endpoint : http://localhost:${localPort}/v1 (after start)`);
  console.log(`\nNext: SLURM script generation and job submission.\n`);
}
