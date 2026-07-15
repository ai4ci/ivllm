import type { Credentials, LockfileV3, OpsMode } from '../types.ts';
import { loadCredentials, assertConfigured } from '../config.ts';
import { makeV3Paths } from '../job.ts';
import { parseVllmConfig, stripMetadata } from '../vllm-config.ts';
import { makeRemoteOps } from '../remote-ops.ts';
import { readFileSync } from 'fs';
import crypto from 'crypto';

/**
 * Parsed CLI arguments for the `ivllm connect` command.
 */
interface ConnectArgs {
  jobName: string;
  configFile?: string;
  localPort: number;
  batch: boolean;
  dryRun: boolean;
  detach: boolean;
  noLaunch: boolean;
  timeLimit: string;
}

/**
 * Print help text for `ivllm connect`.
 */
function printHelp(): void {
  console.log(`
Usage: ivllm connect <job> [options]

Start or connect to a vLLM inference session.

If the job is already running, establishes an SSH tunnel.
If the job is stopped or failed, restarts it.
If the job doesn't exist, creates it and starts it.

Options:
  --config <file>       vLLM config YAML (required for first use)
  --local-port <n>      Local port for SSH tunnel (default: 11434)
  --batch               Submit to standard partition instead of interactive
  --detach              Exit after starting (don't stay in foreground)
  --no-launch           Skip assistant launcher menu
  --time <hh:mm:ss>     SLURM time limit (default: 8:00:00)
  --dry-run             Preview without connecting to HPC
  --help, -h            Show this help message

Examples:
  ivllm connect my-job --config examples/qwen2.5-instruct.yaml
  ivllm connect my-job --dry-run
  ivllm connect my-job --detach
`);
}

/**
 * Parse CLI arguments for `ivllm connect`.
 */
function parseConnectArgs(args: string[]): ConnectArgs {
  const boolFlags = new Set<string>();
  const flags: Record<string, string> = {};

  for (let i = 0; i < args.length; i++) {
    const arg = args[i];
    if (!arg || !arg.startsWith('--')) continue;
    const key = arg.slice(2);
    if (key === 'help' || key === 'h') {
      printHelp();
      process.exit(0);
    }
    const next = args[i + 1];
    if (!next || next.startsWith('--')) {
      boolFlags.add(key);
    } else {
      flags[key] = next;
      i++;
    }
  }

  const jobName = args[0] && !args[0].startsWith('--') ? args[0] : null;
  if (!jobName) {
    console.error('Error: Job name is required as the first argument');
    printHelp();
    process.exit(1);
  }

  return {
    jobName,
    configFile: flags['config'],
    localPort: flags['local-port']
      ? parseInt(flags['local-port'], 10)
      : 11434,
    batch: boolFlags.has('batch'),
    dryRun: boolFlags.has('dry-run'),
    detach: boolFlags.has('detach'),
    noLaunch: boolFlags.has('no-launch'),
    timeLimit: flags['time'] ?? '8:00:00',
  };
}

/**
 * Run the dry-run preview for a connect command.
 * Prints what would happen without connecting to the HPC.
 */
function dryRunPreview(
  jobName: string,
  args: ConnectArgs,
  config: Credentials,
): void {
  const v3paths = makeV3Paths(config.projectDir, jobName);

  console.log(`\n=== ivllm connect (dry-run) ===`);
  console.log(`Job      : ${jobName}`);
  console.log(`Port     : ${args.localPort}`);
  console.log(`Partition: ${args.batch ? 'standard' : 'interactive'}`);
  console.log(`Time     : ${args.timeLimit}`);
  console.log(`\nRemote paths:`);
  console.log(`  Lockfile : ${v3paths.statusFile}`);
  console.log(`  Config   : ${v3paths.vllmConfigFile}`);
  console.log(`  Script   : ${v3paths.scriptFile}`);
  console.log(`  Log      : ${v3paths.logFileGlob} (node-specific, one per node)`);

  if (args.configFile) {
    try {
      const cfg = parseVllmConfig(args.configFile);
      console.log(`\nModel config:`);
      console.log(`  Model      : ${cfg.model}`);
      console.log(`  Max len    : ${cfg.maxModelLen}`);
      console.log(`  TP:PP:DP   : ${cfg.tensorParallelSize}:${cfg.pipelineParallelSize}:${cfg.dataParallelSize}`);
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
  const MIN = 49152, MAX = 65535;
  return crypto.randomInt(0, MAX - MIN + 1) + MIN;
}

/**
 * The `ivllm connect` command handler.
 */
export async function cmdConnect(args: string[]): Promise<void> {
  // Handle help
  if (args.includes('--help') || args.includes('-h')) {
    printHelp();
    return;
  }

  let connectArgs: ConnectArgs;
  try {
    connectArgs = parseConnectArgs(args);
  } catch (e) {
    console.error('Error:', (e as Error).message);
    process.exit(1);
  }

  // Dry-run: preview without connecting (no credentials needed)
  if (connectArgs.dryRun) {
    // Use a placeholder config if credentials aren't configured
    const placeholderConfig: Credentials = {
      loginHost: '<login-host>',
      username: '<username>',
      projectDir: '/projects/<XXXX>',
      defaultLocalPort: 11434,
    };
    dryRunPreview(connectArgs.jobName, connectArgs, placeholderConfig);
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
  const configFile = connectArgs.configFile;
  if (!configFile) {
    console.error('Error: --config <file> is required for first use.');
    console.error('  ivllm connect my-job --config examples/qwen2.5-instruct.yaml');
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
  console.log(`\n=== ivllm connect: ${connectArgs.jobName} ===`);
  console.log(`Model    : ${parsedConfig.model}`);
  console.log(`Server   : ${connectArgs.batch ? 'standard partition' : 'interactive reservation'}`);

  console.log('\nChecking SSH connectivity...');
  await ops.checkSSH();
  console.log('  ✓ SSH connection OK');

  // Stage 2: Create job directory and lockfile
  const v3paths = makeV3Paths(config.projectDir, connectArgs.jobName);
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
    jobName: connectArgs.jobName,
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
    const existingResult = await ops.runRemote(`cat ${v3paths.statusFile} 2>/dev/null`, {
      env: [],
      silent: true,
    });
    if (existingResult.stdout) {
      console.error(`Error: Job '${connectArgs.jobName}' already exists.`);
      console.error(`  Use 'ivllm cancel ${connectArgs.jobName}' to stop it, or`);
      console.error(`  use 'ivllm cancel ${connectArgs.jobName} --force' to force-cancel.`);
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

  console.log(`\nJob '${connectArgs.jobName}' is set up and ready.`);
  console.log(`Lockfile : ${v3paths.statusFile}`);
  console.log(`Config   : ${v3paths.vllmConfigFile}`);
  console.log(`Endpoint : http://localhost:${connectArgs.localPort}/v1 (after start)`);
  console.log(`\nNext: SLURM script generation and job submission.\n`);
}
