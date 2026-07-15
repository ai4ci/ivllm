import type { Credentials } from '../types.ts';
import { loadCredentials, assertConfigured } from '../config.ts';
import { makeV3Paths } from '../job.ts';
import { parseVllmConfig } from '../vllm-config.ts';

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
  console.log(`  Lockfile : ${v3paths.lockfilePath}`);
  console.log(`  Config   : ${v3paths.configPath}`);
  console.log(`  Script   : ${v3paths.scriptPath}`);
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
  console.log(`  ${v3paths.scriptPath}`);
  console.log(`\nTo run for real, omit --dry-run.\n`);
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

  // Check for config file (required for first use)
  if (!connectArgs.configFile) {
    // Try to find a saved config
    const savedPath = `${config.projectDir}/engine/jobs/${connectArgs.jobName}/vllm.yaml`;
    if (!existsSync) {
      // Will check via SSH at runtime
    }
  }

  // ── Placeholder for real implementation ──
  // Steps to implement when SSH integration is added:
  // 1. SSH pre-flight: check connectivity, venv exists
  // 2. Create lockfile on HPC (status.json with "pending")
  // 3. Upload config and generated SLURM script
  // 4. Download model if not cached
  // 5. Submit sbatch job
  // 6. Monitor lockfile for status transitions
  // 7. When running: establish SSH tunnel
  // 8. Print endpoint URL
  // 9. Optional: launch assistant

  console.log(`\nNot yet implemented — use --dry-run to preview, or`);
  console.log(`use 'ivllm start' (v2) for actual job submission.\n`);
}
