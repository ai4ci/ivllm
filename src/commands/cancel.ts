import { loadCredentials, assertConfigured } from '../config.ts';
import { makeV3Paths } from '../job.ts';

/**
 * Parsed CLI arguments for the `ivllm cancel` command.
 */
interface CancelArgs {
  jobName: string;
  force: boolean;
  dryRun: boolean;
}

/**
 * Print help text for `ivllm cancel`.
 */
function printHelp(): void {
  console.log(`
Usage: ivllm cancel <job> [options]

Cancel a running vLLM inference job.

Graceful cancel (default):
  Writes "cancel" to the job's lockfile. The compute-side monitor detects the
  request and shuts down vLLM cleanly, preserving logs and diagnostics.

Force cancel (--force):
  Runs scancel on the SLURM job directly and updates the lockfile. Use this
  when graceful shutdown fails or the monitor is unresponsive.

Options:
  --force               Use scancel directly instead of graceful cancel
  --dry-run             Preview what would happen without executing
  --help, -h            Show this help message

Examples:
  ivllm cancel my-job              # Graceful shutdown
  ivllm cancel my-job --force      # Force kill via scancel
  ivllm cancel my-job --dry-run    # Preview only
`);
}

/**
 * Parse CLI arguments for `ivllm cancel`.
 */
function parseCancelArgs(args: string[]): CancelArgs {
  const boolFlags = new Set<string>();

  for (let i = 0; i < args.length; i++) {
    const arg = args[i];
    if (!arg || !arg.startsWith('--')) continue;
    const key = arg.slice(2);
    if (key === 'help' || key === 'h') {
      printHelp();
      process.exit(0);
    }
    boolFlags.add(key);
  }

  const jobName = args[0] && !args[0].startsWith('--') ? args[0] : null;
  if (!jobName) {
    console.error('Error: Job name is required as the first argument');
    printHelp();
    process.exit(1);
  }

  return {
    jobName,
    force: boolFlags.has('force'),
    dryRun: boolFlags.has('dry-run'),
  };
}

/**
 * The `ivllm cancel` command handler.
 */
export async function cmdCancel(args: string[]): Promise<void> {
  // Handle help
  if (args.includes('--help') || args.includes('-h')) {
    printHelp();
    return;
  }

  // Load and validate credentials
  const config = loadCredentials();
  try {
    assertConfigured(config);
  } catch (e) {
    console.error('Error:', (e as Error).message);
    process.exit(1);
  }

  let cancelArgs: CancelArgs;
  try {
    cancelArgs = parseCancelArgs(args);
  } catch (e) {
    console.error('Error:', (e as Error).message);
    process.exit(1);
  }

  const v3paths = makeV3Paths(config.projectDir, cancelArgs.jobName);

  if (cancelArgs.dryRun) {
    console.log(`\n=== ivllm cancel (dry-run) ===`);
    console.log(`Job       : ${cancelArgs.jobName}`);
    console.log(`Lockfile  : ${v3paths.lockfilePath}`);
    console.log(`Method    : ${cancelArgs.force ? 'force (scancel)' : 'graceful (write cancel)'}`);
    console.log(`\nSteps:`);
    if (cancelArgs.force) {
      console.log(`  1. SSH: scancel --name ${cancelArgs.jobName}`);
      console.log(`  2. SSH: rm -f ${v3paths.lockfilePath}`);
    } else {
      console.log(`  1. SSH: write 'cancel' to ${v3paths.lockfilePath}`);
      console.log(`  2. SSH: tail -f ${v3paths.logPath} until status=stopped`);
      console.log(`  3. SSH: verify cleanup`);
    }
    console.log();
    return;
  }

  // ── Placeholder for real implementation ──
  const method = cancelArgs.force ? 'force' : 'graceful';
  console.log(`\nNot yet implemented — use --dry-run to preview.`);
  console.log(`Would cancel job '${cancelArgs.jobName}' via ${method} shutdown.\n`);
}
