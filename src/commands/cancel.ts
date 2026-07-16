import { loadCredentials, assertConfigured } from '../config.ts';
import { makeJobV3Paths, makeV3Paths } from '../job.ts';

/**
 * The `ivllm cancel` command handler.
 */
export async function cmdCancel(
  jobName: string,
  force: boolean,
  dryRun: boolean,
): Promise<void> {
  // Handle help

  // Load and validate credentials
  const config = loadCredentials();
  try {
    assertConfigured(config);
  } catch (e) {
    console.error('Error:', (e as Error).message);
    process.exit(1);
  }

  const v3paths = makeJobV3Paths(config.projectDir, jobName);

  if (dryRun) {
    console.log(`\n=== ivllm cancel (dry-run) ===`);
    console.log(`Job       : ${jobName}`);
    console.log(`Lockfile  : ${v3paths.statusFile}`);
    console.log(
      `Method    : ${force ? 'force (scancel)' : 'graceful (write cancel)'}`,
    );
    console.log(`\nSteps:`);
    if (force) {
      console.log(`  1. SSH: scancel --name ${jobName}`);
      console.log(`  2. SSH: rm -f ${v3paths.statusFile}`);
    } else {
      console.log(`  1. SSH: write 'cancel' to ${v3paths.statusFile}`);
      console.log(
        `  2. SSH: tail log files (${v3paths.logFileGlob}) until status=stopped`,
      );
      console.log(`  3. SSH: verify cleanup`);
    }
    console.log();
    return;
  }

  // ── Placeholder for real implementation ──
  const method = force ? 'force' : 'graceful';
  console.log(`\nNot yet implemented — use --dry-run to preview.`);
  console.log(`Would cancel job '${jobName}' via ${method} shutdown.\n`);
}
