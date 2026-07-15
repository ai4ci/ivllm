import { loadCredentials, assertConfigured } from '../config.ts';
import { SessionState } from '../types.ts';
import { shutdown } from '../session-helper.ts';

import { makePaths, parseJobDetails } from '../job.ts';
import { makeRemoteOps } from '../remote-ops.ts';

export interface StopArgs {
  jobName: string;
}

/**
 *
 * @param args
 */
export function parseStopArgs(args: string[]): StopArgs {
  const jobName = args[0] && !args[0].startsWith('--') ? args[0] : null;
  if (!jobName) throw new Error('Job name is required as the first argument');
  return { jobName };
}

/**
 *
 * @param args
 */
export async function cmdStop(args: string[]): Promise<void> {
  // Handle help flag
  if (args.includes('--help') || args.includes('-h')) {
    console.log(`
Usage: ivllm stop <job>

Options:
  <job>                 Name of the job to stop and clean up
  --help, -h            Show this help message

Examples:
  ivllm stop my-job
`);
    return;
  }

  const config = loadCredentials();
  try {
    assertConfigured(config);
  } catch (e) {
    console.error('Error:', (e as Error).message);
    process.exit(1);
  }

  const ops = makeRemoteOps(config, false);

  let stopArgs: StopArgs;
  try {
    stopArgs = parseStopArgs(args);
  } catch (e) {
    console.error('Error:', (e as Error).message);
    console.error('Usage: ivllm stop <job>');
    process.exit(1);
  }

  // construct a process state from the jobName:
  const state = new SessionState({
    sessionName: stopArgs.jobName,
    ops: ops,
    paths: makePaths(
      config,
      stopArgs.jobName,
      'unknown', //model
      'unknown', //cache
      'unknown', //vllmVersion
    ),
  });

  // Read job_details.json to get SLURM job ID
  // This isn't really needed as if missing the job will be killed by name

  const { stdout } = await ops.runRemote(
    `cat ${state.paths.remoteJobLockFile} 2>/dev/null`,
  );
  const details = parseJobDetails(stdout);
  if (details?.slurm_job_id) state.slurmJobId = details?.slurm_job_id;

  console.log(`=== ivllm stop: ${state.sessionName} ===`);

  shutdown(state, 'User requested shutdown', 0);

  console.log(`✓ Job '${stopArgs.jobName}' stopped and cleaned up.`);
}
