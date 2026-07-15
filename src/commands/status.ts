import { loadCredentials, assertConfigured } from '../config.ts';

import { parseJobDetails } from '../job.ts';
import type { JobDetails } from "../types.ts";
import { makeRemoteOps } from "../remote-ops.ts";

export interface StatusArgs {
  jobName?: string;
}

/**
 *
 * @param args
 */
export function parseStatusArgs(args: string[]): StatusArgs {
  const jobName = args[0] && !args[0].startsWith('--') ? args[0] : undefined;
  return { jobName };
}

/**
 *
 * @param job
 */
export function formatJobRow(job: JobDetails): string {
  const parts: string[] = [
    job.job_name.padEnd(20),
    job.status.padEnd(14),
    (job.slurm_job_id ?? '-').padEnd(10),
    (job.model ?? '-').padEnd(36),
  ];
  if (job.error) parts.push(`ERROR: ${job.error}`);
  return parts.join('  ').trimEnd();
}

/**
 *
 * @param jobs
 */
export function formatJobTable(jobs: JobDetails[]): string {
  if (jobs.length === 0) return 'No active ivllm jobs found.';
  const header =
    'JOB NAME'.padEnd(20) +
    '  ' +
    'STATUS'.padEnd(14) +
    '  ' +
    'SLURM ID'.padEnd(10) +
    '  ' +
    'MODEL';
  const separator = '-'.repeat(header.length);
  const rows = jobs.map(formatJobRow);
  return [header, separator, ...rows].join('\n');
}

/**
 *
 * @param args
 */
export async function cmdStatus(args: string[]): Promise<void> {
  // Handle help flag
  if (args.includes('--help') || args.includes('-h')) {
    console.log(`
Usage: ivllm status [job]

Options:
  [job]                 Show status of specific job (if omitted, show all jobs)
  --help, -h            Show this help message

Examples:
  ivllm status
  ivllm status my-job
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

  const ops = makeRemoteOps(config, 'real');

  const { jobName } = parseStatusArgs(args);

  if (jobName) {
    // Single job
    const { exitCode, stdout } = await ops.runRemote(
      `cat ~/${jobName}/job_details.json 2>/dev/null`
    );
    if (exitCode !== 0 || !stdout.trim()) {
      console.error(
        `No job '${jobName}' found. (No job_details.json at ~/${jobName}/)`,
      );
      process.exit(1);
    }
    const details = parseJobDetails(stdout);
    if (!details) {
      console.error(`Could not parse job_details.json for '${jobName}'.`);
      process.exit(1);
    }
    console.log(formatJobTable([details]));
  } else {
    // All jobs — use jq to combine all job_details.json into a JSON array
    const { stdout } = await ops.runRemote(
      `shopt -s nullglob; files=(~/*/job_details.json); if [ \${#files[@]} -eq 0 ]; then echo '[]'; else jq -s '.' "\${files[@]}"; fi`,
    );
    let jobs: JobDetails[] = [];
    try {
      const parsed = JSON.parse(stdout || '[]');
      if (Array.isArray(parsed)) {
        jobs = parsed
          .map((j) => parseJobDetails(JSON.stringify(j)))
          .filter((j): j is JobDetails => j !== null);
      }
    } catch {
      // ignore parse errors — show empty list
    }
    console.log(formatJobTable(jobs));
  }
}
