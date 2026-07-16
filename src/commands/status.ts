import { loadCredentials, assertConfigured } from '../config.ts';

import { makeJobV3Paths, parseV3Lockfile } from '../job.ts';
import type { LockfileV3 } from '../types.ts';
import { makeRemoteOps } from '../remote-ops.ts';
import { makeV3Paths } from '../job.ts';

/**
 *
 * @param job
 */
export function formatJobRow(job: LockfileV3): string {
  const parts: string[] = [
    job.jobName.padEnd(20),
    job.status.padEnd(14),
    (job.slurmJobId ?? '-').padEnd(10),
    (job.model ?? '-').padEnd(36),
  ];
  if (job.reason) parts.push(job.reason!);
  return parts.join('  ').trimEnd();
}

/**
 *
 * @param jobs
 */
export function formatJobTable(jobs: LockfileV3[]): string {
  if (jobs.length === 0) return 'No active ivllm jobs found.';
  const header =
    'JOB NAME'.padEnd(20) +
    '  ' +
    'STATUS'.padEnd(14) +
    '  ' +
    'SLURM ID'.padEnd(10) +
    '  ' +
    'MODEL'.padEnd(36) +
    '  ' +
    'REASON';
  const separator = '-'.repeat(header.length);
  const rows = jobs.map(formatJobRow);
  return [header, separator, ...rows].join('\n');
}

/**
 *
 * @param jobName
 */
export async function cmdStatus(jobName?: string): Promise<void> {
  const config = loadCredentials();

  try {
    assertConfigured(config);
  } catch (e) {
    console.error('Error:', (e as Error).message);
    process.exit(1);
  }

  const ops = makeRemoteOps(config, 'real');

  if (jobName) {
    // Single job
    const paths = makeJobV3Paths(config.projectDir, jobName);
    const { exitCode, stdout } = await ops.runRemote(
      `cat ${paths.statusFile} 2>/dev/null`,
    );
    if (exitCode !== 0 || !stdout.trim()) {
      console.error(
        `No job '${jobName}' found. (No status.json at ${paths.statusFile})`,
      );
      process.exit(1);
    }
    const details = parseV3Lockfile(stdout);
    if (!details) {
      console.error(`Could not parse status.json for '${jobName}'.`);
      process.exit(1);
    }
    console.log(formatJobTable([details]));
  } else {
    const paths = makeV3Paths(config.projectDir);

    // All jobs — use jq to combine all job_details.json into a JSON array
    const { stdout } = await ops.runRemote(
      `shopt -s nullglob; files=(${paths.statusGlob}); if [ \${#files[@]} -eq 0 ]; then echo '[]'; else jq -s '.' "\${files[@]}"; fi`,
    );
    let jobs: LockfileV3[] = [];
    try {
      const parsed = JSON.parse(stdout || '[]');
      if (Array.isArray(parsed)) {
        jobs = parsed
          .map((j) => parseV3Lockfile(JSON.stringify(j)))
          .filter((j): j is LockfileV3 => j !== null);
      }
    } catch {
      // ignore parse errors — show empty list
    }
    console.log(formatJobTable(jobs));
  }
}
