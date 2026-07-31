import type { LockfileV3 } from './types.ts';

/**
 * Format a single job's lockfile status as a single-row string.
 *
 * Columns: `JOB`, `STATUS`, `USER`, `MODEL`, `UNTIL`, `INFO`.
 * @param job — Lockfile status object
 * @returns Formatted row string
 */
export function formatJobRow(job: LockfileV3): string {
    const parts: string[] = [
        job.jobName.padEnd(8),
        job.status.padEnd(14),
        (job.user ?? 'unknown').padEnd(12),
        (job.model ?? 'unknown').padEnd(40),
        (job.stopTime ?? '-').padEnd(25),
        job.reason ?? '-',
    ];
    return parts.join('  ').trimEnd();
}

/**
 * Format a list of jobs as a table with header, separator, and rows.
 *
 * Returns `'No active ivllm jobs found.'` for empty arrays.
 * @param jobs — Array of lockfile status objects
 * @returns Formatted table string
 */
export function formatJobTable(jobs: LockfileV3[]): string {
    if (jobs.length === 0) return 'No active ivllm jobs found.';
    const header =
        'JOB'.padEnd(8) +
        '  ' +
        'STATUS'.padEnd(14) +
        '  ' +
        'USER'.padEnd(12) +
        '  ' +
        'MODEL'.padEnd(40) +
        '  ' +
        'UNTIL'.padEnd(25) +
        '  ' +
        'INFO';
    const separator = '-'.repeat(header.length);
    const rows = jobs.map(formatJobRow);
    return [header, separator, ...rows].join('\n');
}
