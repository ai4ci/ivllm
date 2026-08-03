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
        job.jobName.padEnd(10),
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
        'JOB'.padEnd(10) +
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

/**
 * Compare two dotted-numeric version strings (e.g. "3.0.1" vs "3.1.0").
 * @returns negative if a < b, 0 if equal, positive if a > b
 */
export function compareVersions(a: string, b: string): number {
    const pa = a.split('.').map((n) => parseInt(n, 10) || 0);
    const pb = b.split('.').map((n) => parseInt(n, 10) || 0);
    const len = Math.max(pa.length, pb.length);
    for (let i = 0; i < len; i++) {
        const diff = (pa[i] ?? 0) - (pb[i] ?? 0);
        if (diff !== 0) return diff;
    }
    return 0;
}
