import type { LockfileV3 } from './types.ts';
import { format, isPast } from 'date-fns';

/**
 * Format a single job's lockfile status as a single-row string.
 *
 * Columns: `JOB`, `STATUS`, `USER`, `MODEL`, `UNTIL`, `INFO`.
 * @param job — Lockfile status object
 * @returns Formatted row string
 */
export function formatJobRow(job: LockfileV3): string {
    const stop =
        job.stopTime && !isPast(new Date(job.stopTime!))
            ? format(new Date(job.stopTime!), 'HH:mm dd/MM')
            : undefined;

    const parts: string[] = [
        job.jobName.padEnd(10),
        job.status.padEnd(10).slice(0, 10),
        (job.resources ?? '-').padEnd(9).slice(0, 9),
        (stop ?? '-').padEnd(12),
        (job.reason ?? '-').padEnd(12).slice(0, 12),
        (job.user ?? 'unknown').padEnd(12).slice(0, 12),
        job.model ?? 'unknown',
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
        'STATUS'.padEnd(10) +
        '  ' +
        'RESOURCES'.padEnd(9) +
        '  ' +
        'UNTIL'.padEnd(12) +
        '  ' +
        'INFO'.padEnd(12) +
        '  ' +
        'USER'.padEnd(12) +
        '  ' +
        'MODEL';

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
