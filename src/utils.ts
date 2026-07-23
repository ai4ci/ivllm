import type { LockfileV3 } from './types.ts';

/**
 *
 * @param job
 */
export function formatJobRow(job: LockfileV3): string {
    const parts: string[] = [
        job.jobName.padEnd(8),
        job.status.padEnd(10),
        job.user.padEnd(8),
        job.model.padEnd(20),
        (job.stopTime ?? '-').padEnd(8),
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
        'JOB'.padEnd(8) +
        '  ' +
        'STATUS'.padEnd(10) +
        '  ' +
        'USER'.padEnd(8) +
        '  ' +
        'MODEL'.padEnd(20) +
        '  ' +
        'UNTIL'.padEnd(8) +
        '  ' +
        'INFO';
    const separator = '-'.repeat(header.length);
    const rows = jobs.map(formatJobRow);
    return [header, separator, ...rows].join('\n');
}
