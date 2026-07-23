import type { CloseableEventEmitter, LockfileV3, Credentials } from '../types';

export abstract class Backend {
    creds: Credentials;

    constructor(creds: Credentials) {
        this.creds = creds;
    }

    abstract bootstrap(): Promise<void>;
    abstract setup(version: string): Promise<void>;

    abstract connect(
        job: string,
        localPort: number,
    ): Promise<CloseableEventEmitter>;

    abstract requestCancel(job: string, force: boolean): Promise<void>;

    abstract requestStart(
        job: string,
        maxTime: string,
        monitor: boolean,
        config?: string,
    ): Promise<void>;

    abstract getAllJobStatus(): Promise<LockfileV3[]>;

    abstract watchLog(
        job: string,
        node?: string,
        until?: string,
    ): Promise<CloseableEventEmitter>;

    async getJobStatus(job: string): Promise<LockfileV3> {
        const statuses = await this.getAllJobStatus();
        // Assumes 'name' or 'id' holds the job string; adjust property name as needed
        const matchedJob = statuses.find((status) => status.jobName === job);

        if (!matchedJob) {
            throw new Error(`Job status for '${job}' not found.`);
        }

        return matchedJob;
    }

    /**
     * Parse a v3 lockfile JSON string into a {@link LockfileV3} object.
     *
     * Returns `null` for empty, malformed, or unparseable input.
     * @param raw — Raw JSON string from `status.json`
     * @returns Parsed lockfile, or `null` on failure
     */
    protected parseV3Lockfile(raw: string): LockfileV3 | null {
        if (!raw.trim()) return null;
        try {
            const obj = JSON.parse(raw) as Record<string, unknown>;
            if (typeof obj['status'] !== 'string') return null;
            if (typeof obj['jobName'] !== 'string') return null;
            return obj as unknown as LockfileV3;
        } catch {
            return null;
        }
    }

    async isRunning(job: string): Promise<boolean> {
        try {
            const lockfile = await this.getJobStatus(job);
            return lockfile.status == 'running';
        } catch (Error) {
            return false;
        }
    }

    async isStopped(job: string): Promise<boolean> {
        try {
            const lockfile = await this.getJobStatus(job);
            return lockfile.status == 'stopped' || lockfile.status == 'failed';
        } catch (Error) {
            return true;
        }
    }

    async isStartable(job: string): Promise<boolean> {
        try {
            const lockfile = await this.getJobStatus(job);
            return lockfile.status == 'stopped' || lockfile.status == 'failed';
        } catch (Error) {
            return true;
        }
    }

    async isStarting(job: string): Promise<boolean> {
        try {
            const lockfile = await this.getJobStatus(job);
            return (
                lockfile.status == 'pending' ||
                lockfile.status == 'initialising'
            );
        } catch (Error) {
            return false;
        }
    }
}
