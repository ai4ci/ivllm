import type { CloseableEventEmitter, LockfileV3, Credentials } from '../types';
import { IsambardBareMetalBackend } from './IsambardBareMetalBackend';

// ====== Backend registry =======

// Adding a backend:
// TODO: Will need to configure backend in credentials and create a wrapper
// structure for the config file that allows a list of credentials to be supplied
// will need to update config cli option to be an --backend option that
// takes all backend params at once and probably validates.
// Then will need backend implementation (plus maybe RemoteOps) and that added
// here.

// Allowed configuration names
export type BackendType = 'isambard'; // | "isambard_container" | "local-llama-cpp"

// Registry Map to ensure compile-time exhaustive checks
const backendRegistry: Record<
    BackendType,
    new (creds: Credentials) => Backend
> = {
    isambard: IsambardBareMetalBackend,
};

// 5. Build the Factory Function
export function getBackend(creds: Credentials): Backend {
    // Runtime validation for the configuration string
    const backendType = 'isambard';
    // const backendType = creds.backend;
    // TODO: add in this when supporting mulitple backends
    if (!(backendType in backendRegistry)) {
        throw new Error(
            `Unsupported backend type: "${backendType}". Choose from: ${Object.keys(backendRegistry).join(', ')}`,
        );
    }

    const TargetBackend = backendRegistry[backendType as BackendType];
    return new TargetBackend(creds);
}

// ====== Beckend class definition =======

/**
 * Abstract base class for vLLM backend implementations.
 *
 * Provides lifecycle state checks and lockfile parsing. Concrete backends
 * (e.g. `IsambardBareMetalBackend`) implement the abstract methods to
 * interact with their specific runtime (SSH + SLURM, local, containers, etc.).
 */
export abstract class Backend {
    /** HPC connection credentials */
    protected creds: Credentials;

    /**
     * Create a backend with the given credentials.
     * @param creds — SSH and HPC connection details
     */
    constructor(creds: Credentials) {
        this.creds = creds;
    }

    /**
     * Bootstrap the backend — verify connectivity, deploy engine scripts, etc.
     */
    abstract bootstrap(): Promise<void>;

    /**
     * Install or update vLLM on the HPC.
     * @param version — vLLM version to install (e.g. `'0.19.1'`)
     * @param force — If true, reinstall even if version already exists
     */
    abstract setup(version: string, force?: boolean): Promise<void>;

    /**
     * Connect to or start a vLLM job.
     * @param job — Job name
     * @param localPort — Local port for the SSH tunnel
     * @returns A closeable event emitter representing the SSH tunnel
     */
    abstract connect(
        job: string,
        localPort: number,
    ): Promise<CloseableEventEmitter>;

    /**
     * Request graceful or forced cancellation of a job.
     * @param job — Job name
     * @param force — If true, kill via scancel; otherwise write cancel to lockfile
     */
    abstract requestCancel(job: string, force: boolean): Promise<void>;

    /**
     * Start a new vLLM job or restart a stopped one.
     * @param job — Job name
     * @param maxTime — SLURM time limit (e.g. `'08:00:00'`)
     * @param monitor — Whether to run monitors on the compute node
     * @param config — Optional path to vLLM YAML config file
     */
    abstract requestStart(
        job: string,
        maxTime: string,
        monitor: boolean,
        config?: string,
    ): Promise<void>;

    /**
     * List all jobs for this backend.
     * @returns Array of lockfile status objects
     */
    abstract getAllJobStatus(): Promise<LockfileV3[]>;

    /**
     * Tail log output for a job.
     * @param job — Job name
     * @param node — Optional node identifier (0=head or 1+=worker)
     * @param until — Optional string which when detected will stop tail
     * @returns A closeable event emitter for the log stream
     */
    abstract watchLog(
        job: string,
        node?: string,
        until?: string,
    ): Promise<CloseableEventEmitter>;

    /**
     * Get the lockfile status for a specific job.
     * @param job — Job name
     * @returns The job's lockfile status
     * @throws Error if the job is not found
     */
    async getJobStatus(job: string): Promise<LockfileV3> {
        const statuses = await this.getAllJobStatus();
        const matchedJob = statuses.find((s) => s.jobName === job);

        if (!matchedJob) {
            throw new Error(`Job status for '${job}' not found.`);
        }

        return matchedJob;
    }

    /**
     * Parse a v3 lockfile JSON string into a {@link LockfileV3} object.
     *
     * Returns `null` for empty, malformed, or unparseable input.
     * Validates that `status` and `jobName` are present string fields.
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

    /**
     * Check if a job is currently running.
     * @param job — Job name
     * @returns `true` if status is `running`, `false` otherwise
     */
    async isRunning(job: string): Promise<boolean> {
        try {
            const lockfile = await this.getJobStatus(job);
            return lockfile.status === 'running';
        } catch {
            return false;
        }
    }

    /**
     * Check if a job is in a stopped state (stopped or failed state). If the
     * status file is missing the job is said to be in a stopped state.
     * @param job — Job name
     * @returns `true` if status is `stopped` or `failed`, `false` otherwise
     */
    async isStopped(job: string): Promise<boolean> {
        try {
            const lockfile = await this.getJobStatus(job);
            return (
                lockfile.status === 'stopped' || lockfile.status === 'failed'
            );
        } catch {
            return true;
        }
    }

    /**
     * Check if a job can be started (stopped or failed state) or no existing
     * job status file (i.e. has never been run before or status file has been
     * deleted).
     * @param job — Job name
     * @returns `true` if status is `stopped` or `failed`, `false` otherwise
     */
    async isStartable(job: string): Promise<boolean> {
        try {
            const lockfile = await this.getJobStatus(job);
            return (
                lockfile.status === 'stopped' || lockfile.status === 'failed'
            );
        } catch {
            return true;
        }
    }

    /**
     * Check if a job is starting up (pending or initialising state).
     * @param job — Job name
     * @returns `true` if status is `pending` or `initialising`, `false` otherwise
     */
    async isStarting(job: string): Promise<boolean> {
        try {
            const lockfile = await this.getJobStatus(job);
            return (
                lockfile.status === 'pending' ||
                lockfile.status === 'initialising'
            );
        } catch {
            return false;
        }
    }
}
