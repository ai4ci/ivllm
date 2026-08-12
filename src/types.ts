import type EventEmitter from 'events';

// =================================
// CONFIG AND CMD LINE OPTIONS
// =================================

/**
 * SSH and HPC connection credentials.
 *
 * Loaded from `~/.config/ivllm/config.json` by {@link loadCredentials}.
 *
 * | Field | Description |
 * |-------|-------------|
 * | `loginHost` | Login node hostname (e.g. `'XXXX.aip2.isambard'`) |
 * | `username` | HPC username (e.g. `'YYYY.XXXX'`) |
 * | `projectDir` | Shared project directory (e.g. `'/projects/XXXX'`) |
 * | `defaultLocalPort` | Default local port for the SSH tunnel (default `11434`) |
 * | `hfToken` | Optional HuggingFace token for gated models |
 */
export interface Credentials {
    /** Login node hostname (e.g. `'XXXX.aip2.isambard'`) */
    loginHost: string;
    /** HPC username (e.g. `'YYYY.XXXX'`) */
    username: string;
    /** Shared project directory on the HPC (e.g. `'/projects/XXXX'`) */
    projectDir: string;
    /** Optional HuggingFace access token for gated models */
    hfToken?: string;
}

/**
 * A single environment variable entry with a key-value pair.
 *
 * Used to pass user-defined environment variables from the vLLM YAML config
 * into the remote SLURM script before launching the vLLM server.
 *
 * | Field | Description |
 * |-------|-------------|
 * | `key` | Environment variable name (e.g. `'HF_HOME'`) |
 * | `value` | Environment variable value |
 */
export interface EnvVarEntry {
    /** Environment variable name (e.g. `'HF_HOME'`) */
    key: string;
    /** Environment variable value */
    value: string;
}

// =================================
// REMOTE OPERATIONS INTERFACE
// =================================

/**
 * Options for executing a remote command via {@link RemoteOps.runRemote}.
 *
 * | Field | Description |
 * |-------|-------------|
 * | `env` | Environment variables to prefix the command (e.g. HF token) |
 * | `silent` | When `true` capture stdout instead of streaming to terminal |
 */
export type RunRemoteOptions = {
    /** Environment variables to prefix the remote command (e.g. HF token) */
    env: EnvVarEntry[];
    /** When `true`, capture stdout instead of streaming to terminal */
    silent?: boolean;
};

/**
 * Result returned by {@link RemoteOps.runRemote}.
 *
 * | Field | Description |
 * |-------|-------------|
 * | `exitCode` | Process exit code (0 = success) |
 * | `stdout` | Captured standard output |
 */
export type RunRemoteResult = {
    /** Process exit code (0 indicates success) */
    exitCode: number;
    /** Captured standard output from the remote command */
    stdout: string;
};

/**
 * An {@link EventEmitter} representing a long-running client-side resource
 * (SSH tunnel, log tailer, etc.) that can be polled for liveness and asked
 * to shut down.
 *
 * Implementations are free to also emit `'close'` (e.g. when a poll loop
 * detects the resource died, or once `close()` completes) — `isAlive()`/
 * `close()` are the primitives every implementation must provide; events
 * are an optional convenience on top.
 *
 * | Member | Description |
 * |--------|-------------|
 * | `isAlive()` | Check whether the resource is still active |
 * | `close()` | Request shutdown; idempotent, resolves once fully stopped |
 */
export interface CloseableEventEmitter extends EventEmitter {
    /**
     * Check whether the underlying resource is still active.
     * Safe to call repeatedly (e.g. from a polling loop).
     */
    isAlive(): Promise<boolean>;

    /**
     * Request shutdown of the underlying resource. Idempotent — safe to
     * call more than once. Resolves once the resource has fully stopped.
     */
    close(): Promise<void>;
}

// =================================
// VLLM API
// =================================

/**
 * Response body from the vLLM `/v1/models` endpoint.
 *
 * Conforms to the OpenAI API-compatible format (see upstream docs). The `data`
 * array contains one entry per model currently loaded by the server.
 *
 * | Field | Description |
 * |-------|-------------|
 * | `object` | Always `'list'` |
 * | `data` | Array of loaded model objects |
 * | `data[].id` | Model identifier (e.g. `'Qwen/Qwen2.5-7B-Instruct'`) |
 * | `data[].max_model_len` | Maximum context length, if reported |
 * @see https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html
 */
export interface V1ModelsResponse {
    /** Always `'list'` for a models listing response */
    object: 'list';
    /** Array of loaded model objects */
    data: Array<{
        /** Model identifier (e.g. `'Qwen/Qwen2.5-7B-Instruct'`) */
        id: string;
        /** Any additional fields from the vLLM response */
        [key: string]: any;
        /** Maximum context length reported by the server, if available */
        max_model_len?: number;
    }>;
}

// =================================
// V3 LOCKFILE
// =================================

/**
 * Lockfile state machine (v3).
 *
 * `cancel` is a request state — the monitor detects it and transitions
 * to `stopped`. All other states are lifecycle phases.
 */
export type LockfileState =
    | 'pending'
    | 'initialising'
    | 'running'
    | 'failed'
    | 'stopped'
    | 'cancel'
    | 'abort';

/**
 * Full lockfile schema for the v3 `status.json` format.
 *
 * Written by the CLI (on LOGIN) and updated by the bash framework
 * (on COMPUTE). Read by the CLI for monitoring and reconnection.
 *
 * | Field | Description |
 * |-------|-------------|
 * | `status` | Current lifecycle state |
 * | `jobName` | User-provided job name |
 * | `model` | HuggingFace model ID |
 * | `serverPort` | Random high port for vLLM server |
 * | `requestedTime` | ISO-8601 timestamp of job creation |
 * | `idleTimeout` | Minutes of inactivity before auto-shutdown (-1 = never) |
 * | `backend` | Optional backend identifier (e.g. `'isambard-vllm'`) |
 * | `backendConfig` | Backend-specific metadata (opaque to CLI) |
 * | `slurmJobId` | SLURM job ID (Isambard backend) |
 * | `computeHostname` | Compute node hostname for SSH tunnel |
 * | `startTime` | ISO-8601 timestamp of job start |
 * | `stopTime` | ISO-8601 timestamp of job stop |
 * | `reason` | Human-readable reason for stopping/failure |
 * | `exitCode` | vLLM exit code on failure |
 */
export interface LockfileV3 {
    status: LockfileState;
    jobName: string;
    model: string;
    serverPort: number;
    user: string;
    requestedTime: string;
    idleTimeout: number;
    // backend?: string;
    // backendConfig?: Record<string, unknown>;
    slurmJobId?: string;
    computeHostname?: string;
    startTime?: string;
    stopTime?: string;
    reason?: string;
    exitCode?: number;
    resources?: string;
}

/**
 * Optional metadata block in vllm.yaml config files.
 *
 * This block is stripped before the config is passed to vLLM but is
 * preserved in the job directory for debugging and provenance.
 */
export interface VllmConfigMetadata {
    /** Config format version (user-managed) */
    version?: string;
    /** Config author (e.g. email or username) */
    author?: string;
    /** Lifecycle stage for config maturity tracking */
    lifecycle?: 'experimental' | 'maturing' | 'stable' | 'deprecated';
    /** Minimum vLLM version required for this config */
    targetVllmVersion?: string;
    /** Free-text description of what this config is for */
    description?: string;
}

// Benchmarking ------

/** Per-job status inside a BenchmarkStatus snapshot. */
export interface BenchmarkJobStatus {
    status: string;           // LockfileV3.status values, or "unknown"
    reason: string | null;
}

/** Parsed benchmarking_status.json — see ivllm-bench.sh's own doc comment
 *  for the authoritative schema; this type must stay in sync with it. */
export interface BenchmarkStatus {
    pid: number;
    updated: string;          // ISO8601 UTC
    complete: boolean;        // true once EVERY job has a terminal status
    counts: {
        pending: number;
        initialising: number;
        running: number;
        stopped: number;
        failed: number;
    };
    jobs: Record<string, BenchmarkJobStatus>;
}
