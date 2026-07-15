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
  /** Default local port for the SSH tunnel to the vLLM server */
  defaultLocalPort: number;
  /** Optional HuggingFace access token for gated models */
  hfToken?: string;
}

/**
 * Parsed CLI arguments combined with the vLLM YAML config.
 *
 * Produced by {@link parseStartArgs} and passed to {@link runInferenceSession}
 * as the single source of truth for job configuration.
 *
 * | Field | Description |
 * |-------|-------------|
 * | `jobName` | User-provided name for this session |
 * | `credentials` | SSH/HPC credentials from local config |
 * | `configFile` | Path to the local vllm.yaml on disk |
 * | `configYaml` | Parsed {@link ServeOptions} from the YAML file |
 * | `localPort` | Local port for the SSH tunnel |
 * | `gpuCount` | Total GPUs to request (from YAML or `--gpus`) |
 * | `timeLimit` | SLURM time limit (default `'8:00:00'`) |
 * | `serverPort` | Remote port where vLLM listens (default `8000`) |
 * | `mock` | Run mock server without GPU |
 * | `dryRun` | Preview scripts without connecting to HPC |
 * | `noLaunch` | Skip assistant launcher menu |
 * | `isInteractive` | Run via `srun` with TTY binding |
 * | `preCache` | Build JIT cache then exit |
 * | `cacheKey` | Unique key for the JIT compilation cache |
 */
export interface InferenceJobOptions {
  /** User-provided name for this session */
  jobName: string;
  /** SSH and HPC connection credentials */
  credentials: Credentials;
  /** Path to the local vllm.yaml config file */
  configFile: string;
  /** Parsed vLLM YAML configuration */
  configYaml: ServeOptions;
  /** Local port for the SSH tunnel to the remote server */
  localPort: number;
  /** Total GPU count to request (derived from YAML parallelism or `--gpus`) */
  gpuCount: number;
  /** SLURM time limit string (default `'8:00:00'`) */
  timeLimit: string;
  /** Remote port where the vLLM server listens (default `8000`) */
  serverPort: number;
  /** When true, run a mock HTTP server without requiring a GPU */
  mock: boolean;
  /** When true, generate scripts without connecting to the HPC */
  dryRun: boolean;
  /** When true, skip the AI assistant launcher menu */
  noLaunch: boolean;
  /** When true, run via `srun` with a bound terminal instead of sbatch */
  isInteractive: boolean;
  /** When true, compile models then exit once healthy (JIT cache build) */
  preCache: boolean;
  /** Unique key identifying the JIT compilation cache for this configuration */
  cacheKey: string;
}

/**
 * Shared project-level paths used across all jobs for a given vLLM version.
 *
 * These paths are the same for every session running the same vLLM version
 * on the same project allocation. Computed by {@link makeSimplePaths}.
 *
 * | Field | Description |
 * |-------|-------------|
 * | `remoteProjectDir` | Shared project directory (e.g. `/projects/XXXX`) |
 * | `remoteHomeDir` | User home on HPC (e.g. `/projects/XXXX/home/YYYY.XXXX`) |
 * | `remoteProjectVllmDir` | Shared vLLM installation root |
 * | `remoteProjectVllmPluginsDir` | Directory for shared vLLM plugins |
 * | `remoteProjectVllmVersionDir` | Versioned venv directory |
 * | `remoteProjectVllmVenvActivate` | Path to venv activate script |
 * | `nvhpcDir` | NVHPC SDK installation directory |
 * | `nvhpcRoot` | NVHPC root (e.g. `Linux_aarch64/26.3`) |
 */
export interface SimplePaths {
  /** Shared project directory on the HPC (e.g. `/projects/XXXX`) */
  remoteProjectDir: string;
  /** User home directory on the HPC */
  remoteHomeDir: string;
  /** Root of the shared vLLM installation tree */
  remoteProjectVllmDir: string;
  /** Directory for shared vLLM plugins across all jobs */
  remoteProjectVllmPluginsDir: string;
  /** Versioned vLLM installation directory (e.g. `…/ivllm/0.22.0`) */
  remoteProjectVllmVersionDir: string;
  /** Path to the Python virtualenv `activate` script */
  remoteProjectVllmVenvActivate: string;
  /** NVHPC SDK installation directory */
  nvhpcDir: string;
  /** NVHPC root (e.g. `Linux_aarch64/26.3`) */
  nvhpcRoot: string;
}

/**
 * Complete set of file system paths for a single inference session.
 *
 * Extends {@link SimplePaths} with job-scoped paths. Computed by
 * {@link makePaths} from the job name, model name, cache key, and
 * vLLM version.
 *
 * | Field | Description |
 * |-------|-------------|
 * | `remoteJobDir` | Job working directory on the HPC |
 * | `remoteJobLockFile` | Path to the `job_details.json` lockfile |
 * | `remoteJobVllmConfigFile` | Path to the vllm.yaml on the HPC |
 * | `remoteJobVllmPluginsDir` | Per-job plugin directory (symlinked from project) |
 * | `remoteJobScriptFile` | Path to the generated SLURM script |
 * | `remoteJobLogFile` | Path to the vLLM log file |
 * | `remoteProjectHfDir` | Shared HuggingFace cache root |
 * | `remoteProjectHfModelDir` | Cache path for the specific model |
 * | `remoteProjectJobCacheDir` | Shared JIT cache directory |
 * | `remoteProjectJobCacheFile` | Cached tarball of compiled models |
 * | `localCacheDir` | Local machine cache directory |
 * | `localCacheVllmConfigFile` | Local cached copy of the job YAML config |
 */
export interface Paths extends SimplePaths {
  /** Job working directory on the HPC (e.g. `/projects/XXXX/home/YYY.XXXX/<jobName>`) */
  remoteJobDir: string;
  /** Path to the `job_details.json` lockfile on the HPC */
  remoteJobLockFile: string;
  /** Path to the vllm.yaml file on the HPC */
  remoteJobVllmConfigFile: string;
  /** Per-job plugin directory (usually a symlink to the project plugins dir) */
  remoteJobVllmPluginsDir: string;
  /** Path to the generated SLURM batch script on the HPC */
  remoteJobScriptFile: string;
  /** Path to the vLLM log file on the HPC */
  remoteJobLogFile: string;
  /** Shared HuggingFace cache root directory */
  remoteProjectHfDir: string;
  /** Cache directory for the specific model within the HF cache */
  remoteProjectHfModelDir: string;
  /** Shared directory for JIT compilation caches */
  remoteProjectJobCacheDir: string;
  /** Cached tarball path for compiled model JIT caches */
  remoteProjectJobCacheFile: string;
  /** Local machine cache directory (~/.config/ivllm) */
  localCacheDir: string;
  /** Local cached copy of the job's vllm.yaml config */
  localCacheVllmConfigFile: string;
}

// =================================
// V3 PATHS (ENGINE DIR STRUCTURE)
// =================================

/**
 * Paths for a v3 inference job under the engine directory.
 *
 * All job artifacts live under `$PROJECTDIR/engine/jobs/<jobname>/`.
 * Shared libraries live in `$PROJECTDIR/engine/lib/`.
 *
 * | Field | Description |
 * |-------|-------------|
 * | `engineDir` | Root of the engine directory (`$PROJECTDIR/engine`) |
 * | `engineLibDir` | Shared bash libraries (`$PROJECTDIR/engine/lib`) |
 * | `engineJobsDir` | Root of all job directories |
 * | `jobDir` | Per-job working directory |
 * | `statusFile` | Path to `status.json` lockfile |
 * | `scriptFile` | Path to the generated `slurm.sh` |
 * | `vllmConfigFile` | Path to the vllm.yaml on the HPC |
 * | `logFile` | Path to the vLLM log file |
 * | `jitCacheFile` | Path to the JIT cache tarball |
 */
export interface EnginePathsV3 {
  /** Root of the engine directory (`$PROJECTDIR/engine`) */
  engineDir: string;
  /** Shared bash libraries (`$PROJECTDIR/engine/lib`) */
  engineLibDir: string;
  /** Root of all job directories (`$PROJECTDIR/engine/jobs`) */
  engineJobsDir: string;
  /** Per-job working directory (`$PROJECTDIR/engine/jobs/<jobname>`) */
  jobDir: string;
  /** Path to `status.json` lockfile */
  statusFile: string;
  /** Path to the generated `slurm.sh` */
  scriptFile: string;
  /** Path to the vllm.yaml on the HPC (raw, with metadata) */
  vllmConfigFile: string;
  /** Path to the vllm.yaml stripped of metadata */
  strippedConfigFile: string;
  /** Path to the JIT cache tarball */
  jitCacheFile: string;
  /** Glob pattern matching all per-node log files (vllm.<NODEID>.log) */
  logFileGlob: string;
}

/**
 * Parsed vLLM YAML configuration with all serving parameters.
 *
 * Produced by {@link parseVllmConfig} from a YAML file. The `raw` field
 * preserves all unparsed keys for passthrough.
 *
 * | Field | Description |
 * |-------|-------------|
 * | `model` | HuggingFace model ID (e.g. `'Qwen/Qwen2.5-7B-Instruct'`) |
 * | `tensorParallelSize` | Tensor parallelism degree |
 * | `pipelineParallelSize` | Pipeline parallelism degree |
 * | `dataParallelSize` | Data parallelism degree |
 * | `maxModelLen` | Maximum model context length |
 * | `enableAutoToolChoice` | Auto-select tools without explicit prompt hints |
 * | `enableReasoning` | Enable reasoning mode (derived from `reasoning-parser` key) |
 * | `minVllmVersion` | Minimum vLLM version required |
 * | `env` | User-defined environment variables for the vLLM process |
 * | `raw` | Unparsed keys from the YAML (for forward compatibility) |
 */
export interface ServeOptions {
  /** HuggingFace model ID (e.g. `'Qwen/Qwen2.5-7B-Instruct'`) */
  model: string;
  /** Tensor parallelism degree (default `1`) */
  tensorParallelSize: number;
  /** Pipeline parallelism degree (default `1`) */
  pipelineParallelSize: number;
  /** Data parallelism degree (default `1`) */
  dataParallelSize: number;
  /** Number of nodes (overrides other calculations) */
  nnodes: number | unknown;
  /** Maximum model context length in tokens */
  maxModelLen: number;
  /** Enable auto tool choice without explicit prompt hints */
  enableAutoToolChoice: boolean;
  /** Enable reasoning mode (derived from presence of `reasoning-parser` key) */
  enableReasoning: boolean;
  /** Minimum vLLM version string required (e.g. `'0.20.0'`) */
  minVllmVersion: string;
  /** Idle timeout in minutes (-1 = never). Config-driven auto-shutdown. */
  idleTimeout: number;
  /** User-defined environment variables for the vLLM process */
  env: EnvVarEntry[];
  /** Unparsed keys from the YAML (preserved for forward compatibility) */
  raw: Record<string, unknown>;
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

/**
 * Mutable runtime state for a remote process (SLURM job, tunnel, etc.).
 *
 * Shared by {@link SessionState} via inheritance. Fields are lazily
 * assigned as the session lifecycle progresses.
 *
 * | Field | Description |
 * |-------|-------------|
 * | `sessionName` | User-provided job name |
 * | `ops` | Remote operations (SSH/SCP) for the login node |
 * | `paths` | File system paths for the project-level directory tree |
 * | `vllmVersion` | vLLM version to use |
 * | `remoteCommand` | Remote command string (for interactive sessions) |
 * | `slurmJobId` | SLURM job ID once submitted |
 * | `process` | Active srun/SSH process (interactive mode) |
 * | `tunnel` | Active SSH tunnel process |
 * | `heartbeatTimer` | Interval timer for the health check heartbeat |
 * | `crashDiagnosticsPrinted` | Guard to avoid duplicate diagnostics output |
 * | `shuttingDown` | Guard to prevent double-shutdown |
 */
export class ProcessState {
  /** User-provided session/job name */
  sessionName!: string;
  /** Remote operations interface for SSH/SCP to the login node */
  ops!: RemoteOps;
  /** File system paths for the project-level directory tree */
  paths!: SimplePaths;
  /** vLLM version string (e.g. `'0.22.0'`) */
  vllmVersion!: string;
  /** Remote command string for interactive sessions */
  remoteCommand?: string;
  /** SLURM job ID once the job has been submitted */
  slurmJobId?: string;
  /** Active srun or SSH process (interactive mode only) */
  process?: CloseableEventEmitter;
  /** Active SSH tunnel process */
  tunnel?: CloseableEventEmitter;
  /** Interval timer handle for the health-check heartbeat */
  heartbeatTimer?: Timer;
  /** Guard to prevent duplicate crash diagnostics output */
  crashDiagnosticsPrinted?: boolean;
  /** Guard to prevent double-shutdown during cleanup */
  shuttingDown?: boolean;

  constructor(init?: Partial<ProcessState>) {
    if (init) Object.assign(this, init);
  }
}

/**
 * Full runtime state for an inference session.
 *
 * Extends {@link ProcessState} with job-specific paths, local operations,
 * and the original parsed arguments. Used by the session pipeline
 * ({@link runInferenceSession}) and shutdown logic ({@link shutdown}).
 */
export class SessionState extends ProcessState {
  /** Full set of paths including job-scoped entries */
  declare paths: Paths;
  /** Local operations (health checks, model queries, port detection) */
  localOps!: LocalOps;
  /** Original parsed CLI arguments and vLLM YAML config */
  startArgs!: InferenceJobOptions;

  constructor(init?: Partial<SessionState>) {
    super();
    if (init) Object.assign(this, init);
  }
}

// =================================
// JOB CONFIGURATION OPTIONS
// =================================

/**
 * Lifecycle status of a vLLM inference job as tracked in the lockfile.
 *
 * | Status | Meaning |
 * |--------|---------|
 * | `'pending'` | Lockfile created, SLURM job not yet submitted |
 * | `'initialising'` | SLURM job running — model download, venv setup, vLLM startup |
 * | `'running'` | vLLM `/health` returned 2xx, server is accepting requests |
 * | `'failed'` | vLLM exited with a non-zero code or was killed |
 * | `'timeout'` | SLURM job exceeded its time limit |
 */
export type JobStatus =
  | 'pending'
  | 'initialising'
  | 'running'
  | 'failed'
  | 'timeout';

/**
 * Metadata extracted from a locally cached job config file.
 *
 * Produced by {@link listJobConfigs}. Used by the `ivllm list` and
 * `ivllm status` commands to display job information.
 *
 * | Field | Description |
 * |-------|-------------|
 * | `jobName` | Name used by the user (e.g. `'qwen2'`) |
 * | `filePath` | Absolute path to the cached YAML file |
 * | `model` | Parsed model ID from the YAML |
 * | `tensorParallelSize` | Tensor parallelism from the YAML |
 * | `pipelineParallelSize` | Pipeline parallelism from the YAML |
 */
export interface JobConfigEntry {
  /** Name used by the user (e.g. `'qwen2'`) */
  jobName: string;
  /** Absolute path to the cached YAML config file on disk */
  filePath: string;
  /** HuggingFace model ID parsed from the YAML, if present */
  model?: string;
  /** Tensor parallelism parsed from the YAML, if present */
  tensorParallelSize?: number;
  /** Pipeline parallelism parsed from the YAML, if present */
  pipelineParallelSize?: number;
}

/**
 * Remote metadata written to `job_details.json` on the HPC login node.
 *
 * Written by the SLURM script on the remote side as the job progresses
 * through states: `pending → initialising → running → (failed|timeout)`.\n* Polling local code reads this file via SSH to track status and extract
 * the compute hostname for SSH tunnel establishment.
 *
 * | Field | Description |
 * |-------|-------------|
 * | `status` | Current {@link JobStatus} |
 * | `job_name` | User-provided job name |
 * | `slurm_job_id` | SLURM-assigned job ID once submitted |
 * | `compute_hostname` | Compute node hostname (extracted for tunneling) |
 * | `model` | HuggingFace model ID |
 * | `server_port` | Port where vLLM listens |
 * | `error` | Error message if the job failed |
 * @see parseJobDetails
 */
export interface JobDetails {
  /** Current job {@link JobStatus} */
  status: JobStatus;
  /** User-provided job name */
  job_name: string;
  /** SLURM-assigned job ID (set once `sbatch` succeeds) */
  slurm_job_id?: string;
  /** Compute node hostname (extracted for SSH tunneling) */
  compute_hostname?: string;
  /** HuggingFace model ID */
  model?: string;
  /** Port where the vLLM server listens */
  server_port?: number;
  /** Error message if the job failed */
  error?: string;
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
 * Interface for executing remote operations on the Isambard HPC login node.
 *
 * Implemented by {@link makeRemoteOps} with two modes:
 *
 * - **Real mode**: Actual SSH/SCP execution
 * - **Dry-run mode**: Mock implementations for E2E testing
 *
 * | Method | Description |
 * |--------|-------------|
 * | `runRemote` | Execute a command on the login node |
 * | `streamSrun` | Stream an `srun` command with TTY output |
 * | `copyFile` | SCP a file to the login node |
 * | `tailRemoteLog` | Tail a remote log file via SSH |
 * | `spawnTunnel` | Create an SSH port-forwarding tunnel |
 * | `matchVllmVersion` | Find best installed vLLM version |
 * | `checkSSH` | Verify SSH connectivity to the login node |
 */
export interface RemoteOps {
  /**
   * Execute a command on the login node via SSH.
   * @param command - Shell command to run
   * @param options - Execution options (env, silent)
   * @returns Promise resolving to exit code and stdout
   */
  runRemote(
    command: string,
    options?: RunRemoteOptions,
  ): Promise<RunRemoteResult>;

  /**
   * Execute an `srun` command on the login node with TTY output.
   * Parses the stderr for a SLURM job ID and stores it in `sessionState`.
   * @param command - `srun` command to execute
   * @param sessionState - State whose `slurmJobId` is updated on detection
   * @param options - Execution options
   * @returns Closeable event emitter representing the running process
   */
  streamSrun(
    command: string,
    sessionState: ProcessState,
    options?: RunRemoteOptions,
  ): CloseableEventEmitter;

  /**
   * Copy a local file to the login node via SCP.
   * @param localPath - Path to the local source file
   * @param remotePath - Destination path on the login node
   */
  copyFile(localPath: string, remotePath: string): Promise<void>;

  /**
   * Continuously tail a remote file and stream lines to stdout.
   * @param remotePath - Absolute path to the remote log file
   * @param prefix - Optional string prepended to every output line
   * @returns An object with a `stop()` method to close the connection
   */
  tailRemoteLog(remotePath: string, prefix?: string): { stop: () => void };

  /**
   * Spawn a persistent forward SSH tunnel (localPort → remoteHost:remotePort).
   * @param localPort - Port to listen on locally
   * @param remoteHost - Remote host (typically a compute node)
   * @param remotePort - Remote port (e.g. vLLM server port)
   * @returns Closeable event emitter representing the tunnel process
   */
  spawnTunnel(
    localPort: number,
    remoteHost: string,
    remotePort: number,
  ): CloseableEventEmitter;

  /**
   * Find the best installed vLLM version meeting a minimum requirement.
   * @param minVllmVersion - Minimum acceptable version string
   * @returns The selected version string
   */
  matchVllmVersion(minVllmVersion: string): Promise<string>;

  /**
   * Verify SSH connectivity to the login node.
   * @returns Promise resolving to true when connectivity is confirmed
   */
  checkSSH(): Promise<boolean>;
}

/**
 * Interface for local operations (health checks, model queries, port detection).
 *
 * Implemented by {@link makeLocalOps} with two modes:
 *
 * - **Real mode**: HTTP requests to localhost + `lsof`/`ps` on the local machine
 * - **Dry-run mode**: Mock implementations returning synthetic data
 *
 * | Method | Description |
 * |--------|-------------|
 * | `checkLocalHealth` | Probe the `/health` endpoint |
 * | `queryModels` | GET `/v1/models` from the vLLM server |
 * | `isLocalPortInUse` | Check if a local port is occupied |
 */
export interface LocalOps {
  /**
   * Probe the vLLM `/health` endpoint and return whether it responds 2xx.
   * @param localPort - Local port of the SSH tunnel
   * @returns Promise resolving to true when the endpoint responds 2xx
   */
  checkLocalHealth(localPort: number): Promise<boolean>;

  /**
   * Query the OpenAI-compatible `/v1/models` endpoint for available models.
   * @param localPort - Local port of the SSH tunnel
   * @returns Promise resolving to the models response
   */
  queryModels(localPort: number): Promise<V1ModelsResponse>;

  /**
   * Check whether a local port is occupied by another process.
   * @param localPort - Port number to check
   * @returns Promise resolving to { pid, process } if occupied, or null
   */
  isLocalPortInUse(
    localPort: number,
  ): Promise<{ pid: string; process: string } | null>;
}

/**
 * Interface for monitoring a remote job's lifecycle.
 *
 * Called after a job is submitted or started to observe progress,
 * check health, and trigger shutdown on failure.
 *
 * | Method | Description |
 * |--------|-------------|
 * | `start` | Begin monitoring the given process state |
 */
export interface RemoteMonitor {
  /**
   * Begin monitoring a remote job.
   * @param state - The {@link ProcessState} to monitor
   */
  start(state: ProcessState): Promise<void>;
}

/**
 * An {@link EventEmitter} that can be forcefully terminated.
 *
 * Used to represent long-running child processes (SSH tunnels, srun commands)
 * that may need to be killed during shutdown. Extends Node's `EventEmitter`
 * and adds a `kill()` method for process termination.
 *
 * | Member | Description |
 * |--------|-------------|
 * | `kill()` | Terminate the underlying process, optionally with a signal |
 */
export interface CloseableEventEmitter extends EventEmitter {
  /**
   * Terminate the underlying process.
   * @param signal - Optional signal to send (default `'SIGTERM'`)
   * @returns true if the process was successfully targeted
   */
  kill(signal?: NodeJS.Signals | number): boolean;
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
  | 'cancel';

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
 * | `vllmPid` | vLLM process ID on the compute node |
 * | `reason` | Human-readable reason for stopping/failure |
 * | `exitCode` | vLLM exit code on failure |
 */
export interface LockfileV3 {
  status: LockfileState;
  jobName: string;
  model: string;
  serverPort: number;
  requestedTime: string;
  idleTimeout: number;
  backend?: string;
  backendConfig?: Record<string, unknown>;
  slurmJobId?: string;
  computeHostname?: string;
  startTime?: string;
  stopTime?: string;
  vllmPid?: number;
  reason?: string;
  exitCode?: number;
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

// =================================

// Previously used as the input for `renderInferenceScript()` before the
// project migrated to Handlebars templates (`src/templates/inference.ts`).
//
// The old function assembled the SLURM batch script from this object,
// interpolating values into a string template. After the refactor,
// {@link makeSimplePaths} and {@link makePaths} build all path information
// into the {@link Paths} object, and the template engine consumes only
// the paths, env vars, and control flags — no longer this monolithic
// options interface.
//
// | Field | Description |
// |-------|-------------|
// | `jobName` | User-provided session name |
// | `model` | HuggingFace model ID |
// | `vllmVersion` | vLLM version string |
// | `hfHome` | HuggingFace cache directory |
// | `configFileName` | Name of the local YAML config |
// | `workDir` | Job working directory on the HPC |
// | `serverPort` | Port the vLLM server listens on |
// | `gpuCount` | Total GPUs to allocate |
// | `nodeCount` | Number of SLURM nodes |
// | `timeLimit` | SLURM time limit string |
// | `envVars` | Environment variables to inject |
// | `isInteractive` | Whether to run via `srun` (TTY) |
// | `cacheKey` | JIT compilation cache key |
// | `preCache` | Whether to pre-cache compiled models |
//
// export interface InferenceScriptOptions {
//     jobName: string;
//     model: string;
//     vllmVersion: string;
//     hfHome: string;
//     configFileName: string;
//     workDir: string;
//     serverPort: number;
//     gpuCount: number;
//     nodeCount: number;
//     timeLimit: string;
//     envVars: EnvVarEntry[];
//     isInteractive: boolean;
//     cacheKey: string;
//     preCache: boolean;
// }

/**
 * Previously passed to the monitor pipeline inside
 * `runInferenceSession()` before the project migrated to a
 * {@link RemoteMonitor} interface.
 *
 * The old function used this to connect to a running vLLM server,
 * probe health endpoints, and drive the shutdown sequence. After the
 * refactor, {@link SessionState} and {@link ProcessState} carry all the
 * runtime information the monitor needs; the config is no longer
 * forwarded separately.
 *
 * | Field | Description |
 * |-------|-------------|
 * | `localPort` | Local SSH-tunnel port |
 * | `serverPort` | Remote vLLM server port |
 * | `config` | SSH connection credentials |
 * | `model` | Model identifier string |
 * | `maxModelLen` | Maximum context length, if known |
 * | `enableAutoToolChoice` | Auto-tool-choice flag |
 * | `enableReasoning` | Reasoning-mode flag |
 * | `isInteractive` | Interactive-session flag |
 */
// export interface MonitorRuntimeOpts {
//     localPort: number;
//     serverPort: number;
//     config: Credentials;
//     model: string;
//     maxModelLen?: number;
//     enableAutoToolChoice: boolean;
//     enableReasoning: boolean;
//     isInteractive: boolean;
// }
