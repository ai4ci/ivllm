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
  /** Root of the model dir (`$PROJECTDIR/model`) */
  modelDir: string;
  /** Root of the hf executable (`$PROJECTDIR/model/venv`) */
  hfVenvDir: string;
  /** Root of the HF_HOME directory (`$PROJECTDIR/model/hf`) */
  hfHomeDir: string;
  /** Root of the engine directory (`$PROJECTDIR/engine`) */
  engineDir: string;
  /** Shared bash libraries (`$PROJECTDIR/engine/lib`) */
  engineLibDir: string;
  /** Root of all job directories (`$PROJECTDIR/engine/jobs`) */
  engineJobsDir: string;
  /** Glob to select all status files (`$PROJECTDIR/engine/jobs/XX/status.json`) */
  statusGlob: string;
}

export interface JobEnginePathsV3 extends EnginePathsV3 {
  /** Per-job working directory (`$PROJECTDIR/engine/jobs/<jobname>`) */
  jobDir: string;
  /** Path to `status.json` lockfile (`$PROJECTDIR/engine/jobs/<jobname>/status.json`) */
  statusFile: string;
  /** Path to the generated `slurm.sh` (`$PROJECTDIR/engine/jobs/<jobname>/slurm.sh`) */
  scriptFile: string;
  /** Path to the vllm.yaml on the HPC (raw, with metadata) (`$PROJECTDIR/engine/jobs/<jobname>/original-vllm.yaml`) */
  vllmConfigFile: string;
  /** Path to the vllm.yaml stripped of metadata (`$PROJECTDIR/engine/jobs/<jobname>/vllm.yaml`)*/
  strippedConfigFile: string;
  /** Path to the JIT cache tarball (`$PROJECTDIR/engine/jobs/<jobname>/jit-cache.tar.gz`) */
  jitCacheFile: string;
  /** Glob pattern matching all per-node log files (vllm.<NODEID>.log) */
  logFileGlob: string;
}

export interface VllmEnginePathsV3 extends EnginePathsV3 {
  /** Root of the vllm version install dir (`$PROJECTDIR/engine/vllm/<version>/`) */
  vllmVersionDir: string;
}

export interface VllmJobEnginePathsV3
  extends JobEnginePathsV3, VllmEnginePathsV3 {}

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
  paths!: EnginePathsV3;
  /** vLLM version string (e.g. `'0.22.0'`) */
  vllmVersion!: string;
  /** Remote command string for interactive sessions */
  remoteCommand?: string;
  /** SLURM job ID once the job has been submitted */
  slurmJobId?: string;
  /** Active SSH tunnel process */
  tunnel?: CloseableEventEmitter;
  /** Interval timer handle for the health-check heartbeat */
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
  declare paths: VllmJobEnginePathsV3;
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

// =================================
// REMOTE OPERATIONS INTERFACE
// =================================

/**
 * Operation mode for {@link makeRemoteOps}.
 * - `'real'`: Actual SSH/SCP execution against the HPC login node
 * - `'mock'`: Local filesystem sandbox for integration testing (lockfiles
 *   are created and read from a temp directory; no SSH involved)
 * - `'dry-run'`: Print what would happen without executing anything
 */
export type OpsMode = 'real' | 'mock' | 'dry-run';

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
