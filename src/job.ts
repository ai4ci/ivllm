import type {
  Credentials,
  InferenceJobOptions,
  LockfileV3,
  EnginePathsV3,
  JobEnginePathsV3,
  VllmJobEnginePathsV3,
  VllmEnginePathsV3,
} from './types';
import { jobConfigPath, parseVllmConfig, saveJobConfig } from './vllm-config';
import crypto from 'crypto';

/**
 *
 * @param raw
 */
export function parseLockFile(raw: string): LockfileV3 | null {
  if (!raw.trim()) return null;
  try {
    const obj = JSON.parse(raw) as Record<string, unknown>;
    if (typeof obj['status'] !== 'string') return null;
    return obj as unknown as LockfileV3;
  } catch {
    return null;
  }
}

//TODO: refactor into

/**
 *
 * @param projectDir
 * @param projectHfDir
 * @param model
 */
export function hfCachePath(projectHfDir: string, model: string): string {
  const cacheKey = model.includes('/')
    ? 'models--' + model.replace('/', '--')
    : 'models--' + model;
  return `${projectHfDir}/hub/${cacheKey}`;
}

/**
 * The main options parsing, yaml config loading and defaults.
 * @param args
 * @param config
 */
export async function parseStartArgs(
  jobName: string,
  path: string | unknown,
  mock: boolean,
  dryRun: boolean,
  config: Credentials,
): Promise<InferenceJobOptions> {
  // First positional arg is job name — it must not start with --
  if (!jobName) throw new Error('Job name is required as the first argument');

  const configPath = path ? path! : jobConfigPath(jobName);

  // TODO: can't check config until ssh connected

  const yaml = parseVllmConfig(configPath);
  if (flags['config']) saveJobConfig(jobName, flags['config']);

  const nnodes = yaml.nnodes;
  const gpuCount = flags['gpus']
    ? parseInt(flags['gpus'], 10)
    : typeof nnodes === 'number' && nnodes > 0
      ? nnodes * 4
      : (yaml.tensorParallelSize ?? 1) *
        (yaml.pipelineParallelSize ?? 1) *
        (yaml.dataParallelSize ?? 1);

  const cacheKey = `${yaml.model}/${yaml.tensorParallelSize}/${yaml.pipelineParallelSize}`;

  return {
    jobName,
    credentials: config,
    configFile: configPath,
    configYaml: yaml,
    isInteractive: false,
    localPort: flags['local-port']
      ? parseInt(flags['local-port'], 10)
      : (config.defaultLocalPort ?? 11434),
    gpuCount,
    timeLimit: flags['time'] ?? '8:00:00',
    serverPort: flags['server-port']
      ? parseInt(flags['server-port'], 10)
      : generateRandomHighPort(),
    mock,
    dryRun,
    noLaunch,
    preCache,
    cacheKey,
  };
}

function generateRandomHighPort(): number {
  const MIN_PORT = 49152;
  const MAX_PORT = 65535;
  const range = MAX_PORT - MIN_PORT + 1;

  // Use crypto.randomInt for cryptographically secure random integers
  const randomPort = crypto.randomInt(0, range) + MIN_PORT;

  return randomPort;
}

/**
 * Build v3 paths from a project directory and job name.
 *
 * Paths follow the new $PROJECTDIR/engine/ structure:
 * ```
 * $PROJECTDIR/engine/
 * ├── lib/                Shared bash framework
 * ├── jobs/<jobname>/     Per-job data
 * │   ├── status.json     Lockfile
 * │   ├── vllm.yaml       Config (raw, with metadata)
 * │   ├── vllm.stripped.yaml  Config (stripped for vLLM)
 * │   ├── slurm.sh        SLURM batch script
 * │   ├── jit-cache.tar.gz    JIT compilation cache
 * │   └── vllm.*.log      Log files (per-node)
 * ├── vllm/               vLLM installations
 * └── hf/                 HuggingFace cache
 * ```
 *
 * @param projectDir - Shared project directory (e.g. `/projects/XXXX`)
 * @param jobName - User-provided job name
 * @returns EnginePathsV3 object
 */
export function makeV3Paths(projectDir: string): EnginePathsV3 {
  const modelDir = `${projectDir.replace(/\/+$/, '')}/mdoel`;
  const engineDir = `${projectDir.replace(/\/+$/, '')}/engine`;
  const engineLibDir = `${engineDir}/lib`;
  const engineJobsDir = `${engineDir}/jobs`;
  const hfVenvDir = `${modelDir}/venv`;
  const hfHomeDir = `${modelDir}/hf`;
  const statusGlob = `${engineJobsDir}/*/status.json`;

  return {
    engineDir,
    engineLibDir,
    engineJobsDir,
    modelDir,
    hfVenvDir,
    hfHomeDir,
    statusGlob,
  };
}

export function makeJobV3Paths(
  projectDir: string,
  jobName: string,
): JobEnginePathsV3 {
  const paths = makeV3Paths(projectDir);
  const jobDir = `${paths.engineJobsDir}/${jobName}`;

  return {
    ...paths,
    jobDir,
    statusFile: `${jobDir}/status.json`,
    scriptFile: `${jobDir}/slurm.sh`,
    vllmConfigFile: `${jobDir}/vllm.yaml`,
    strippedConfigFile: `${jobDir}/vllm.stripped.yaml`,
    jitCacheFile: `${jobDir}/jit-cache.tar.gz`,
    logFileGlob: `${jobDir}/vllm.*.log`,
  };
}

export function makeVllmV3Paths(
  projectDir: string,
  vllmVersion: string,
): VllmEnginePathsV3 {
  const paths = makeV3Paths(projectDir);
  const vllmVersionDir = `${paths.engineDir}/vllm/${vllmVersion}`;

  return {
    ...paths,
    vllmVersionDir,
  };
}

export function makeVllmJobV3Paths(
  projectDir: string,
  vllmVersion: string,
  jobName: string,
): VllmJobEnginePathsV3 {
  const paths = makeJobV3Paths(projectDir, jobName);
  const vllmVersionDir = `${paths.engineDir}/vllm/${vllmVersion}`;

  return {
    ...paths,
    vllmVersionDir,
  };
}

// ── v3 lockfile parser ───────────────────────────────────────────────────────

/**
 * Parse a v3 lockfile JSON string into a {@link LockfileV3} object.
 *
 * Returns `null` for empty, malformed, or unparseable input.
 * @param raw — Raw JSON string from `status.json`
 * @returns Parsed lockfile, or `null` on failure
 */
export function parseV3Lockfile(raw: string): LockfileV3 | null {
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
