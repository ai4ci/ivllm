import type {
  Credentials,
  JobDetails,
  InferenceJobOptions,
  Paths,
  SimplePaths,
  LockfileV3,
  EnginePathsV3,
} from './types';
import { jobConfigPath, parseVllmConfig, saveJobConfig } from './vllm-config';
import { existsSync } from 'fs';
import os from 'os';
import crypto from 'crypto';

/**
 *
 * @param raw
 */
export function parseJobDetails(raw: string): JobDetails | null {
  if (!raw.trim()) return null;
  try {
    const obj = JSON.parse(raw) as Record<string, unknown>;
    if (typeof obj['status'] !== 'string') return null;
    return obj as unknown as JobDetails;
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
  args: string[],
  config: Credentials,
): Promise<InferenceJobOptions> {
  // First positional arg is job name — it must not start with --
  const jobName = args[0] && !args[0].startsWith('--') ? args[0] : null;
  if (!jobName) throw new Error('Job name is required as the first argument');

  // Parse boolean flags and key=value flags
  const boolFlags = new Set<string>();
  const flags: Record<string, string> = {};
  for (let i = 1; i < args.length; i++) {
    const arg = args[i];
    if (!arg?.startsWith('--')) continue;
    const key = arg.slice(2);
    const next = args[i + 1];
    if (!next || next.startsWith('--')) {
      boolFlags.add(key);
    } else {
      flags[key] = next;
      i++;
    }
  }

  const mock = boolFlags.has('mock');
  const dryRun = boolFlags.has('dry-run');
  const noLaunch = boolFlags.has('no-launch');
  const preCache = boolFlags.has('create-cache');
  const configPath = flags['config'] ?? jobConfigPath(jobName);

  if (!existsSync(configPath)) {
    throw new Error(
      `No --config provided and no stored config found for '${jobName}'.\n  First run: ivllm start ${jobName} --config <path>`,
    );
  }

  const yaml = parseVllmConfig(configPath);
  if (flags['config']) saveJobConfig(jobName, flags['config']);

  const gpuCount = flags['gpus']
    ? parseInt(flags['gpus'], 10)
    : yaml.nnodes
      ? yaml.nnodes * 4
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
 *
 * @param config
 * @param vllmVersion
 */
export function makeSimplePaths(
  config: Credentials,
  vllmVersion: string,
): SimplePaths {
  const remoteProjectDir = config.projectDir;
  const remoteHomeDir = `${remoteProjectDir.replace('/projects', '/home')}/${config.username}`;
  const remoteProjectVllmDir = `${remoteProjectDir}/ivllm`;
  const remoteProjectVllmPluginsDir = `${remoteProjectVllmDir}/plugins`;
  const remoteProjectVllmVersionDir = `${remoteProjectVllmDir}/${vllmVersion}`;
  const remoteProjectVllmVenvActivate = `${remoteProjectVllmVersionDir}/bin/activate`;
  const nvhpcDir = `${remoteProjectVllmDir}/nvhpc`;
  const nvhpcRoot = `${nvhpcDir}/Linux_aarch64/26.3`;

  return {
    remoteProjectDir,
    remoteHomeDir,
    remoteProjectVllmDir,
    remoteProjectVllmPluginsDir,
    remoteProjectVllmVersionDir,
    remoteProjectVllmVenvActivate,
    nvhpcDir,
    nvhpcRoot,
  };
}

/**
 *
 * @param config
 * @param jobName
 * @param model
 * @param cacheKey
 * @param vllmVersion
 */
export function makePaths(
  config: Credentials,
  jobName: string,
  model: string,
  cacheKey: string,
  vllmVersion: string,
): Paths {
  const hfModelKey = model.includes('/')
    ? 'models--' + model.replace('/', '--')
    : 'models--' + model;

  const base = makeSimplePaths(config, vllmVersion);
  const remoteJobDir = `${base.remoteHomeDir}/${jobName}`;
  const remoteJobLockFile = `${remoteJobDir}/job_details.json`;
  const remoteJobScriptFile = `${remoteJobDir}/slurm.sh`;
  const remoteJobLogFile = `${remoteJobDir}/vllm.log`;
  const remoteJobVllmConfigFile = `${remoteJobDir}/${jobName}.yaml`;
  const remoteJobVllmPluginsDir = `${remoteJobDir}/plugins`;
  const remoteProjectHfDir = `${base.remoteProjectDir}/hf`;
  const remoteProjectHfModelDir = `${remoteProjectHfDir}/hub/${hfModelKey}`;
  const remoteProjectJobCacheDir = `${base.remoteProjectVllmDir}/cache/${cacheKey}`;
  const remoteProjectJobCacheFile = `${remoteProjectJobCacheDir}/cache.tar.gz`;
  const localCacheDir = `${os.homedir()}/.config/ivllm`;
  const localCacheVllmConfigFile = `${localCacheDir}/${jobName}.yaml`;

  return {
    ...base,
    remoteJobDir,
    remoteJobLockFile,
    remoteJobScriptFile,
    remoteJobLogFile,
    remoteJobVllmConfigFile,
    remoteJobVllmPluginsDir,
    remoteProjectHfDir,
    remoteProjectHfModelDir,
    remoteProjectJobCacheDir,
    remoteProjectJobCacheFile,
    localCacheDir,
    localCacheVllmConfigFile,
  };
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
export function makeV3Paths(
  projectDir: string,
  jobName: string,
): EnginePathsV3 {
  const engineDir = `${projectDir.replace(/\/+$/, '')}/engine`;
  const engineLibDir = `${engineDir}/lib`;
  const engineJobsDir = `${engineDir}/jobs`;
  const jobDir = `${engineJobsDir}/${jobName}`;

  return {
    engineDir,
    engineLibDir,
    engineJobsDir,
    jobDir,
    statusFile: `${jobDir}/status.json`,
    scriptFile: `${jobDir}/slurm.sh`,
    vllmConfigFile: `${jobDir}/vllm.yaml`,
    strippedConfigFile: `${jobDir}/vllm.stripped.yaml`,
    jitCacheFile: `${jobDir}/jit-cache.tar.gz`,
    logFileGlob: `${jobDir}/vllm.*.log`,
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
