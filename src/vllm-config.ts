import {
  readFileSync,
  writeFileSync,
  mkdirSync,
  existsSync,
  copyFileSync,
  readdirSync,
} from 'fs';
import { join } from 'path';
import { tmpdir, homedir } from 'os';
import yaml from 'js-yaml';
import type {
  JobConfigEntry,
  ServeOptions,
  EnvVarEntry,
  VllmConfigMetadata,
} from './types';

/** Keys that are ivllm-specific and must be stripped before passing the config to `vllm serve`. */
// nnodes is stripped because it is not used in ray. It is translated into a number of gpus and then recalculated
// in the mp templates (and held back in the ray templates)
export const IVLLM_ONLY_KEYS = new Set([
  'min-vllm-version',
  'env',
  'nnodes',
  'ivllm',
  'metadata',
  'idle-timeout',
]);

export const JOB_CONFIG_DIR = join(homedir(), '.config', 'ivllm');

/** Lists all stored job configs in the job config directory. */
export function listJobConfigs(): JobConfigEntry[] {
  if (!existsSync(JOB_CONFIG_DIR)) return [];
  const files = readdirSync(JOB_CONFIG_DIR)
    .filter((f) => f.endsWith('.yaml'))
    .sort();
  const entries: JobConfigEntry[] = [];
  for (const file of files) {
    const filePath = join(JOB_CONFIG_DIR, file);
    const jobName = file.replace(/\.yaml$/, '');
    try {
      const parsed = parseVllmConfig(filePath);
      entries.push({
        jobName,
        filePath,
        model: parsed.model,
        tensorParallelSize: parsed.tensorParallelSize,
        pipelineParallelSize: parsed.pipelineParallelSize,
      });
    } catch {
      // If a config file is malformed, still list it but without parsed fields
      entries.push({ jobName, filePath });
    }
  }
  return entries;
}

/**
 * Returns the path where a named job's vLLM config is stored locally.
 * @param jobName
 */
export function jobConfigPath(jobName: string): string {
  return join(JOB_CONFIG_DIR, `${jobName}.yaml`);
}

/**
 * Saves a copy of the given vLLM config file to the local job config store.
 * @param jobName
 * @param sourcePath
 */
export function saveJobConfig(jobName: string, sourcePath: string): void {
  if (!existsSync(JOB_CONFIG_DIR)) {
    mkdirSync(JOB_CONFIG_DIR, { recursive: true });
  }
  copyFileSync(sourcePath, jobConfigPath(jobName));
}

/**
 * Parse a vLLM YAML config file and extract fields that ivllm needs locally
 * (for HF model download and SLURM GPU allocation).
 * @param filePath
 */
export function parseVllmConfig(filePath: string): ServeOptions {
  const raw = readFileSync(filePath, 'utf-8');
  const doc = yaml.load(raw) as Record<string, unknown>;

  const model = doc['model'];
  if (typeof model !== 'string')
    throw new Error('Configuration file does not have `model` parameter');

  const maxModelLen = doc['max-model-len'] ?? doc['max_model_len'];
  if (typeof maxModelLen !== 'number')
    throw new Error(
      'Configuration file does not have a numeric `max-model-len` paramter',
    );

  // vLLM YAML uses kebab-case keys; also accept underscore variant
  const tp = doc['tensor-parallel-size'] ?? doc['tensor_parallel_size'];
  const tensorParallelSize = typeof tp === 'number' ? tp : 1;

  // vLLM YAML uses kebab-case keys; also accept underscore variant
  const nnodes: number | unknown = doc['nnodes'];

  // vLLM YAML uses kebab-case keys; also accept underscore variant
  const pp = doc['pipeline-parallel-size'] ?? doc['pipeline_parallel_size'];
  const pipelineParallelSize = typeof pp === 'number' ? pp : 1;

  const dp = doc['data-parallel-size'] ?? doc['data_parallel_size'];
  const dataParallelSize = typeof dp === 'number' ? dp : 1;

  const minVer = doc['min-vllm-version'];
  const minVllmVersion = typeof minVer === 'string' ? minVer : '0.15.0';

  const toolChoice =
    doc['enable-auto-tool-choice'] ?? doc['enable_auto_tool_choice'];
  const enableAutoToolChoice =
    typeof toolChoice === 'boolean' ? toolChoice : true;

  // Derive enableReasoning from the presence of reasoning-parser (not from enable-reasoning,
  // which is an opencode.js concept and not a valid vLLM config key).
  const reasoningParser = doc['reasoning-parser'] ?? doc['reasoning_parser'];
  const enableReasoning =
    typeof reasoningParser === 'string' && reasoningParser.length > 0
      ? true
      : false;

  const idleTimeoutVal = doc['idle-timeout'] ?? doc['idle_timeout'];
  const idleTimeout = typeof idleTimeoutVal === 'number' ? idleTimeoutVal : 30;

  const envBlock = doc['env'];
  const env: EnvVarEntry[] = [];
  if (envBlock && typeof envBlock === 'object' && !Array.isArray(envBlock)) {
    for (const [k, v] of Object.entries(envBlock)) {
      if (typeof v === 'string' || typeof v === 'number') {
        env.push({ key: k, value: String(v) });
      }
    }
  }

  return {
    model,
    tensorParallelSize,
    pipelineParallelSize,
    dataParallelSize,
    nnodes,
    maxModelLen,
    enableAutoToolChoice,
    enableReasoning,
    minVllmVersion,
    idleTimeout,
    env,
    raw: stripIvllmKeys(doc),
  };
}

/**
 * Parse a vLLM YAML config file and separate out the metadata block.
 *
 * Returns both the config (with metadata stripped) and the metadata block
 * for storage in the job directory.
 * @param filePath — Path to the YAML config file
 * @returns Parsed config and optional metadata
 */
export function parseVllmConfigWithMetadata(filePath: string): {
  config: ServeOptions;
  metadata: VllmConfigMetadata | null;
} {
  const raw = readFileSync(filePath, 'utf-8');
  const doc = yaml.load(raw) as Record<string, unknown>;

  // Extract metadata block
  let metadata: VllmConfigMetadata | null = null;
  const metaBlock = doc['metadata'];
  if (metaBlock && typeof metaBlock === 'object' && !Array.isArray(metaBlock)) {
    metadata = metaBlock as VllmConfigMetadata;
  }

  // Use existing parser for the config (it will strip metadata via IVLLM_ONLY_KEYS)
  const config = parseVllmConfig(filePath);

  return { config, metadata };
}

/**
 * Strip the metadata block from a raw YAML string.
 *
 * Returns the YAML with only the `metadata:` key removed. All other content
 * (including vLLM config keys) is preserved exactly.
 * @param rawYaml — Full YAML content as a string
 * @returns YAML string without the metadata block
 */
export function stripMetadata(rawYaml: string): string {
  const doc = yaml.load(rawYaml) as Record<string, unknown>;
  delete doc['metadata'];
  return yaml.dump(doc, { lineWidth: -1 });
}

/**
 *
 * @param filePath
 */
export function readVllmYaml(filePath: string): Record<string, unknown> {
  const raw = readFileSync(filePath, 'utf-8');
  const doc = yaml.load(raw) as Record<string, unknown>;
  return doc;
}

/**
 * Returns a cleaned YAML string with all ivllm-specific keys removed.
 * vLLM errors on unknown config keys — always use this when uploading to the remote.
 * @param filePath
 * @param doc
 */
export function stripIvllmKeys(
  doc: Record<string, unknown>,
): Record<string, unknown> {
  for (const key of IVLLM_ONLY_KEYS) {
    delete doc[key];
  }
  // Sort keys alphabetically so identical configs produce identical cache keys
  const sorted = Object.fromEntries(
    Object.entries(doc).sort((a, b) =>
      a[0].toLowerCase().localeCompare(b[0].toLowerCase()),
    ),
  );
  return sorted;
}

/**
 * Writes a stripped (ivllm-keys removed) copy of the YAML config to a temp file
 * and returns the temp file path. Caller is responsible for deleting it.
 * @param filePath
 */
export function writeStrippedConfig(filePath: string): string {
  const raw = readVllmYaml(filePath);
  const sorted = stripIvllmKeys(raw);
  const stripped = yaml.dump(sorted, { lineWidth: -1 });
  const tmpPath = join(tmpdir(), `ivllm-stripped-${Date.now()}.yaml`);
  writeFileSync(tmpPath, stripped, 'utf-8');
  return tmpPath;
}

/**
 * Reads env vars from a vLLM config file and returns them as an array
 * of { key, value } entries suitable for rendering into SLURM scripts.
 * @param filePath
 */
export function parseEnvVars(filePath: string): EnvVarEntry[] {
  const raw = readFileSync(filePath, 'utf-8');
  const doc = yaml.load(raw) as Record<string, unknown>;
  const envBlock = doc['env'];
  if (!envBlock || typeof envBlock !== 'object' || Array.isArray(envBlock))
    return [];
  const entries: EnvVarEntry[] = [];
  for (const [k, v] of Object.entries(envBlock)) {
    if (typeof v === 'string' || typeof v === 'number') {
      entries.push({ key: k, value: String(v) });
    }
  }
  return entries;
}
