/**
 * AI coding assistant utilities.
 * Detects available assistants, generates config/env vars for each.
 */

import { spawnSync } from 'child_process';
import { basename } from 'path';

export interface AssistantConfig {
  name: string;
  env: Record<string, string>;
  args: string[];
}

export interface AssistantOption {
  id: number;
  label: string;
  assistant: string;
  useScoder: boolean;
}

export interface OpencodeConfigOptions {
  model: string;
  localPort: number;
  maxModelLen?: number;
  toolCall?: boolean;
  reasoning?: boolean;
  endpointHost?: string;
}

export type AssistantName = 'opencode' | 'claude' | 'copilot' | 'pi';
export type LaunchWrapper = 'none' | 'scoder' | 'sbx';

interface AssistantDefinition {
  name: AssistantName;
  label: string;
}

export type AssistantEnvOptions = OpencodeConfigOptions;

export interface LaunchCommandOptions extends AssistantEnvOptions {
  assistant: AssistantName;
  wrapper: LaunchWrapper;
  cwd: string;
  sandboxName?: string;
}

export interface SbxSandbox {
  name: string;
  agent: string;
  workspace: string;
  status?: string;
}

const ASSISTANTS: AssistantDefinition[] = [
  { name: 'opencode', label: 'OpenCode' },
  { name: 'copilot', label: 'GitHub Copilot' },
  { name: 'claude', label: 'Claude Code' },
  { name: 'pi', label: 'Pi' },
];

/**
 * Check whether a named binary is available on the system PATH.
 *
 * Uses `which <name>` under the hood and returns `true` when the
 * command exits successfully with non-empty output.
 * @param name - Name of the binary to look for (e.g. `'opencode'`)
 * @returns `true` if the binary is found on PATH, `false` otherwise
 */
export function binaryExists(name: string): boolean {
  const result = spawnSync('which', [name], {
    shell: true,
    encoding: 'utf-8',
  });
  return (
    result.status === 0 &&
    result.output[1] != null &&
    result.output[1].trim().length > 0
  );
}

/**
 * Return the subset of known assistant binaries that are available on PATH.
 *
 * Checks for `'opencode'`, `'claude'`, and `'copilot'` in order and
 * filters out any that are not found via {@link binaryExists}.
 * @returns Array of assistant names that are installed and discoverable
 */
export function getAvailableAssistants(): string[] {
  const candidates = ['opencode', 'claude', 'copilot'];
  return candidates.filter(binaryExists);
}

/**
 * Check whether the `scoder` binary is available on PATH.
 *
 * Convenience wrapper around {@link binaryExists}.
 * @returns `true` if `scoder` is found on PATH
 */
export function getScoderAvailable(): boolean {
  return binaryExists('scoder');
}

/**
 * Check whether the `sbx` binary is available on PATH.
 *
 * Convenience wrapper around {@link binaryExists}.
 * @returns `true` if `sbx` is found on PATH
 */
export function getSbxAvailable(): boolean {
  return binaryExists('sbx');
}

/**
 * Return the human-readable label for a given assistant name.
 *
 * Looks up the label from the internal {@link ASSISTANTS} registry.
 * Falls back to the raw name when no label is defined.
 * @param assistant - The assistant identifier (e.g. `'opencode'`, `'claude'`)
 * @returns The display label (e.g. `'OpenCode'`) or the raw name as fallback
 */
export function getAssistantLabel(assistant: AssistantName): string {
  return ASSISTANTS.find((item) => item.name === assistant)?.label ?? assistant;
}

/**
 * Return the list of launch wrappers applicable for a given assistant.
 *
 * Wrappers determine whether the assistant runs directly, via `scoder`,
 * or inside an `sbx` sandbox. For `'pi'` (or when the local assistant
 * binary exists), `'none'` and optionally `'scoder'` are included. If
 * `sbx` is available on PATH, `'sbx'` is always added.
 * @param assistant - The assistant to check wrappers for
 * @param availableAssistants - Installed assistant binary names from PATH
 * @param hasScoder - Whether the `scoder` binary is available
 * @param hasSbx - Whether the `sbx` binary is available
 * @returns Array of applicable wrapper identifiers
 */
export function getAvailableWrappers(
  assistant: AssistantName,
  availableAssistants: string[],
  hasScoder: boolean,
  hasSbx: boolean,
): LaunchWrapper[] {
  const wrappers: LaunchWrapper[] = [];
  const hasLocalAssistant = availableAssistants.includes(assistant);

  // For Pi, we allow wrappers even if the binary isn't installed
  // since it's primarily configuration-based
  if (assistant === 'pi' || hasLocalAssistant) {
    wrappers.push('none');
    if (hasScoder) {
      wrappers.push('scoder');
    }
  }

  if (hasSbx) {
    wrappers.push('sbx');
  }

  return wrappers;
}

/**
 * Generate an OpenCode JSON configuration for connecting to an Isambard vLLM server.
 *
 * Produces a config object with `$schema`, model name, and a provider block
 * using the `@ai-sdk/openai-compatible` adapter. Context length defaults to
 * 4096 tokens. Supports optional `toolCall` and `reasoning` model flags.
 * @param opts - Configuration options including model name, port, and flags
 * @returns OpenCode config record ready for JSON serialization
 */
export function generateOpencodeConfig(
  opts: OpencodeConfigOptions,
): Record<string, unknown> {
  const context = opts.maxModelLen ?? 4096;
  const endpointHost = opts.endpointHost ?? 'localhost';

  const modelEntry: Record<string, unknown> = {
    name: `${opts.model} (Isambard)`,
    limit: { context, output: context },
  };
  if (opts.toolCall) modelEntry['tool_call'] = true;
  if (opts.reasoning) modelEntry['reasoning'] = true;

  return {
    $schema: 'https://opencode.ai/config.json',
    model: `isambard-vllm/${opts.model}`,
    provider: {
      'isambard-vllm': {
        npm: '@ai-sdk/openai-compatible',
        name: 'Isambard vLLM Server',
        options: {
          baseURL: `http://${endpointHost}:${opts.localPort}/v1`,
          apiKey: 'EMPTY',
        },
        models: {
          [opts.model]: modelEntry,
        },
      },
    },
  };
}

/**
 * Generate runtime environment variable overrides for OpenCode.
 *
 * Serialises the {@link generateOpencodeConfig} output into the
 * `OPENCODE_CONFIG_CONTENT` environment variable so OpenCode can be
 * started without a config file on disk.
 * @param opts - Configuration options for the vLLM endpoint
 * @returns Record with `OPENCODE_CONFIG_CONTENT` set to the JSON config
 */
export function generateOpencodeEnv(
  opts: OpencodeConfigOptions,
): Record<string, string> {
  return {
    OPENCODE_CONFIG_CONTENT: JSON.stringify(generateOpencodeConfig(opts)),
  };
}

/**
 * Generate a Pi assistant `models.json` configuration for vLLM integration.
 *
 * Configures the `isambard-vllm` provider with OpenAI-compatible completions
 * API. When `reasoning` is enabled, adds a `thinkingLevelMap` to allow
 * Pi to disable thinking tokens (off/minimal/low/medium → null, high → 'high',
 * xhigh → 'max'). The `compat.supportsDeveloperRole` is set to `false` since
 * vLLM does not understand the "developer" role used for reasoning models.
 * @param opts - Configuration options including model name and reasoning flag
 * @returns Pi models.json configuration record
 */
export function generatePiModelsConfig(opts: OpencodeConfigOptions): Record<string, unknown> {
  const context = opts.maxModelLen ?? 4096;

  const modelEntry: Record<string, unknown> = {
    id: opts.model,
    name: `${opts.model} (Isambard)`,
    contextWindow: context,
    maxTokens: context,
    input: ['text'],
    reasoning: opts.reasoning ?? false,
    // vLLM doesn't understand the "developer" role used for reasoning-capable models
    // so we send the system prompt as a "system" message instead.
    compat: {
      supportsDeveloperRole: false,
    },
  };

  // Add thinking level map if reasoning is enabled
  if (opts.reasoning) {
    modelEntry['thinkingLevelMap'] = {
      off: null, // Disable thinking when not needed
      minimal: null,
      low: null,
      medium: null,
      high: 'high',
      xhigh: 'max',
    };
  }

  return {
    providers: {
      'isambard-vllm': {
        baseUrl: `http://localhost:${opts.localPort}/v1`,
        apiKey: 'EMPTY',
        api: 'openai-completions',
        models: [modelEntry],
      },
    },
  };
}

/**
 * Generate environment variables for GitHub Copilot to proxy through vLLM.
 *
 * Sets `COPILOT_PROVIDER_BASE_URL` to point at the local vLLM tunnel
 * and `COPILOT_MODEL` to the requested model name.
 * @param localPort - Local port of the SSH tunnel to the vLLM server
 * @param model - Model name to use for Copilot completions
 * @returns Copilot environment variable record
 */
export function generateCopilotEnv(
  localPort: number,
  model: string,
): Record<string, string> {
  return {
    COPILOT_PROVIDER_BASE_URL: `http://localhost:${localPort}/v1`,
    COPILOT_MODEL: model,
  };
}

/**
 * Generate environment variables for Claude Code to proxy through vLLM.
 *
 * Sets the {@link ANTHROPIC_BASE_URL}, {@link ANTHROPIC_API_KEY} (to the
 * `'ollama'` placeholder), and the default model names for each Claude
 * model tier (`sonnet`, `opus`, `haiku`) plus the sub-agent model.
 *
 * **Environment variables set**
 *
 * | Variable | Value |
 * |----------|-------|
 * | `ANTHROPIC_BASE_URL` | `http://localhost:{localPort}` |
 * | `ANTHROPIC_API_KEY` | `'ollama'` |
 * | `ANTHROPIC_DEFAULT_SONNET_MODEL` | {model} |
 * | `ANTHROPIC_DEFAULT_OPUS_MODEL` | {model} |
 * | `ANTHROPIC_DEFAULT_HAIKU_MODEL` | {model} |
 * | `CLAUDE_CODE_SUBAGENT_MODEL` | {model} |
 * @param localPort - Local port of the SSH tunnel to the vLLM server
 * @param model - Model name to use for all Claude model tiers
 * @returns Claude Code environment variable record
 */
export function generateClaudeEnv(
  localPort: number,
  model: string,
): Record<string, string> {
  return {
    ANTHROPIC_BASE_URL: `http://localhost:${localPort}`,
    ANTHROPIC_API_KEY: 'ollama',
    ANTHROPIC_DEFAULT_SONNET_MODEL: model,
    ANTHROPIC_DEFAULT_OPUS_MODEL: model,
    ANTHROPIC_DEFAULT_HAIKU_MODEL: model,
    CLAUDE_CODE_SUBAGENT_MODEL: model,
  };
}

/**
 * Generate the environment variables required to launch the given assistant
 * against the vLLM inference endpoint.
 *
 * Selects the appropriate env-var schema (`OPENAI_*`, `ANTHROPIC_*`,
 * `COPILOT_*`, or `PI_*`) based on the assistant name.
 * @param assistant - Target assistant binary name
 * @param opts - Endpoint and model configuration
 * @returns Environment variable map ready for `spawn`/`spawnSync` (not `env`)
 */
export function generateAssistantEnv(
  assistant: AssistantName,
  opts: AssistantEnvOptions,
): Record<string, string> {
  const endpointHost = opts.endpointHost ?? 'localhost';

  switch (assistant) {
    case 'opencode':
      return generateOpencodeEnv({ ...opts, endpointHost });
    case 'copilot':
      return {
        ...generateCopilotEnv(opts.localPort, opts.model),
        COPILOT_PROVIDER_BASE_URL: `http://${endpointHost}:${opts.localPort}/v1`,
      };
    case 'claude':
      return {
        ...generateClaudeEnv(opts.localPort, opts.model),
        ANTHROPIC_BASE_URL: `http://${endpointHost}:${opts.localPort}`,
      };
    case 'pi':
      // Pi assistant works via configuration files, not environment variables
      return {};
  }
}

/**
 * Build an {@link sbx} sandbox name from the assistant type and working directory.
 *
 * The sandbox name is formatted as `{assistant}-{workspace_basename}` where
 * the workspace basename is sanitized to lowercase alphanumeric characters.
 *
 * **Naming convention**
 *
 * | Component | Source |
 * |-----------|--------|
 * | `assistant` | The assistant name, defaults to `'opencode'` if undefined |
 * | `workspace_basename` | `basename(cwd)` sanitized to lowercase alphanumeric |
 *
 * **Examples**
 *
 * ```ts
 * buildSandboxName('opencode', '/home/user/my-project');
 * // → 'opencode-my-project'
 *
 * buildSandboxName(undefined, '/home/user/docs');
 * // → 'opencode-docs' (uses default 'opencode')
 * ```
 * @param assistant - Target assistant name, or `undefined` for the `'opencode'` default
 * @param cwd - Working directory whose basename is used in the sandbox name
 * @returns A lowercase sandbox name with hyphens and alphanumeric characters only
 */
export function buildSandboxName(
  assistant: AssistantName | undefined,
  cwd: string,
): string {
  const assistantName = assistant ?? 'opencode';
  const workspace = basename(cwd)
    .replace(/[^A-Za-z0-9.+-]+/g, '-')
    .replace(/^-+|-+$/g, '');
  return `${assistantName}-${workspace || 'workspace'}`;
}

/**
 * Build an {@link sbx} sandbox creation command.
 *
 * Runs `sbx create --name` with the assistant name and working directory,
 * using {@link shellQuote} to safely escape arguments.
 *
 * **Command structure**
 *
 * ```bash
 * sbx create --name <sandboxName> <assistant> <cwd>
 * ```
 *
 * If `sandboxName` is not provided, the name is generated from the
 * assistant and working directory using {@link buildSandboxName}.
 *
 * **Examples**
 *
 * ```ts
 * buildSandboxCreateCommand('opencode', '/home/user/my-project', 'opencode-my-project');
 * // → "sbx create --name 'opencode-my-project' opencode '/home/user/my-project'"
 * ```
 * @param assistant - Target assistant name (e.g. `'opencode'`, `'claude'`)
 * @param cwd - Working directory (quoted for shell safety)
 * @param sandboxName - Optional explicit sandbox name; defaults to {@link buildSandboxName}
 * @returns The shell command string for creating the sandbox
 */
export function buildSandboxCreateCommand(
  assistant: AssistantName,
  cwd: string,
  sandboxName?: string,
): string {
  const name = sandboxName ?? buildSandboxName(assistant, cwd);
  return `sbx create --name ${shellQuote(name)} ${assistant} ${shellQuote(cwd)}`;
}

/**
 * Shell-escape a single string value by wrapping it in single quotes.
 *
 * Handles embedded single quotes by ending the current quote, inserting
 * an escaped single quote ('\'''), and restarting the quote — the
 * standard POSIX sh technique for quoting literals with apostrophes.
 * @param value - String to shell-escape
 * @returns The value wrapped in single quotes with internal quotes escaped
 */
function shellQuote(value: string): string {
  return `'${value.replace(/'/g, "'\\''")}'`;
}

/**
 * Format an environment variable record as a space-separated string of
 * KEY=VALUE pairs, with each value shell-escaped via {@link shellQuote}.
 * @param env - Environment variable record
 * @returns Space-separated KEY='value' string
 */
function formatInlineEnv(env: Record<string, string>): string {
  return Object.entries(env ?? {})
    .map(([key, value]) => `${key}=${shellQuote(value)}`)
    .join(' ');
}

/**
 * Return assistant-specific CLI arguments.
 *
 * Currently only GitHub Copilot requires extra arguments ('--continue');
 * all other assistants use an empty array.
 * @param assistant - Target assistant name
 * @returns Array of CLI arguments (empty for all except Copilot)
 */
function getAssistantArgs(assistant: AssistantName): string[] {
  return assistant === 'copilot' ? ['--continue'] : [];
}

/**
 * Build a complete shell command to launch an AI assistant connected to
 * the local vLLM server.
 *
 * The command varies based on the {@link LaunchWrapper}:
 *
 * - **`direct`**: cd WORKDIR && ENV ASSISTANT [args]
 * - **`scoder`**: cd WORKDIR && ENV scoder --llm-port PORT ASSISTANT [args]
 * - **`sbx`**: sbx exec -it -w WORKDIR -e ENV SANDBOX ASSISTANT [args]
 * @param opts - Launch options
 * @param opts.assistant - Target assistant (e.g. 'opencode', 'claude')
 * @param opts.wrapper - Execution wrapper ('direct', 'scoder', or 'sbx')
 * @param opts.cwd - Working directory for the assistant
 * @param opts.localPort - Local port of the vLLM server
 * @param opts.model - Model name to use
 * @param opts.sandboxName - Optional explicit sandbox name (for sbx wrapper)
 * @param opts.maxModelLen - Optional maximum context length
 * @param opts.toolCall - Optional enable tool calling
 * @param opts.reasoning - Optional enable reasoning
 * @returns A shell-ready command string
 */
export function buildLaunchCommand(opts: LaunchCommandOptions): string {
  const endpointHost =
    opts.wrapper === 'sbx' ? 'host.docker.internal' : 'localhost';
  const env = generateAssistantEnv(opts.assistant, { ...opts, endpointHost });
  const agentCommand = [
    opts.assistant,
    ...getAssistantArgs(opts.assistant),
  ].join(' ');

  if (opts.wrapper === 'sbx') {
    const sandboxName =
      opts.sandboxName ?? buildSandboxName(opts.assistant, opts.cwd);
    const envFlags = Object.entries(env)
      .map(([key, value]) => `-e ${shellQuote(`${key}=${value}`)}`)
      .join(' ');
    return `sbx exec -it -w ${shellQuote(opts.cwd)} ${envFlags} ${sandboxName} ${agentCommand}`.trim();
  }

  const envPrefix = formatInlineEnv(env);
  const base =
    opts.wrapper === 'scoder'
      ? `scoder --llm-port ${opts.localPort} ${agentCommand}`
      : agentCommand;
  return `cd ${shellQuote(opts.cwd)} && ${envPrefix} ${base}`.trim();
}

/**
 * Parse the JSON output of sbx ls --json into typed {@link SbxSandbox}
 * records, filtering out malformed entries.
 *
 * Validates that each entry has the required name, agent, state, and cwd
 * properties. Entries missing any of these fields are silently excluded.
 * @param raw - Raw JSON string from sbx ls --json, or empty string
 * @returns Array of validated SbxSandbox records (empty array on parse failure)
 */
export function parseSbxSandboxes(raw: string): SbxSandbox[] {
  if (!raw.trim()) return [];

  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch {
    return [];
  }

  const rows: unknown[] = Array.isArray(parsed)
    ? parsed
    : typeof parsed === 'object' &&
        parsed !== null &&
        Array.isArray((parsed as Record<string, unknown>)['sandboxes'])
      ? ((parsed as Record<string, unknown>)['sandboxes'] as unknown[])
      : [];

  return rows
    .map((row) => {
      const record = row as Record<string, unknown>;
      const name = String(
        record['name'] ??
          record['Name'] ??
          record['sandbox'] ??
          record['SANDBOX'] ??
          '',
      );
      const agent = String(
        record['agent'] ?? record['Agent'] ?? record['AGENT'] ?? '',
      );
      const workspace = String(
        record['workspace'] ?? record['Workspace'] ?? record['WORKSPACE'] ?? '',
      );
      const status = String(
        record['status'] ?? record['Status'] ?? record['STATUS'] ?? '',
      );

      if (!name || !agent || !workspace) return null;
      return {
        name,
        agent,
        workspace,
        status: status || undefined,
      } as SbxSandbox;
    })
    .filter((row): row is SbxSandbox => row !== null);
}

/**
 * Run sbx ls --json and parse the output into {@link SbxSandbox}
 * records.
 *
 * This is a convenience wrapper around {@link parseSbxSandboxes} that
 * executes the sbx CLI and handles the JSON parsing in one step.
 * @returns Array of parsed SbxSandbox records
 */
function listSbxSandboxes(): SbxSandbox[] {
  const result = spawnSync('sbx', ['ls', '--json'], {
    encoding: 'utf-8',
  });
  if (result.status !== 0) {
    throw new Error(result.stderr?.trim() || 'sbx ls --json failed');
  }
  return parseSbxSandboxes(result.stdout);
}

/**
 * Find an existing {@link SbxSandbox} matching the given assistant and
 * working directory.
 *
 * A sandbox is considered a match if it has the same agent, cwd, and
 * is in either the 'running' or 'paused' state.
 * @param sandboxes - Array of sandbox records
 * @param assistant - Target assistant name to match
 * @param cwd - Working directory to match
 * @returns The first matching sandbox, or null if none found
 */
export function findMatchingSandbox(
  sandboxes: SbxSandbox[],
  assistant: AssistantName,
  cwd: string,
): SbxSandbox | null {
  // Normalize the input cwd for comparison
  const normalizedCwd = cwd.trim();

  return (
    sandboxes.find((sandbox) => {
      // Normalize the sandbox fields for comparison
      const normalizedAgent = sandbox.agent?.toLowerCase() ?? '';
      const normalizedWorkspace = sandbox.workspace?.trim() ?? '';

      // Check for exact match first
      if (
        normalizedAgent === assistant.toLowerCase() &&
        normalizedWorkspace === normalizedCwd
      ) {
        return true;
      }

      // Check if workspace matches when ignoring trailing slashes
      const workspaceWithoutTrailingSlash = normalizedWorkspace.replace(
        /\/+$/,
        '',
      );
      const cwdWithoutTrailingSlash = normalizedCwd.replace(/\/+$/, '');
      if (
        normalizedAgent === assistant.toLowerCase() &&
        workspaceWithoutTrailingSlash === cwdWithoutTrailingSlash
      ) {
        return true;
      }

      return false;
    }) ?? null
  );
}

/**
 * Ensure an {@link sbx} sandbox exists for the given assistant and working
 * directory.
 *
 * If a matching sandbox already exists (via {@link findMatchingSandbox}),
 * returns its name. Otherwise creates a new sandbox via {@link sbx} CLI
 * and returns the sandbox name.
 *
 * **Workflow**
 *
 * 1. List existing sandboxes via {@link listSbxSandboxes}
 * 2. Search for a match via {@link findMatchingSandbox}
 * 3. If no match, create a new sandbox using `sbx create --name` command
 * @param assistant - Target assistant name (e.g. 'opencode', 'claude')
 * @param cwd - Working directory to match
 * @returns An object with the sandbox name and a `created` flag indicating
 *   whether a new sandbox was created
 */
export function ensureSbxSandbox(
  assistant: AssistantName,
  cwd: string,
): { sandboxName: string; created: boolean } {
  const sandboxes = listSbxSandboxes();
  const existing = findMatchingSandbox(sandboxes, assistant, cwd);
  if (existing) {
    return { sandboxName: existing.name, created: false };
  }

  const sandboxName = buildSandboxName(assistant, cwd);
  const result = spawnSync(
    'sbx',
    ['create', '--name', sandboxName, assistant, cwd],
    {
      encoding: 'utf-8',
      stdio: 'inherit',
    },
  );
  if (result.status !== 0) {
    throw new Error(`sbx create failed for ${sandboxName}`);
  }
  return { sandboxName, created: true };
}

/**
 * Build the list of menu options based on available assistants and scoder.
 *
 * For each assistant produces a direct launch option and, when scoder
 * is available, a scoder-wrapped option.
 * @param assistants - Assistant names to include in the menu
 * @param hasScoder - Whether the scoder launcher is available
 * @returns Ordered array of {@link AssistantOption} entries with incrementing IDs
 */
export function buildAssistantMenuOptions(
  assistants: string[],
  hasScoder: boolean,
): AssistantOption[] {
  const options: AssistantOption[] = [];
  let id = 1;

  for (const assistant of assistants) {
    // Direct launch option
    options.push({ id, label: `${assistant}`, assistant, useScoder: false });
    id++;

    // Scoder option (if available)
    if (hasScoder) {
      options.push({
        id,
        label: `scoder ${assistant}`,
        assistant,
        useScoder: true,
      });
      id++;
    }
  }

  return options;
}

/**
 * Get the binary name and args for launching an assistant.
 *
 * Returns `{ binary: 'scoder', args: [...] }` when `useScoder` is `true`
 * so the assistant runs inside the scoder sandbox; otherwise returns
 * the assistant binary name directly.
 * @param assistant - Target assistant name
 * @param useScoder - Whether to wrap the command in the scoder launcher
 * @returns Object containing the binary path and its arguments
 */
export function getLaunchCommand(
  assistant: AssistantName,
  useScoder: boolean,
): { binary: string; args: string[] } {
  if (useScoder) {
    return {
      binary: 'scoder',
      args: [assistant, ...getAssistantArgs(assistant)],
    };
  }
  return { binary: assistant, args: getAssistantArgs(assistant) };
}

export interface OpencodeSnippetOptions {
  model: string;
  localPort: number;
  maxModelLen?: number;
  toolCall?: boolean;
  reasoning?: boolean;
}

/**
 * Generate an opencode provider configuration snippet for use
 * in opencode.json.
 *
 * Creates a JSON configuration object that defines an isambard-vllm
 * provider with the configured model, base URL, and model display name.
 * @param opts - Configuration options
 * @param opts.model - HuggingFace model name (e.g. 'Qwen/Qwen2.5-7B-Instruct')
 * @param opts.localPort - Local port of the vLLM server (default 11434)
 * @param opts.maxModelLen - Maximum context length (default 4096)
 * @param opts.outputLimit - Maximum output length (defaults to maxModelLen)
 * @param opts.toolCall - Enable tool calling (default true)
 * @param opts.reasoning - Enable reasoning (default true)
 * @returns JSON string ready for insertion into opencode.json
 */
export function formatOpencodeSnippet(opts: OpencodeSnippetOptions): string {
  const { model, localPort, maxModelLen, toolCall, reasoning } = opts;
  const context = maxModelLen ?? 4096;

  const modelEntry: Record<string, unknown> = {
    name: `${model} (Isambard)`,
    limit: { context, output: context },
  };
  if (toolCall) modelEntry['tool_call'] = true;
  if (reasoning) modelEntry['reasoning'] = true;

  const snippet = {
    $schema: 'https://opencode.ai/config.json',
    model: `isambard-vllm/${model}`,
    provider: {
      'isambard-vllm': {
        npm: '@ai-sdk/openai-compatible',
        name: 'Isambard vLLM Server',
        options: {
          baseURL: `http://localhost:${localPort}/v1`,
          apiKey: 'EMPTY',
        },
        models: {
          [model]: modelEntry,
        },
      },
    },
  };

  return JSON.stringify(snippet, null, 2);
}

