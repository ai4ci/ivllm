import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'fs';
import { homedir } from 'os';
import { join } from 'path';
import type { Credentials } from './types.ts';

// =================================
// CONSTANTS (internal/private)
// =================================

/**
 * Local directory path where the ivllm application config is stored.
 *
 * Equivalent to `~/.config/ivllm`. All project configs (credentials,
 * job YAML files) live under this directory.
 * @see saveConfig
 * @see loadCredentials
 */
const CONFIG_DIR = join(homedir(), '.config', 'ivllm');

/**
 * Absolute path to the JSON file containing SSH/HPC credentials.
 *
 * Located at `{@link CONFIG_DIR}/config.json`. Written by
 * {@link saveConfig} and read by {@link loadCredentials}.
 * @see CONFIG_DIR
 * @see loadCredentials
 * @see saveConfig
 */
const CONFIG_PATH = join(CONFIG_DIR, 'config.json');

/**
 * Default credential values used when no local config file exists.
 *
 * | Field | Default Value |
 * |-------|---------------|
 * | `loginHost` | `''` (empty — will fail {@link assertConfigured}) |
 * | `username` | `''` (empty — will fail {@link assertConfigured}) |
 * | `projectDir` | `''` (empty — will fail {@link assertConfigured}) |
 * | `defaultLocalPort` | `11434` (default OpenAI-compatible port) |
 * @see loadCredentials
 */
const DEFAULTS: Credentials = {
    loginHost: '',
    username: '',
    projectDir: '',
};

// =================================
// EXPORTED FUNCTIONS
// =================================

/**
 * Loads `{@link Credentials}` from the local config file (`config.json`).
 *
 * If the config file does not exist, returns `{@link DEFAULTS}` — a partial
 * object with empty `loginHost` and `username`, and the default local port of
 * `11434`.
 *
 * When the file exists, merges stored values over the defaults so that only
 * explicitly configured fields override the base values.
 * @returns Parsed credentials (defaults if no config file exists)
 * @see DEFAULTS
 * @see saveConfig
 */
export function loadCredentials(): Credentials {
    if (!existsSync(CONFIG_PATH)) {
        return { ...DEFAULTS };
    }
    const raw = readFileSync(CONFIG_PATH, 'utf-8');
    return { ...DEFAULTS, ...JSON.parse(raw) } as Credentials;
}

/**
 * Persists `{@link Credentials}` to the local `config.json` file.
 *
 * Creates {@link CONFIG_DIR} if it does not exist, then writes the
 * credentials as formatted JSON to {@link CONFIG_PATH}.
 * @param config - Credentials object to persist (typically from
 *                 {@link loadCredentials} after interactive setup)
 * @see loadCredentials
 * @see CONFIG_PATH
 * @see CONFIG_DIR
 */
export function saveConfig(config: Credentials): void {
    if (!existsSync(CONFIG_DIR)) {
        mkdirSync(CONFIG_DIR, { recursive: true });
    }
    writeFileSync(CONFIG_PATH, JSON.stringify(config, null, 2) + '\n', 'utf-8');
}

/**
 * Validates that the given `{@link Credentials}` have the required fields
 * for establishing an HPC connection.
 *
 * Checks `loginHost` and `username` — both must be non-empty strings.
 * Throws a descriptive `{@link Error}` with instructions to run the
 * `ivllm config` command if a field is missing.
 * @param config - Credentials to validate (see {@link Credentials})
 * @throws {Error} If `loginHost` is missing, suggests `ivllm config --login-host <host>`
 * @throws {Error} If `username` is missing, suggests `ivllm config --username <user>`
 * @see loadCredentials
 * @see saveConfig
 */
export function assertConfigured(config: Credentials): void {
    if (!config.loginHost) {
        throw new Error(
            'loginHost not configured. Run: ivllm config --login-host <host>',
        );
    }
    if (!config.username) {
        throw new Error(
            'username not configured. Run: ivllm config --username <user>',
        );
    }
    if (!config.projectDir) {
        throw new Error(
            'project dir not configured. Run: ivllm config --project-dir <project>',
        );
    }
}
