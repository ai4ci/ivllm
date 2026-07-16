import type {
  RunRemoteOptions,
  RunRemoteResult,
  CloseableEventEmitter,
  EnvVarEntry
} from '../types';

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
export abstract class RemoteOps {
  /**
   * Execute a command on the login node via SSH.
   * @param command - Shell command to run
   * @param options - Execution options (env, silent)
   * @returns Promise resolving to exit code and stdout
   */
  abstract runRemote(
    command: string,
    options?: RunRemoteOptions,
  ): Promise<RunRemoteResult>;

  /**
   * Copy a local file to the login node via SCP.
   * @param localPath - Path to the local source file
   * @param remotePath - Destination path on the login node
   */
  abstract copyFile(localPath: string, remotePath: string): Promise<void>;

  /**
   * Continuously tail a remote file and stream lines to stdout.
   * @param remotePath - Absolute path to the remote log file
   * @param prefix - Optional string prepended to every output line
   * @returns An object with a `stop()` method to close the connection
   */
  abstract tailRemoteLog(
    remotePath: string,
    prefix?: string,
  ): { stop: () => void };

  /**
   * Spawn a persistent forward SSH tunnel (localPort → remoteHost:remotePort).
   * @param localPort - Port to listen on locally
   * @param remoteHost - Remote host (typically a compute node)
   * @param remotePort - Remote port (e.g. vLLM server port)
   * @returns Closeable event emitter representing the tunnel process
   */
  abstract spawnTunnel(
    localPort: number,
    remoteHost: string,
    remotePort: number,
  ): CloseableEventEmitter;

  /**
   * Find the best installed vLLM version meeting a minimum requirement.
   * @param minVllmVersion - Minimum acceptable version string
   * @returns The selected version string
   */
  abstract matchVllmVersion(minVllmVersion: string): Promise<string>;

  /**
   * Verify SSH connectivity to the login node.
   * @returns Promise resolving to true when connectivity is confirmed
   */
  abstract checkSSH(): Promise<boolean>;

  /**
   * Prepend environment variable definitions to a shell command.
   *
   * Joins all {@link EnvVarEntry} objects as KEY=VALUE prefixes separated
   * by spaces, then prepends them to the raw command string.
   *
   * makeFullCommand('ls -la', [{ key: 'HF_HOME', value: '/tmp/hf' }]);
   * // → 'HF_HOME=/tmp/hf ls -la'
   * @param command - The shell command to execute
   * @param env - Environment variables to prepend
   * @returns The command string with env vars as prefix
   */
  protected makeFullCommand(command: string, env: EnvVarEntry[]): string {
      const envPrefix = env.map((v) => `${v.key}=${v.value}`).join(' ') + ' ';
      const fullCommand = (envPrefix + command).trim();
      return fullCommand;
  }
}
