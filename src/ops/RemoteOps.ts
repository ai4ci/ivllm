import type {
    RunRemoteOptions,
    RunRemoteResult,
    CloseableEventEmitter,
    EnvVarEntry,
    LockfileV3,
} from '../types';

/**
 * Abstract interface for executing remote operations on the HPC login node.
 *
 * Implemented by {@link SshRemoteOps} for the Isambard backend.
 * Provides SSH exec, SCP transfer, tunnel spawning, and sync command support.
 *
 * | Method | Description |
 * |--------|-------------|
 * | `runRemote` | Execute a command on the login node via SSH |
 * | `runRemoteSync` | Execute a command and stream output |
 * | `copyFile` | SCP a file to the login node |
 * | `copyDirectory` | Rsync a directory up or down |
 * | `spawnTunnel` | Create an SSH port-forwarding tunnel |
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
     * Copy a local directory to/from a remote pathvia rsync.
     *
     * Uses archive mode (-a) to preserve permissions/timestamps, compresses data (-z),
     * and forwards SSH multiplexing via the '-e' flag.
     * @param localPath - Path to the local directory
     * @param remotePath - Destination or source path on the login node
     * @param direction - 'up' to upload (local -> remote), 'down' to download (remote -> local)
     * @throws Error with the rsync exit code if the transfer fails
     */
    abstract copyDirectory(
        localPath: string,
        remotePath: string,
        direction: 'up' | 'down',
    ): Promise<void>;

    /**
     * Execute a command on the Isambard login node via SSH and stream output.
     *
     * Spawns ssh with batch mode (BatchMode=yes) for passwordless
     * execution, and SSH multiplexing via {@link SSH_MUX_OPTS}. Output is
     * forwarded to the local terminal.
     *
     * @param config - SSH {@link Credentials}
     * @param env - Environment variables to prefix the command
     * @returns An event emitter that can be closed with `.kill()`
     */
    abstract runRemoteSync(
        command: string,
        env: EnvVarEntry[],
    ): CloseableEventEmitter;

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
