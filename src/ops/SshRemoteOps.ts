import type {
    Credentials,
    RunRemoteOptions,
    RunRemoteResult,
    CloseableEventEmitter,
    EnvVarEntry,
} from '../types';
import { RemoteOps } from './RemoteOps';
import { spawn } from 'child_process';

/**
 * Reusable SSH multiplexing options for SSH/SCP commands.
 *
 * The first connection spawns a background ControlMaster; subsequent
 * connections within 10 minutes (600 s) reuse the existing socket, avoiding
 * repeated handshakes and login rate-limits on busy HPC login nodes.
 *
 * **Options**
 *
 * | Flag | Purpose |
 * |------|---------|
 * | ControlMaster=auto | Reuse an existing master, or create one |
 * | ControlPersist=600 | Keep master alive for 10 minutes after last connection closes |
 * | ControlPath | Per-user socket at /tmp/ivllm-ssh-%r@%h:%p |
 *
 */
const SSH_MUX_OPTS = [
    '-o',
    'ControlMaster=auto',
    '-o',
    'ControlPersist=600',
    '-o',
    'ControlPath=/tmp/ivllm-ssh-%r@%h:%p',
] as const;

export class SshRemoteOps extends RemoteOps {
    private config: Credentials;
    private lastCheckTime = 0;
    private readonly CACHE_TTL_MS = 5000; // Skip system execution if checked in the last 5 seconds

    constructor(config: Credentials) {
        super();
        // this.sandboxDir = sandboxDir;
        this.config = config;
    }

    /**
     * Execute a command on the login node via SSH and capture stdout.
     *
     * Spawns ssh with batch mode and multiplexing via {@link SSH_MUX_OPTS}.
     * When `options.silent` is `false`, stdout/stderr are forwarded to
     * the local terminal; otherwise output is captured and returned.
     * @param command — Shell command to execute on the login node
     * @param options — Execution options
     * @param options.env — Environment variables to prefix the command
     * @param options.silent — When `false` stream stdout to terminal
     * @returns Exit code and captured stdout
     */
    async runRemote(
        command: string,
        options: RunRemoteOptions = { env: [], silent: true },
    ): Promise<RunRemoteResult> {
        return new Promise((resolve, reject) => {
            const target = `${this.config.username}@${this.config.loginHost}`;
            const fullCommand = this.makeFullCommand(command, options.env);

            const proc = spawn(
                'ssh',
                [...SSH_MUX_OPTS, '-o', 'BatchMode=yes', target, fullCommand],
                {
                    stdio: ['ignore', 'pipe', 'inherit'],
                },
            );

            let stdout = '';

            proc.stdout?.on('data', (chunk: Buffer) => {
                stdout += chunk.toString();
                if (!options.silent) console.log(chunk.toString());
            });

            proc.on('error', reject);
            proc.on('close', (code) =>
                resolve({ exitCode: code ?? 0, stdout: stdout.trim() }),
            );
        });
    }

    /**
     * Copy a local file to the login node via SCP.
     *
     * Spawns scp with batch mode and multiplexing via {@link SSH_MUX_OPTS}.
     * @param localPath — Path to the local source file
     * @param remotePath — Destination path on the login node
     * @throws {Error} with the SCP exit code if the transfer fails
     */
    async copyFile(localPath: string, remotePath: string): Promise<void> {
        return new Promise((resolve, reject) => {
            const target = `${this.config.username}@${this.config.loginHost}:${remotePath}`;
            const proc = spawn(
                'scp',
                [...SSH_MUX_OPTS, '-o', 'BatchMode=yes', localPath, target],
                {
                    stdio: 'inherit',
                },
            );
            proc.on('error', reject);
            proc.on('close', (code) => {
                if (code === 0) resolve();
                else reject(new Error(`scp exited with code ${code}`));
            });
        });
    }

    /**
     * Copy a local directory to/from a path on the Isambard login node via rsync.
     *
     * Uses archive mode (-a) to preserve permissions/timestamps, compresses data (-z),
     * and forwards SSH multiplexing via the '-e' flag.
     * @param localPath - Path to the local directory
     * @param remotePath - Destination or source path on the login node
     * @param direction - 'up' to upload (local -> remote), 'down' to download (remote -> local)
     * @throws {Error} with the rsync exit code if the transfer fails
     */
    async copyDirectory(
        localPath: string,
        remotePath: string,
        direction: 'up' | 'down',
    ): Promise<void> {
        return new Promise((resolve, reject) => {
            const remoteTarget = `${this.config.username}@${this.config.loginHost}:${remotePath}`;

            // Determine source and destination arguments based on direction
            const src = direction === 'up' ? localPath : remoteTarget;
            const dest = direction === 'up' ? remoteTarget : localPath;

            // Construct the custom SSH command string using your multiplexing options
            // e.g., "ssh -o ControlMaster=auto -o BatchMode=yes"
            const sshCommand = `ssh ${[...SSH_MUX_OPTS, '-o', 'BatchMode=yes'].join(' ')}`;

            const proc = spawn(
                'rsync',
                [
                    '-az', // Archive mode (recursive, preserves attributes) + Compression
                    '-e',
                    sshCommand, // Instruct rsync to use our specific multiplexed SSH configuration
                    src,
                    dest,
                ],
                {
                    stdio: 'inherit',
                },
            );

            proc.on('error', reject);
            proc.on('close', (code) => {
                if (code === 0) resolve();
                else reject(new Error(`rsync exited with code ${code}`));
            });
        });
    }

    /**
     * Execute a command on the login node via SSH and stream output.
     *
     * Spawns ssh with batch mode and multiplexing via {@link SSH_MUX_OPTS}.
     * Stdout and stderr are forwarded directly to the local terminal.
     * @param command — Shell command to execute
     * @param env — Environment variables to prefix the command
     * @returns A process that can be terminated with `.kill()`
     */
    runRemoteSync(command: string, env: EnvVarEntry[]): CloseableEventEmitter {
        const target = `${this.config.username}@${this.config.loginHost}`;
        const fullCommand = this.makeFullCommand(command, env);

        const proc = spawn(
            'ssh',
            [...SSH_MUX_OPTS, '-o', 'BatchMode=yes', target, fullCommand],
            {
                stdio: ['ignore', 'inherit', 'inherit'],
                detached: false,
            },
        );

        return proc;
    }

    /**
     * Spawn a persistent SSH port-forwarding tunnel.
     *
     * Maps `localhost:<localPort>` → `<remoteHost>:<remotePort>` through
     * the login node. Uses `ServerAliveInterval`/`ServerAliveCountMax` for
     * keepalive and `ExitOnForwardFailure` to detect bad hostnames.
     * @param localPort — Port to listen on locally
     * @param remoteHost — Remote host (e.g. compute node hostname)
     * @param remotePort — Remote port (e.g. vLLM server port)
     * @returns A process that can be terminated with `.kill()`
     */
    spawnTunnel(
        localPort: number,
        remoteHost: string,
        remotePort: number,
    ): CloseableEventEmitter {
        const target = `${this.config.username}@${this.config.loginHost}`;
        const proc = spawn(
            'ssh',
            [
                ...SSH_MUX_OPTS,
                '-N',
                '-o',
                'BatchMode=yes',
                '-o',
                'ServerAliveInterval=10',
                '-o',
                'ServerAliveCountMax=3',
                '-o',
                'ExitOnForwardFailure=yes',
                '-L',
                `${localPort}:${remoteHost}:${remotePort}`,
                target,
            ],
            { stdio: 'ignore', detached: false },
        );

        return proc;
    }

    /**
     * Verify SSH connectivity to the login node.
     *
     * Runs `echo ok` via {@link runRemote} and checks the exit code.
     * On failure, logs an error and calls `process.exit(1)`.
     * @returns `true` when connectivity is confirmed
     */
    async checkSSH(): Promise<boolean> {
        // console.log('Checking SSH connectivity...');
        // const { exitCode: sshCheck } = await this.runRemote('echo ok');
        // if (sshCheck !== 0) {
        //     console.error('Error: Cannot connect to login node.');
        //     process.exit(1);
        // }
        // console.log('✓ SSH connectivity OK');
        // return true;

        const now = Date.now();
        const target = `${this.config.username}@${this.config.loginHost}`;

        // 1. Return cached check if a query is active or resolved recently
        if (now - this.lastCheckTime < this.CACHE_TTL_MS) {
            return true;
        }

        // 1. Try a fast, silent local multiplexer check first
        const isMuxAlive = await new Promise<boolean>((resolve) => {
            const checkProc = spawn(
                'ssh',
                [...SSH_MUX_OPTS, '-o', 'BatchMode=yes', '-O', 'check', target],
                {
                    stdio: 'ignore',
                    detached: false,
                },
            );
            checkProc.on('close', (code) => resolve(code === 0));
        });

        if (isMuxAlive) {
            console.log('✓ SSH multiplexer is alive and healthy.');
            this.lastCheckTime = Date.now();
            return true;
        }

        // 2. Multiplexer is dead or hasn't started yet. Attempt to establish/refresh it.
        console.log('Initialising master SSH connection...');

        return new Promise<boolean>((resolve) => {
            // We add '-N' (Do not execute a remote command) and '-f' (Go to background after authentication)
            // This causes SSH to establish the Master socket and cleanly daemonize locally.
            const connectArgs = [
                ...SSH_MUX_OPTS,
                '-o',
                'BatchMode=yes',
                '-o',
                'ConnectTimeout=10', // Give it a strict 10s network deadline
                '-N',
                '-f',
                target,
            ];

            // Capture stderr to pull out the exact reason for failure (e.g., Auth denied, DNS failure, Host Unreachable)
            const errorChunks: string[] = [];
            const connProc = spawn('ssh', connectArgs, {
                stdio: ['ignore', 'ignore', 'pipe'],
                detached: false,
            });

            connProc.stderr?.on('data', (chunk) => {
                errorChunks.push(chunk.toString());
            });

            connProc.on('close', (code) => {
                if (code === 0) {
                    console.log(
                        '✓ Successfully established new SSH master connection.',
                    );
                    this.lastCheckTime = Date.now();
                    resolve(true);
                    return;
                }

                // 3. Extraction of Diagnostic Information on failure
                const rawError = errorChunks.join('').trim();
                console.error(
                    '\n❌ CRITICAL: Failed to establish SSH connection to the login node.',
                );
                console.error(`Exit Code from SSH binary: ${code}`);

                if (rawError) {
                    console.error(`System Error Message:\n--> ${rawError}`);
                } else {
                    console.error(
                        'No stderr reported. This usually indicates a silent local configuration issue or immediate network rejection.',
                    );
                }

                // Provide actionable hints based on common SSH outputs
                if (
                    rawError.includes('401') ||
                    rawError.includes('invalid_grant') ||
                    rawError.includes('token')
                ) {
                    console.error(
                        '💡 Hint: Check your SSH keys. Do you need to run keycloak authentication (e.g. clifton auth)?',
                    );
                } else if (rawError.includes('Permission denied')) {
                    console.error(
                        '💡 Hint: Check your SSH keys. BatchMode is active, so password prompts are disabled.',
                    );
                } else if (rawError.includes('Could not resolve hostname')) {
                    console.error(
                        '💡 Hint: DNS resolution failed. Check your cluster address or VPN connectivity.',
                    );
                } else if (rawError.includes('Connection timed out')) {
                    console.error(
                        `💡 Hint: Network timeout. Verify that you are connected to the network and ${this.config.loginHost} is online.`,
                    );
                } else if (rawError.includes('ControlSocket')) {
                    console.error(
                        '💡 Hint: Local filesystem issue. Ensure that /tmp/ is writable and your control path name length is valid.',
                    );
                }

                process.exit(1);
            });
        });
    }
}
