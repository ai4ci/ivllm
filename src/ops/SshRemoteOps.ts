import type {
    Credentials,
    RunRemoteOptions,
    RunRemoteResult,
    CloseableEventEmitter,
    EnvVarEntry,
} from '../types';
import { RemoteOps } from './RemoteOps';
import { spawn } from 'child_process';
import { semverGte, revSemverSort } from '../semver.ts';

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
 * **Note**: streamSrun and spawnTunnel deliberately disable multiplexing
 * (ControlMaster=no) because the master process can interfere with TTY
 * output and tunnel lifecycle detection respectively.
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
     *
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
     *
     * @param localPath — Path to the local source file
     * @param remotePath — Destination path on the login node
     * @throws Error with the SCP exit code if the transfer fails
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
     * @throws Error with the rsync exit code if the transfer fails
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
                    '--info=progress2', // Clean progress bar output for directory syncing
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
     *
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
     *
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
     *
     * @returns `true` when connectivity is confirmed
     */
    async checkSSH(): Promise<boolean> {
        console.log('Checking SSH connectivity...');
        const { exitCode: sshCheck } = await this.runRemote('echo ok');
        if (sshCheck !== 0) {
            console.error('Error: Cannot connect to login node.');
            process.exit(1);
        }
        console.log('✓ SSH connectivity OK');
        return true;
    }
}
