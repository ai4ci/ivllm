import type {
    Credentials,
    RunRemoteOptions,
    RunRemoteResult,
    CloseableEventEmitter,
    EnvVarEntry,
} from '../types';
import { RemoteOps } from './RemoteOps';
import { spawn } from 'child_process';
import { EventEmitter } from 'events';
import fs from 'fs';
import path from 'path';
import net from 'net';

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
            let lineBuffer = '';

            proc.stdout?.on('data', (chunk: Buffer) => {
                const text = chunk.toString();
                stdout += text;

                if (options.silent) return;

                lineBuffer += text;
                const lines = lineBuffer.split('\n');
                lineBuffer = lines.pop() ?? ''; // keep the trailing partial line buffered
                for (const line of lines) console.log(line);
            });

            proc.on('error', reject);
            proc.on('close', (code) => {
                if (!options.silent && lineBuffer.length > 0) {
                    console.log(lineBuffer); // flush any unterminated trailing line
                }
                resolve({ exitCode: code ?? 0, stdout: stdout.trim() });
            });
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
            const target = `${this.config.username}@${this.config.loginHost}`;
            const remoteDir = path.dirname(remotePath);
            const remoteCommand = `umask 002 && mkdir -p "${remoteDir}" && cat > "${remotePath}"`;
            const proc = spawn(
                'ssh',
                [...SSH_MUX_OPTS, '-o', 'BatchMode=yes', target, remoteCommand],
                { stdio: ['pipe', 'inherit', 'inherit'] },
            );
            fs.createReadStream(localPath).pipe(proc.stdin);
            proc.on('error', reject);
            proc.on('close', (code) => {
                if (code === 0) resolve();
                else reject(new Error(`scp-via-cat exited with code ${code}`));
            });
        });
    }

    /**
     * Copy a local directory to/from a path on the Isambard login node via rsync.
     *
     * Uses archive mode (-a) to preserve permissions/timestamps, compresses data (-z),
     * and forwards SSH multiplexing via the '-e' flag.
     *
     * On 'up' transfers `--no-perms` + `--rsync-path 'umask 002 && rsync'`
     * alone is enough for newly-created directories (a fresh directory gets
     * a default 0777 request, which the umask correctly masks to 0775). rsync
     * moves then copies so permissions of files end up correctly group writeable.
     * Appending `/` to source and destination paths allow for copy of directory
     * contents like rsync does natively otherwise 'src_a/src_b' -> 'target-a/target_b' on
     * gets copied as 'target_a/target_b/src_b'
     * @param localPath - Path to the local directory
     * @param remotePath - Destination or source path on the login node
     * @param direction - 'up' to upload (local -> remote), 'down' to download (remote -> local)
     * @throws {Error} with the rsync exit code if the transfer fails, or if the follow-up chmod fails
     */
    async copyDirectory(
        localPath: string,
        remotePath: string,
        direction: 'up' | 'down',
    ): Promise<void> {
        const remoteTarget = `${this.config.username}@${this.config.loginHost}:${remotePath}`;

        // Determine source and destination arguments based on direction
        const src = direction === 'up' ? localPath : remoteTarget;
        const dest = direction === 'up' ? remoteTarget : localPath;

        // Construct the custom SSH command string using the multiplexing options
        const sshCommand = `ssh ${[...SSH_MUX_OPTS, '-o', 'BatchMode=yes'].join(' ')}`;

        if (direction === 'up') {
            await this.runRemote(`umask 0002 && mkdir -p "${remotePath}"`);
        } else {
            fs.mkdirSync(dest, { recursive: true });
        }

        await new Promise<void>((resolve, reject) => {
            // Rsync must not change remote permissions and must provide a
            // umask to allow group writable settings on the target directory
            const proc = spawn(
                'rsync',
                [
                    '-a',
                    '--no-perms',
                    '--no-owner',
                    '--no-group',
                    '-z',
                    '--rsync-path',
                    `umask 0002 && rsync`, // umask in front of rsync
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

        // This causes an unpleasant hang on isambard due to the large number of small
        // files in the engine directory... For this to work we woudl need to
        // move the script directory to somewhere else. - NOW DONE but still
        // this turns out to be unnecessary
        // Thsi was added to fix a locally failing test but in fact e2e testing
        // shows it is not needed on isambard due to the umask most likely.
        // if (direction === 'up') {
        //     const { exitCode, stdout } = await this.runRemote(
        //         `chmod -R g+rwX "${remotePath}"`,
        //         { env: [], silent: true },
        //     );
        //     if (exitCode !== 0) {
        //         throw new Error(
        //             `chmod -R g+rwX failed on ${remotePath} (exit ${exitCode}): ${stdout}`,
        //         );
        //     }
        // }
    }

    // TODO: Rethink login here between runRemote, and runRemoteSync
    // Currently runRemote is a fire and forget job launcher as well as a
    // background capture output and return it function. These work because when
    // we are interested in the output and caputure it the comands we issued are
    // quick
    // runRemoteSync return a process handler and will emit output to the
    // console, and now hopefully will terminate remote job when it is terminated
    // however the behaviour of these two things is not clear.

    /**
     * Execute a command on the login node via SSH and stream output.
     *
     * This directly streams output to terminal (does not capture) and
     * remote command dies with local process.
     *
     * Spawns ssh with batch mode and multiplexing via {@link SSH_MUX_OPTS}.
     * Stdout and stderr are forwarded directly to the local terminal.
     * @param command — Shell command to execute
     * @param env — Environment variables to prefix the command
     * @returns A process that can be terminated with `.kill()`
     */
    runRemoteSync(command: string, env: EnvVarEntry[]): CloseableEventEmitter {
        const target = `${this.config.username}@${this.config.loginHost}`;
        // const killableCommand = `trap "kill -TERM -\$\$" EXIT; ${command}`;
        // const fullCommand = this.makeFullCommand(killableCommand, env);
        const fullCommand = this.makeFullCommand(command, env);

        const proc = spawn(
            'ssh',
            [
                ...SSH_MUX_OPTS,
                '-tt',
                '-o',
                'BatchMode=yes',
                target,
                fullCommand,
            ],
            {
                stdio: ['ignore', 'inherit', 'inherit'],
                detached: false,
            },
        );

        return Object.assign(proc, {
            isAlive: async () => proc.exitCode === null && !proc.killed,
            close: () =>
                new Promise<void>((resolve) => {
                    if (proc.exitCode !== null) {
                        resolve();
                        return;
                    }
                    proc.once('close', () => resolve());
                    proc.kill();
                }),
        });
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
        const forwardSpec = `${localPort}:${remoteHost}:${remotePort}`;
        const emitter = new EventEmitter() as unknown as CloseableEventEmitter;

        let closed = false;
        let pollTimer: ReturnType<typeof setInterval> | undefined;

        const runControl = (args: string[]) =>
            new Promise<number>((resolve) => {
                const p = spawn(
                    'ssh',
                    [...SSH_MUX_OPTS, '-o', 'BatchMode=yes', ...args, target],
                    { stdio: 'ignore' },
                );
                p.on('close', (code) => resolve(code ?? 1));
                p.on('error', () => resolve(1));
            });

        // Registers the forward on the existing multiplexed master. Kicked off
        // immediately; every call below awaits this instead of re-issuing it.
        const registered = runControl(['-O', 'forward', '-L', forwardSpec]);

        const checkPortListening = () =>
            new Promise<boolean>((resolve) => {
                const sock = net.createConnection({
                    port: localPort,
                    host: '127.0.0.1',
                });
                sock.once('connect', () => {
                    sock.destroy();
                    resolve(true);
                });
                sock.once('error', () => resolve(false));
                sock.setTimeout(2000, () => {
                    sock.destroy();
                    resolve(false);
                });
            });

        const markClosed = () => {
            if (closed) return;
            closed = true;
            if (pollTimer) clearInterval(pollTimer);
            emitter.emit('close');
        };

        emitter.isAlive = async () => {
            if ((await registered) !== 0) return false;
            return checkPortListening();
        };

        emitter.close = async () => {
            if (closed) return;
            await registered;
            await runControl(['-O', 'cancel', '-L', forwardSpec]);
            markClosed();
        };

        registered.then((code) => {
            if (code !== 0) {
                emitter.emit(
                    'error',
                    new Error(`Failed to register SSH forward (exit ${code})`),
                );
                markClosed();
                return;
            }

            // Poll independently of any process handle — catches the master
            // dying, or the forward being cancelled out from under us.
            pollTimer = setInterval(async () => {
                if (closed) return;
                if (!(await checkPortListening())) markClosed();
            }, 5000);
        });

        return emitter;
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
                if (rawError.includes('keyboard-interactive')) {
                    console.error(
                        '💡 Hint: You need to cache your credentials with ssh-agent so that you can login to the server without authentication',
                        '(e.g. ssh-add ~/.ssh/id_ed25519)',
                    );
                } else if (
                    rawError.includes('401') ||
                    rawError.includes('invalid_grant') ||
                    rawError.includes('token') ||
                    code == 255
                ) {
                    console.error(
                        '💡 Hint: Check your SSH keys. Do you need to run keycloak authentication',
                        '(e.g. clifton auth)?',
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
