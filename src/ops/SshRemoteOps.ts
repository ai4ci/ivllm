import type {
  Credentials,
  RunRemoteOptions,
  RunRemoteResult,
  CloseableEventEmitter,
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
   * Execute a command on the Isambard login node via SSH and capture its
   * standard output.
   *
   * Spawns ssh with batch mode (BatchMode=yes) for passwordless
   * execution, and SSH multiplexing via {@link SSH_MUX_OPTS}. When
   * options.silent is false, stdout/stderr are forwarded directly to the
   * local terminal; otherwise output is captured and returned.
   *
   * **SSH options**
   *
   * | Option | Purpose |
   * |--------|---------|\n * | ControlMaster=auto + ControlPersist=600 | Multiplexed connection (see {@link SSH_MUX_OPTS}) |
   * | BatchMode=yes | Fail immediately on auth issues (no interactive prompts) |
   * @param config - SSH {@link Credentials}
   * @param command - Shell command to execute on the login node
   * @param options - Execution options
   * @param options.env - Environment variables to prefix the command
   * @param options.silent - When false stream stdout to terminal
   * @returns A promise resolving to the exit code and captured stdout
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
        resolve({ exitCode: code ?? 1, stdout: stdout.trim() }),
      );
    });
  }

  /**
   * Copy a local file to a path on the Isambard login node via SCP.
   *
   * Spawns scp with batch mode and SSH multiplexing via {@link SSH_MUX_OPTS}
   * for efficient bulk transfers.
   * @param config - SSH {@link Credentials}
   * @param localPath - Path to the local source file
   * @param remotePath - Destination path on the login node
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
   * Continuously tail a remote file via SSH and stream each new line to
   * process.stdout with an optional prefix.
   *
   * Spawns ssh tail -n +1 -f <remotePath> using SSH multiplexing ({@link
   * SSH_MUX_OPTS}). Lines are buffered to handle partial writes and emitted
   * one at a time.
   *
   * // Stream setup logs in real time
   * tailer.stop(); // close the SSH connection
   * @param config - SSH {@link Credentials}
   * @param remotePath - Absolute path to the remote log file
   * @param prefix - Optional string prepended to every output line
   * @returns An object with a stop() method to close the tail connection
   */
  tailRemoteLog(remotePath: string, prefix = ''): { stop: () => void } {
    const target = `${this.config.username}@${this.config.loginHost}`;
    const proc = spawn(
      'ssh',
      [
        ...SSH_MUX_OPTS,
        '-o',
        'BatchMode=yes',
        target,
        `tail -n +1 -f ${remotePath} 2>/dev/null`,
      ],
      { stdio: ['ignore', 'pipe', 'ignore'] },
    );

    let buf = '';
    proc.stdout?.on('data', (chunk: Buffer) => {
      buf += chunk.toString();
      const lines = buf.split('\n');
      buf = lines.pop() ?? '';
      for (const line of lines) {
        process.stdout.write(prefix + line + '\n');
      }
    });

    return {
      stop: () => {
        try {
          proc.kill();
        } catch {
          /* ignore */
        }
      },
    };
  }

  /**
   * Spawn a persistent forward SSH tunnel that maps a local port to a
   * remote host:port through the Isambard login node.
   *
   * The tunnel is created with ssh -N -L and **must not use multiplexing**
   * (ControlMaster=no) — multiplexing would cause the master to exit when
   * the last connection closes, triggering a false shutdown in callers.
   *
   * **SSH options**
   *
   * | Flag | Purpose |
   * |------|---------|\n * | -N | No remote command — just forward ports |
   * | ControlMaster=no | Dedicated connection (multiplexing breaks tunnel lifecycle detection) |
   * | BatchMode=yes | No interactive prompts |
   * | ServerAliveInterval=10 + ServerAliveCountMax=3 | Detect dead tunnels after ~30 s |
   * | ExitOnForwardFailure=yes | Fail fast if port forwarding cannot be established |
   *
   * // Forward local:11434 → gh200-1:8000 through the login node
   * @param config - SSH {@link Credentials}
   * @param localPort - Port to listen on locally
   * @param remoteHost - Remote host (typically a compute node, e.g. 'gh200-1')
   * @param remotePort - Remote port (e.g. the vLLM server port 8000)
   * @returns The ChildProcess representing the SSH tunnel (can be killed)
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
   * Discover all vLLM versions installed under '$PROJECT_DIR'/ivllm/.
   *
   * Discovers installed vLLM versions on the login node and parses them
   * output to extract version strings matching '\\d+.\\d+'.
   * @param config - SSH {@link Credentials}
   * @returns An array of installed version strings (e.g. ['0.19.1', '0.22.0'])
   */
  private async listInstalledVersions(): Promise<string[]> {
    const { stdout } = await this.runRemote(
      `ls -d ${this.config.projectDir}/ivllm/\*/bin 2>/dev/null | sed 's|.*/ivllm/||; s|/bin||'`,
    );
    return stdout
      .trim()
      .split('\n')
      .filter((v) => v && /^\d+\.\d+/.test(v));
  }

  /**
   * Select the best installed vLLM version that meets a minimum version
   * requirement.
   *
   * First lists all installed versions via {@link listInstalledVersions},
   * then either picks the highest version that is >= minVllmVersion (if
   * given) or simply returns the highest installed version.
   *
   * **Selection strategy**
   *
   * | Condition | Behaviour |
   * |-----------|-----------|\n * | minVllmVersion provided + candidates exist | Highest version >= minimum |
   * | minVllmVersion provided + no candidates | throw with error listing installed versions |
   * | minVllmVersion falsy + versions exist | Highest installed version |
   * | No versions installed | throw suggesting ivllm setup |
   * @param config - SSH {@link Credentials}
   * @param minVllmVersion - Minimum acceptable version string (e.g. '0.20.0'); falsy means "any version"
   * @returns The selected installed version string
   * @throws If no vLLM installation is found or no version meets the minimum
   */
  async matchVllmVersion(minVllmVersion: string): Promise<string> {
    const installed = await this.listInstalledVersions();
    if (installed.length === 0) {
      throw new Error(
        `No vLLM installation found at ${this.config.projectDir}/ivllm/. Run 'ivllm setup <version>'.`,
      );
    }

    const minVersion = minVllmVersion;
    const bestVersion = minVersion
      ? this.selectBestVersion(installed, minVersion)
      : revSemverSort(installed)[0];

    if (!bestVersion) {
      throw new Error(
        minVersion
          ? `Config requires vLLM >= ${minVersion} but installed versions are: ${installed.join(', ')}`
          : `No suitable vLLM version found.`,
      );
    }

    return bestVersion;
  }

  /**
   * Find the highest installed version that satisfies a minimum version
   * constraint.
   *
   * Filters the installed array using {@link semverGte}, then sorts the
   * candidates descending and returns the first one.
   *
   * selectBestVersion(['0.19.0', '0.20.0', '0.22.0'], '0.21.0');
   * // → '0.22.0'
   *
   * selectBestVersion(['0.18.0', '0.19.1'], '0.22.0');
   * // → null (no candidate meets minimum)
   * @param installed - Array of installed version strings
   * @param minVersion - Minimum acceptable version (e.g. '0.20.0')
   * @returns The best matching version, or null if no candidate qualifies
   */
  private selectBestVersion(
    installed: string[],
    minVersion: string,
  ): string | null {
    const candidates = installed.filter((v) => semverGte(v, minVersion));
    if (candidates.length === 0) return null;
    return revSemverSort(candidates)[0]!;
  }

  /**
   * Verify that an SSH connection to the login node works.
   *
   * Executes echo ok on the login node via {@link runRemote} and checks
   * the exit code. Logs status messages to the console.
   *
   * **Side effects**
   *
   * - On failure: logs Error: Cannot connect to login node. and calls
   *   process.exit(1) (this function does not throw on its own)
   * - On success: logs ✓ SSH connectivity OK
   * @param credentials
   * @returns true when connectivity is confirmed
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
