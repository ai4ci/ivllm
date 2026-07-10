import { copyFileSync, mkdirSync } from 'fs';
import { basename, join } from 'path';
import EventEmitter from 'node:events';
import { spawn } from 'child_process';
import readline from 'readline';
import { semverGte, revSemverSort } from './semver.ts';

import type {
  CloseableEventEmitter,
  Credentials,
  RemoteOps,
  RunRemoteOptions,
  RunRemoteResult,
  EnvVarEntry,
  ProcessState,
} from './types.ts';
import { tmpdir } from 'node:os';

/**
 * Factory that returns a {@link RemoteOps} implementation — either real
 * SSH/SCP-based ops or dry-run mocks for E2E testing.
 *
 * **Real mode** (dryRun: false)
 *
 * Delegates to the five internal helpers (runRemote, copyFile,
 * streamSrun, tailRemoteLog, spawnTunnel) plus matchVllmVersion
 * and checkSSH.
 *
 * **Dry-run mode** (dryRun: true)
 *
 * | Method | Mock behaviour |
 * |--------|----------------|
 * | runRemote | Logs [dry-run] prefix, returns fake stdout based on command type (sbatch → job ID, squeue → test state, etc.) |
 * | copyFile | Copies to '$TMPDIR'/ivllm-dryrun/ and prints source → destination |
 * | streamSrun | Logs command, sets sessionState.slurmJobId = '123456', returns mock emitter |
 * | tailRemoteLog | Logs dummy line, returns { stop: () => {} } |
 * | spawnTunnel | Logs tunnel details, returns mock emitter |
 * | matchVllmVersion | Returns best of ['0.22.0', minVllmVersion] |
 * | checkSSH | Returns true |
 * @param config - SSH {@link Credentials} used to build remote commands
 * @param dryRun - When true return mock implementations for testing
 * @returns An object conforming to the {@link RemoteOps} interface
 */
export function makeRemoteOps(config: Credentials, dryRun: boolean): RemoteOps {
  if (!dryRun) {
    return {
      runRemote: (cmd, opts) => runRemote(config, cmd, opts),
      copyFile: (local, remote) => copyFile(config, local, remote),
      streamSrun: (cmd, sessionState, opts) =>
        streamSrun(config, cmd, opts, sessionState),
      tailRemoteLog: (remote, prefix) => tailRemoteLog(config, remote, prefix),
      spawnTunnel: (localPort, remoteHost, remotePort) =>
        spawnTunnel(config, localPort, remoteHost, remotePort),
      matchVllmVersion: (minVllmVersion) =>
        matchVllmVersion(config, minVllmVersion),
      checkSSH: () => checkSSH(config),
    };
  }

  // Mock all network actions for E2E testing.
  return {
    async runRemote(command: string, opts) {
      const fullCommand = makeFullCommand(command, opts?.env || []);
      console.log(`  [dry-run] Would run remotely:\n    ${fullCommand}`);
      return {
        exitCode: 0,
        stdout: command.startsWith('sbatch')
          ? 'Submitted batch job 123456'
          : command.startsWith('squeue')
            ? 'test-state test-reason'
            : command.startsWith('sacct')
              ? '123456 RUNNING'
              : command.startsWith('cat')
                ? 'lockfile' // needs json example
                : command.startsWith('ls -d')
                  ? '0.99.99' // version info
                  : '',
      };
    },
    async copyFile(localPath, remotePath) {
      const dryRunDir = join(tmpdir(), 'ivllm-dryrun');
      mkdirSync(dryRunDir, { recursive: true });
      const destName = basename(remotePath);
      const dest = join(dryRunDir, destName);
      copyFileSync(localPath, dest);
      console.log(`  [dry-run] Would scp: ${localPath} → ${remotePath}`);
      console.log(`           (preview: ${dest})`);
    },
    streamSrun(command, sessionState, opts) {
      const fullCommand = makeFullCommand(command, opts?.env || []);
      console.log(`  [dry-run] Would stream remotely:\n    ${fullCommand}`);
      console.log(`srun: job 123456`);
      sessionState.slurmJobId = '123456';
      return createMockSSh('streaming srun', 2222);
    },
    tailRemoteLog(remote, prefix) {
      console.log(
        (prefix ?? '') + `  [dry-run] dummy log entry from ${remote}`,
      );
      return { stop: () => {} };
    },
    spawnTunnel(local, remoteHost, remotePort) {
      console.log(
        `  [dry-run] mock SSH tunnel created with ${local}:${remoteHost}:${remotePort}`,
      );

      // 1. Create the main process and its standard I/O stubs as plain EventEmitters
      return createMockSSh('tunnel', 1111);
    },
    async matchVllmVersion(minVllmVersion) {
      const tmp = selectBestVersion(['0.22.0', minVllmVersion], minVllmVersion);
      return Promise.resolve(tmp!);
    },
    async checkSSH() {
      console.log('  [dry-run] skipping SSH check');
      return true;
    },
  };
}

// ======================
// HELPERS
// ======================

/**
 * Create a mock {@link CloseableEventEmitter} that simulates a persistent
 * SSH child process for dry-run mode.
 *
 * The mock is a plain EventEmitter augmented with process-like properties
 * (pid, exitCode, stdin, stdout, stderr) so that code expecting a
 * ChildProcess shape works without modification.
 *
 * **Mock properties**
 *
 * | Property | Value |
 * |----------|-------|
 * | pid | The pid argument |
 * | exitCode | null — stays null so the process appears permanently alive |
 * | stdin | EventEmitter with a no-op write |
 * | stdout.pipe / stderr.pipe | Identity functions (return destination) |
 * | kill() | Emits 'close', 0 and returns true |
 * @param name - Label for console messages during kill()
 * @param pid - Simulated process ID
 * @returns A mock CloseableEventEmitter usable in place of a ChildProcess
 */
function createMockSSh(name: string, pid: number): CloseableEventEmitter {
  const mockSshTunnel = Object.assign(new EventEmitter(), {
    stdin: Object.assign(new EventEmitter(), { write: () => true }),
    stdout: Object.assign(new EventEmitter(), {
      pipe: (destination: EventEmitter) => destination,
    }),
    stderr: Object.assign(new EventEmitter(), {
      pipe: (destination: EventEmitter) => destination,
    }),
    // 2. Add required identity properties so it looks like a running process
    pid: pid,
    exitCode: null, // Stays null so it looks permanently alive
    kill: () => {
      console.log(`  [dry-run] Shutting down ${name}`);
      mockSshTunnel.emit('close', 0);
      return true;
    },
  });
  return mockSshTunnel;
}

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
function makeFullCommand(command: string, env: EnvVarEntry[]): string {
  const envPrefix = env.map((v) => `${v.key}=${v.value}`).join(' ') + ' ';
  const fullCommand = (envPrefix + command).trim();
  return fullCommand;
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
 * @param options.silent - When true capture output; when false stream to terminal
 * @returns A promise resolving to the exit code and captured stdout
 */
function runRemote(
  config: Credentials,
  command: string,
  options: RunRemoteOptions = { env: [], silent: true },
): Promise<RunRemoteResult> {
  return new Promise((resolve, reject) => {
    const target = `${config.username}@${config.loginHost}`;
    const fullCommand = makeFullCommand(command, options.env);

    const proc = spawn(
      'ssh',
      [...SSH_MUX_OPTS, '-o', 'BatchMode=yes', target, fullCommand],
      {
        stdio: options.silent
          ? ['ignore', 'pipe', 'pipe']
          : ['ignore', 'inherit', 'inherit'],
      },
    );

    let stdout = '';
    if (options.silent) {
      proc.stdout?.on('data', (chunk: Buffer) => {
        stdout += chunk.toString();
      });
    }

    proc.on('error', reject);
    proc.on('close', (code) =>
      resolve({ exitCode: code ?? 1, stdout: stdout.trim() }),
    );
  });
}

/**
 * Execute an srun command on the login node and stream its output to the
 * local terminal while parsing the stderr for a Slurm job ID.
 *
 * Unlike {@link runRemote}, this function uses a pseudo-TTY (-t) to
 * bypass remote log buffering, and **does not use SSH multiplexing**
 * (ControlMaster=no) so that the job ID can be detected in the output
 * stream.
 *
 * **Output handling**
 *
 * - stdout → forwarded to process.stdout (unless silent: true)
 * - stderr → parsed line-by-line via readline; if a line starts with
 *   "srun job " after sanitisation, the job ID is extracted and stored
 *   in sessionState.slurmJobId
 *
 * // Interactive srun — output streams to terminal
 * @param config - SSH {@link Credentials}
 * @param command - srun command to execute on the login node
 * @param options - Execution options (default silent: false)
 * @param sessionState - Process state whose slurmJobId is updated on detection
 * @returns A CloseableEventEmitter that can be killed to cancel the remote job
 */
function streamSrun(
  config: Credentials,
  command: string,
  options: RunRemoteOptions = { env: [], silent: false },
  sessionState: ProcessState,
): CloseableEventEmitter {
  const target = `${config.username}@${config.loginHost}`;
  const fullCommand = makeFullCommand(command, options.env);

  // Use -t for pseudo-tty streaming to bypass log buffering
  // Do not use multiplexing as prevents the jobid from being found
  const proc = spawn(
    'ssh',
    [
      '-t',
      '-o',
      'ControlMaster=no',
      '-o',
      'BatchMode=yes',
      target,
      fullCommand,
    ],
    {
      stdio: ['ignore', 'pipe', 'pipe'],
    },
  );

  proc.stdout.on('data', (data) => {
    if (!options.silent) {
      process.stdout.write(data.toString());
    }
  });

  // Use readline interface to parse logs cleanly line by line
  const rl = readline.createInterface({
    input: proc.stderr!,
    terminal: false,
  });

  // let idReceived = false;
  rl.on('line', (line) => {
    // Always print errors to local console unless silent
    console.error(line);

    // See if we can find the slurm job id in the output
    // const srunMatch = line.match(/srun: (?:job|Job) (\d+)/i);

    if (sessionState.slurmJobId === undefined) {
      const clean = line.replaceAll(/[^a-zA-Z0-9 ]/g, '');
      // console.log(clean);
      if (clean.startsWith('srun job ')) {
        const foundId = line.split(' ')[2]!;
        sessionState.slurmJobId = foundId;
        console.log(
          `\n[DEBUG] Bound Slurm Job ID to state: ${sessionState.slurmJobId}`,
        );
      }
    }
  });

  return proc;
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
function copyFile(
  config: Credentials,
  localPath: string,
  remotePath: string,
): Promise<void> {
  return new Promise((resolve, reject) => {
    const target = `${config.username}@${config.loginHost}:${remotePath}`;
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
function tailRemoteLog(
  config: Credentials,
  remotePath: string,
  prefix = '',
): { stop: () => void } {
  const target = `${config.username}@${config.loginHost}`;
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
function spawnTunnel(
  config: Credentials,
  localPort: number,
  remoteHost: string,
  remotePort: number,
): CloseableEventEmitter {
  const target = `${config.username}@${config.loginHost}`;
  // N.B. Mustn't use multiplexing otherwise it looks like the tunnel immediately exits
  // Which triggers shutdown.
  const proc = spawn(
    'ssh',
    [
      '-N',
      '-o',
      'ControlMaster=no', // Force a dedicated connection
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
async function listInstalledVersions(config: Credentials): Promise<string[]> {
  const { stdout } = await runRemote(
    config,
    `ls -d ${config.projectDir}/ivllm/\*/bin 2>/dev/null | sed 's|.*/ivllm/||; s|/bin||'`,
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
export async function matchVllmVersion(
  config: Credentials,
  minVllmVersion: string,
): Promise<string> {
  const installed = await listInstalledVersions(config);
  if (installed.length === 0) {
    throw new Error(
      `No vLLM installation found at ${config.projectDir}/ivllm/. Run 'ivllm setup <version>'.`,
    );
  }

  const minVersion = minVllmVersion;
  const bestVersion = minVersion
    ? selectBestVersion(installed, minVersion)
    : semverSort(installed)[0];

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
export function selectBestVersion(
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
export async function checkSSH(credentials: Credentials): Promise<boolean> {
  console.log('Checking SSH connectivity...');
  const { exitCode: sshCheck } = await runRemote(
    credentials as Credentials,
    'echo ok',
  );
  if (sshCheck !== 0) {
    console.error('Error: Cannot connect to login node.');
    process.exit(1);
  }
  console.log('✓ SSH connectivity OK');
  return true;
}
