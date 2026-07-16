import {
  copyFileSync,
  mkdirSync,
  writeFileSync,
  readFileSync,
  existsSync,
  rmSync,
} from 'fs';
import { basename, join } from 'path';
import EventEmitter from 'node:events';
import { spawn, execSync } from 'child_process';
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
  OpsMode,
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
 * @param mode - Operation mode: 'real' (SSH), 'mock' (local sandbox), 'dry-run' (print only)
 * @returns An object conforming to the {@link RemoteOps} interface
 */
export function makeRemoteOps(
  config: Credentials,
  mode: OpsMode = 'real',
): RemoteOps {
  if (mode === 'real') {
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

  // Dry-run mode: print what would happen, return canned responses.
  if (mode === 'dry-run') {
    return buildDryRunOps();
  }

  // Mock mode: execute commands against a local filesystem sandbox.
  const mockFs = new MockRemoteFs();
  return buildMockOps(mockFs);
}

// ── Dry-run implementation ───────────────────────────────────────────────────

function buildDryRunOps(): RemoteOps {
  return {
    async runRemote(command: string, opts) {
      const fullCommand = makeFullCommand(command, opts?.env || []);
      console.log(`  [dry-run] Would run remotely:
    ${fullCommand}`);
      return {
        exitCode: 0,
        stdout: command.startsWith('sbatch')
          ? 'Submitted batch job 123456'
          : command.startsWith('squeue')
            ? 'test-state test-reason'
            : command.startsWith('sacct')
              ? '123456 RUNNING'
              : command.startsWith('ls -d')
                ? '0.99.99'
                : '',
      };
    },
    async copyFile(localPath, remotePath) {
      const dryRunDir = join(tmpdir(), 'ivllm-dryrun');
      mkdirSync(dryRunDir, { recursive: true });
      const destPath = join(dryRunDir, basename(remotePath));
      copyFileSync(localPath, destPath);
      console.log(`  [dry-run] Would scp: ${localPath} → ${remotePath}`);
      console.log(`           (preview: ${destPath})`);
    },
    streamSrun(command, sessionState, _opts) {
      const fullCommand = makeFullCommand(command, _opts?.env || []);
      console.log(`  [dry-run] Would stream remotely:
    ${fullCommand}`);
      console.log(`srun: job 123456`);
      sessionState.slurmJobId = '123456';
      return createMockSSh('streaming srun', 2222);
    },
    tailRemoteLog(_remote, _prefix) {
      return { stop: () => {} };
    },
    spawnTunnel(local, remoteHost, remotePort) {
      console.log(
        `  [dry-run] mock SSH tunnel: ${local}:${remoteHost}:${remotePort}`,
      );
      return createMockSSh('tunnel', 1111);
    },
    async matchVllmVersion(minVllmVersion) {
      return (
        selectBestVersion(['0.22.0', minVllmVersion], minVllmVersion) ??
        '0.22.0'
      );
    },
    async checkSSH() {
      console.log('  [dry-run] skipping SSH check');
      return true;
    },
  };
}

// ── Mock implementation (local filesystem sandbox) ──────────────────────────

function buildMockOps(mockFs: MockRemoteFs): RemoteOps {
  return {
    async runRemote(command: string, opts) {
      return mockFs.exec(command, opts?.env);
    },
    async copyFile(localPath, remotePath) {
      const dest = join(mockFs.baseDir, basename(remotePath));
      mkdirSync(mockFs.baseDir, { recursive: true });
      copyFileSync(localPath, dest);
      console.log(`  [mock] cp ${localPath} → ${dest}`);
    },
    streamSrun(command, sessionState, _opts) {
      console.log(`  [mock] srun: ${command}`);
      sessionState.slurmJobId = '123456';
      return createMockSSh('streaming srun', 2222);
    },
    tailRemoteLog(_remote, _prefix) {
      return { stop: () => {} };
    },
    spawnTunnel(_local, _remoteHost, _remotePort) {
      return createMockSSh('tunnel', 1111);
    },
    async matchVllmVersion(minVllmVersion) {
      return (
        selectBestVersion(['0.22.0', minVllmVersion], minVllmVersion) ??
        '0.22.0'
      );
    },
    async checkSSH() {
      return true;
    },
  };
}

// ── MockRemoteFs — local filesystem sandbox ─────────────────────────────────

/**
 * Simulates a remote HPC filesystem for integration testing.
 *
 * Commands like `mkdir`, `cat`, `jq`, `set -C` are executed against a
 * local temp directory. This allows tests to create lockfiles, read them
 * back, and simulate state transitions without real SSH.
 */
class MockRemoteFs {
  readonly baseDir: string;

  constructor() {
    this.baseDir = join(tmpdir(), `ivllm-mock-${Date.now()}`);
    mkdirSync(this.baseDir, { recursive: true });
  }

  /**
   * Execute a shell command against the local sandbox.
   * Supports: mkdir, cat, echo (with heredoc), jq, set -C, sbatch, scancel.
   */
  async exec(command: string, _env?: EnvVarEntry[]): Promise<RunRemoteResult> {
    // sbatch → fake job ID
    if (command.startsWith('sbatch')) {
      return { exitCode: 0, stdout: 'Submitted batch job 123456' };
    }

    // scancel → no-op
    if (command.startsWith('scancel')) {
      return { exitCode: 0, stdout: '' };
    }

    // Wrap the command so it runs inside the sandbox directory.
    // Redirect stderr to avoid noise, but capture actual outputs.
    const wrapped = `set -e; cd '${this.baseDir}'; ${command} 2>/dev/null`;

    try {
      const stdout = execSync(wrapped, {
        encoding: 'utf-8',
        timeout: 5000,
      }).trim();
      return { exitCode: 0, stdout };
    } catch (e: unknown) {
      const err = e as {
        status?: number;
        stdout?: string;
        stderr?: string;
        message?: string;
      };
      return {
        exitCode: err.status ?? 1,
        stdout: ((err.stdout as string) ?? '').toString().trim(),
      };
    }
  }

  /** Read a file from the sandbox. */
  readFile(filePath: string): string {
    const fullPath = join(this.baseDir, filePath);
    if (!existsSync(fullPath)) return '';
    return readFileSync(fullPath, 'utf-8').trim();
  }

  /** Check if a path exists in the sandbox. */
  exists(filePath: string): boolean {
    return existsSync(join(this.baseDir, filePath));
  }

  /** Clean up the sandbox directory. */
  cleanup(): void {
    try {
      rmSync(this.baseDir, { recursive: true, force: true });
    } catch {
      // ignore cleanup errors
    }
  }
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
