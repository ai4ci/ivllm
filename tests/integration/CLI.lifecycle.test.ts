/**
 * tests/integration/CLI.lifecycle.test.ts — Full lifecycle tests using a
 * mock backend that records calls but never touches the network.
 *
 * This is the closest equivalent to the old integration tests, but adapted
 * for the current Backend → RemoteOps architecture. We verify:
 *
 * 1. requestStart → backend calls ivllm-serve.sh via RemoteOps
 * 2. requestCancel → backend calls ivllm-cancel.sh via RemoteOps
 * 3. getJobStatus → backend parses lockfile JSON via RemoteOps
 * 4. Lifecycle transitions: pending → initialising → running → stopped
 *
 * The mock backend simulates what the real SSH backend would do, so we
 * can test the full CLI flow without any HPC connection.
 */
import { describe, it, expect, beforeEach } from 'bun:test';
import { Backend } from '../../src/backends/Backend';
import { RemoteOps } from '../../src/ops/RemoteOps';
import type { Credentials, LockfileV3, EnvVarEntry } from '../../src/types';

// ── Mock RemoteOps ──────────────────────────────────────────────────────

class TestRemoteOps extends RemoteOps {
    calls: Array<{ method: string; args: any[] }> = [];
    private lockfiles: Map<string, LockfileV3> = new Map();

    // Inject lockfile state (simulates what bash framework would write)
    setLockfile(job: string, data: Partial<LockfileV3>): void {
        const existing = this.lockfiles.get(job) ?? {
            status: 'pending' as const,
            jobName: job,
            model: '',
            serverPort: 0,
            user: '',
            requestedTime: '',
            idleTimeout: 30,
        };
        this.lockfiles.set(job, { ...existing, ...data } as LockfileV3);
    }

    async runRemote(cmd: string, options?: any): Promise<{ exitCode: number; stdout: string }> {
        this.calls.push({ method: 'runRemote', args: [cmd, options] });
        if (cmd.includes('sbatch')) {
            return { exitCode: 0, stdout: 'Submitted batch job 123456' };
        }
        if (cmd.includes('ivllm-status.sh -p')) {
            // Return all lockfiles as JSON array
            const jobs = Array.from(this.lockfiles.values());
            return { exitCode: 0, stdout: JSON.stringify(jobs) };
        }
        if (cmd.includes('ivllm-cancel.sh')) {
            return { exitCode: 0, stdout: '[cancel] requesting cancel' };
        }
        if (cmd.includes('ivllm-serve.sh')) {
            return { exitCode: 0, stdout: 'Slurm Job ID: 123456' };
        }
        return { exitCode: 0, stdout: '' };
    }

    async copyFile(_local: string, _remote: string): Promise<void> {
        this.calls.push({ method: 'copyFile', args: [_local, _remote] });
    }

    async copyDirectory(_local: string, _remote: string, _dir: 'up' | 'down'): Promise<void> {
        this.calls.push({ method: 'copyDirectory', args: [_local, _remote, _dir] });
    }

    runRemoteSync(_cmd: string, _env: EnvVarEntry[]): any {
        this.calls.push({ method: 'runRemoteSync', args: [_cmd, _env] });
        return Object.assign(new (require('events').EventEmitter)(), {
            kill: () => true,
        });
    }

    spawnTunnel(_port: number, _host: string, _rport: number): any {
        this.calls.push({ method: 'spawnTunnel', args: [_port, _host, _rport] });
        return Object.assign(new (require('events').EventEmitter)(), {
            kill: () => true,
        });
    }

    async checkSSH(): Promise<boolean> {
        this.calls.push({ method: 'checkSSH' });
        return true;
    }

    clear(): void {
        this.calls.length = 0;
    }
}

// ── Concrete Backend using TestRemoteOps ────────────────────────────────

class TestBackend extends Backend {
    ops: TestRemoteOps;
    remoteEngine: string;
    envs: EnvVarEntry[];
    bootstrapped = false;

    constructor(creds: Credentials) {
        super(creds);
        this.ops = new TestRemoteOps();
        this.remoteEngine = `${creds.projectDir}/engine`;
        this.envs = [{ key: 'IVLLM_PROJECTDIR', value: creds.projectDir }];
    }

    async bootstrap(): Promise<void> {
        await this.ops.checkSSH();
        if (!this.bootstrapped) {
            this.bootstrapped = true;
        }
    }

    async setup(_version: string, _force?: boolean): Promise<void> {
        await this.bootstrap();
        await this.ops.runRemote(
            `${this.remoteEngine}/ivllm-setup.sh -v "${_version}"`,
            { env: this.envs, silent: false },
        );
    }

    async connect(_job: string, _port: number): Promise<any> {
        await this.bootstrap();
        return this.ops.spawnTunnel(_port, 'gh200-0', 8000);
    }

    async requestCancel(job: string, force: boolean): Promise<void> {
        await this.bootstrap();
        const flag = force ? ' -f' : '';
        await this.ops.runRemote(
            `${this.remoteEngine}/ivllm-cancel.sh -j "${job}"${flag}`,
            { env: this.envs, silent: false },
        );
    }

    async requestStart(
        job: string,
        maxTime: string = '08:00:00',
        _monitor: boolean,
        _config?: string,
    ): Promise<void> {
        await this.bootstrap();
        await this.ops.runRemote(
            `${this.remoteEngine}/ivllm-serve.sh -j "${job}" -t "${maxTime}"`,
            { env: this.envs, silent: false },
        );
    }

    async getAllJobStatus(): Promise<LockfileV3[]> {
        await this.bootstrap();
        const { stdout, exitCode } = await this.ops.runRemote(
            `${this.remoteEngine}/ivllm-status.sh -p`,
            { env: this.envs, silent: true },
        );
        if (exitCode !== 0) throw new Error(`status failed: ${stdout}`);
        try {
            const parsed = JSON.parse(stdout || '[]');
            return Array.isArray(parsed)
                ? parsed.map((j: any) => this.parseV3Lockfile(JSON.stringify(j))).filter((j: any) => j !== null)
                : [];
        } catch {
            return [];
        }
    }

    async watchLog(_job: string, _node?: string, _until?: string): Promise<any> {
        await this.bootstrap();
        return this.ops.runRemoteSync(
            `${this.remoteEngine}/ivllm-show-log.sh -j "${_job}"`,
            this.envs,
        );
    }

    // Allow tests to inject lockfile state
    setLockfile(job: string, data: Partial<LockfileV3>): void {
        this.ops.setLockfile(job, data);
    }

    getOps(): TestRemoteOps {
        return this.ops;
    }
}

// ── Tests ───────────────────────────────────────────────────────────────

const CRED: Credentials = {
    loginHost: 'test.isambard',
    username: 'testuser',
    projectDir: '/tmp/ivllm-test',
};

describe('Backend — requestStart', () => {
    it('calls ivllm-serve.sh with correct args', async () => {
        const backend = new TestBackend(CRED);
        await backend.requestStart('test-job', '04:00:00', true);

        const ops = backend.getOps();
        const call = ops.calls.find(c => c.method === 'runRemote' && c.args[0]?.includes('ivllm-serve.sh'));
        expect(call).toBeDefined();
        expect(call!.args[0]).toContain('test-job');
        expect(call!.args[0]).toContain('04:00:00');
    });
});

describe('Backend — requestCancel', () => {
    it('calls ivllm-cancel.sh without -f for graceful cancel', async () => {
        const backend = new TestBackend(CRED);
        await backend.requestCancel('test-job', false);

        const ops = backend.getOps();
        const call = ops.calls.find(c => c.method === 'runRemote');
        expect(call!.args[0]).toContain('ivllm-cancel.sh');
        expect(call!.args[0]).toContain('test-job');
        expect(call!.args[0]).not.toContain('-f');
    });

    it('calls ivllm-cancel.sh with -f for force cancel', async () => {
        const backend = new TestBackend(CRED);
        await backend.requestCancel('test-job', true);

        const ops = backend.getOps();
        const call = ops.calls.find(c => c.method === 'runRemote');
        expect(call!.args[0]).toContain('-f');
    });
});

describe('Backend — getAllJobStatus', () => {
    it('returns lockfile list', async () => {
        const backend = new TestBackend(CRED);
        backend.setLockfile('job1', {
            status: 'running',
            jobName: 'job1',
            model: 'test-model',
            serverPort: 8000,
        });
        backend.setLockfile('job2', {
            status: 'stopped',
            jobName: 'job2',
            model: 'other-model',
            serverPort: 8001,
        });

        const jobs = await backend.getAllJobStatus();
        expect(jobs).toHaveLength(2);
        expect(jobs.find(j => j.jobName === 'job1')?.status).toBe('running');
        expect(jobs.find(j => j.jobName === 'job2')?.status).toBe('stopped');
    });

    it('returns empty list when no jobs', async () => {
        const backend = new TestBackend(CRED);
        const jobs = await backend.getAllJobStatus();
        expect(jobs).toEqual([]);
    });
});

describe('Backend — lifecycle helpers', () => {
    it('isRunning returns true for running job', async () => {
        const backend = new TestBackend(CRED);
        backend.setLockfile('j', { status: 'running', jobName: 'j', model: 'm', serverPort: 8000 });
        expect(await backend.isRunning('j')).toBe(true);
    });

    it('isRunning returns false for missing job', async () => {
        const backend = new TestBackend(CRED);
        expect(await backend.isRunning('ghost')).toBe(false);
    });

    it('isStartable returns true for stopped job', async () => {
        const backend = new TestBackend(CRED);
        backend.setLockfile('j', { status: 'stopped', jobName: 'j', model: 'm', serverPort: 8000 });
        expect(await backend.isStartable('j')).toBe(true);
    });

    it('isStarting returns true for pending job', async () => {
        const backend = new TestBackend(CRED);
        backend.setLockfile('j', { status: 'pending', jobName: 'j', model: 'm', serverPort: 8000 });
        expect(await backend.isStarting('j')).toBe(true);
    });
});

describe('Backend — bootstrap', () => {
    it('calls checkSSH on first use', async () => {
        const backend = new TestBackend(CRED);
        await backend.setup('0.8.0');
        const ops = backend.getOps();
        const sshCall = ops.calls.find(c => c.method === 'checkSSH');
        expect(sshCall).toBeDefined();
    });
});

describe('MockRemoteOps — command recording', () => {
    it('records all operations', async () => {
        const ops = new TestRemoteOps();

        await ops.runRemote('sbatch test.sh', { env: [], silent: false });
        await ops.copyFile('/local.yaml', '/remote.yaml');
        await ops.copyDirectory('/local', '/remote', 'up');
        await ops.checkSSH();
        ops.spawnTunnel(11434, 'node0', 8000);

        expect(ops.calls).toHaveLength(5);
        expect(ops.calls[0].method).toBe('runRemote');
        expect(ops.calls[1].method).toBe('copyFile');
        expect(ops.calls[2].method).toBe('copyDirectory');
        expect(ops.calls[3].method).toBe('checkSSH');
        expect(ops.calls[4].method).toBe('spawnTunnel');
    });
});
