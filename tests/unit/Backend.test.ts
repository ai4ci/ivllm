/**
 * tests/unit/Backend.test.ts — Tests for the abstract Backend class.
 *
 * Tests the base class methods that work purely on data (lockfile parsing,
 * state checking) without needing any RemoteOps implementation.
 */
import { describe, it, expect } from 'bun:test';
import { Backend } from '../../src/backends/Backend';
import type { LockfileV3, Credentials } from '../../src/types';

/** Concrete implementation that delegates to an in-memory store */
class InMemoryBackend extends Backend {
    private lockfiles: Map<string, LockfileV3> = new Map();

    constructor(creds: Credentials) {
        super(creds);
    }

    async bootstrap(): Promise<void> {}
    async setup(_version: string): Promise<void> {}
    async connect(_job: string, _port: number): Promise<any> {
        return { kill: () => true };
    }
    async requestCancel(_job: string, _force: boolean): Promise<void> {}
    async requestStart(
        _job: string,
        _maxTime: string,
        _monitor: boolean,
        _config?: string,
    ): Promise<void> {}
    async getAllJobStatus(): Promise<LockfileV3[]> {
        return Array.from(this.lockfiles.values());
    }
    async watchLog(_job: string, _node?: string, _until?: string): Promise<any> {
        return { kill: () => true };
    }

    /** Internal test helper — set lockfile state */
    setJob(name: string, data: Partial<LockfileV3>): void {
        const existing = this.lockfiles.get(name) ?? {
            status: 'pending' as const,
            jobName: name,
            model: '',
            serverPort: 0,
            user: '',
            requestedTime: '',
            idleTimeout: 30,
        };
        this.lockfiles.set(name, { ...existing, ...data } as LockfileV3);
    }

    removeJob(name: string): void {
        this.lockfiles.delete(name);
    }
}

describe('parseV3Lockfile', () => {
    const backend = new InMemoryBackend({
        loginHost: 'test',
        username: 'test',
        projectDir: '/tmp',
    });

    it('parses valid lockfile', () => {
        const raw = JSON.stringify({
            status: 'running',
            jobName: 'myjob',
            model: 'test-model',
            serverPort: 8000,
            user: 'root',
            requestedTime: '2025-01-01T00:00:00+00:00',
            idleTimeout: 30,
        });
        const result = (backend as any).parseV3Lockfile(raw);
        expect(result).not.toBeNull();
        expect(result!.status).toBe('running');
        expect(result!.jobName).toBe('myjob');
    });

    it('returns null for empty string', () => {
        expect((backend as any).parseV3Lockfile('')).toBeNull();
        expect((backend as any).parseV3Lockfile('  ')).toBeNull();
    });

    it('returns null for malformed JSON', () => {
        expect((backend as any).parseV3Lockfile('{invalid')).toBeNull();
    });

    it('returns null when status is missing', () => {
        expect((backend as any).parseV3Lockfile('{"jobName":"x"}')).toBeNull();
    });

    it('returns null when jobName is missing', () => {
        expect((backend as any).parseV3Lockfile('{"status":"running"}')).toBeNull();
    });
});

describe('getJobStatus', () => {
    it('finds job in list', async () => {
        const backend = new InMemoryBackend({
            loginHost: 'test',
            username: 'test',
            projectDir: '/tmp',
        });
        backend.setJob('myjob', {
            status: 'running',
            model: 'test-model',
            serverPort: 8000,
        });
        const result = await backend.getJobStatus('myjob');
        expect(result.jobName).toBe('myjob');
        expect(result.status).toBe('running');
    });

    it('throws when job not found', async () => {
        const backend = new InMemoryBackend({
            loginHost: 'test',
            username: 'test',
            projectDir: '/tmp',
        });
        await expect(backend.getJobStatus('missing')).rejects.toThrow(
            "Job status for 'missing' not found",
        );
    });
});

describe('isRunning', () => {
    it('returns true for running job', async () => {
        const backend = new InMemoryBackend({
            loginHost: 'test',
            username: 'test',
            projectDir: '/tmp',
        });
        backend.setJob('j', { status: 'running', model: 'm', serverPort: 8000 });
        expect(await backend.isRunning('j')).toBe(true);
    });

    it('returns false for pending job', async () => {
        const backend = new InMemoryBackend({
            loginHost: 'test',
            username: 'test',
            projectDir: '/tmp',
        });
        backend.setJob('j', { status: 'pending', model: 'm', serverPort: 8000 });
        expect(await backend.isRunning('j')).toBe(false);
    });

    it('returns false for missing job', async () => {
        const backend = new InMemoryBackend({
            loginHost: 'test',
            username: 'test',
            projectDir: '/tmp',
        });
        expect(await backend.isRunning('ghost')).toBe(false);
    });
});

describe('isStopped', () => {
    it('returns true for stopped job', async () => {
        const backend = new InMemoryBackend({
            loginHost: 'test',
            username: 'test',
            projectDir: '/tmp',
        });
        backend.setJob('j', { status: 'stopped', model: 'm', serverPort: 8000 });
        expect(await backend.isStopped('j')).toBe(true);
    });

    it('returns true for failed job', async () => {
        const backend = new InMemoryBackend({
            loginHost: 'test',
            username: 'test',
            projectDir: '/tmp',
        });
        backend.setJob('j', { status: 'failed', model: 'm', serverPort: 8000 });
        expect(await backend.isStopped('j')).toBe(true);
    });

    it('returns false for running job', async () => {
        const backend = new InMemoryBackend({
            loginHost: 'test',
            username: 'test',
            projectDir: '/tmp',
        });
        backend.setJob('j', { status: 'running', model: 'm', serverPort: 8000 });
        expect(await backend.isStopped('j')).toBe(false);
    });

    it('returns true for missing job (no status file)', async () => {
        const backend = new InMemoryBackend({
            loginHost: 'test',
            username: 'test',
            projectDir: '/tmp',
        });
        // No job set — missing lockfile means "stopped" by absence
        expect(await backend.isStopped('ghost')).toBe(true);
    });
});

describe('isStartable', () => {
    it('returns true for stopped job', async () => {
        const backend = new InMemoryBackend({
            loginHost: 'test',
            username: 'test',
            projectDir: '/tmp',
        });
        backend.setJob('j', { status: 'stopped', model: 'm', serverPort: 8000 });
        expect(await backend.isStartable('j')).toBe(true);
    });

    it('returns true for failed job', async () => {
        const backend = new InMemoryBackend({
            loginHost: 'test',
            username: 'test',
            projectDir: '/tmp',
        });
        backend.setJob('j', { status: 'failed', model: 'm', serverPort: 8000 });
        expect(await backend.isStartable('j')).toBe(true);
    });

    it('returns false for running job', async () => {
        const backend = new InMemoryBackend({
            loginHost: 'test',
            username: 'test',
            projectDir: '/tmp',
        });
        backend.setJob('j', { status: 'running', model: 'm', serverPort: 8000 });
        expect(await backend.isStartable('j')).toBe(false);
    });

    it('returns true for missing job (no status file)', async () => {
        const backend = new InMemoryBackend({
            loginHost: 'test',
            username: 'test',
            projectDir: '/tmp',
        });
        // No job set — missing lockfile means the job was never started,
        // so it is definitely startable.
        expect(await backend.isStartable('ghost')).toBe(true);
    });
});

describe('isStarting', () => {
    it('returns true for pending job', async () => {
        const backend = new InMemoryBackend({
            loginHost: 'test',
            username: 'test',
            projectDir: '/tmp',
        });
        backend.setJob('j', { status: 'pending', model: 'm', serverPort: 8000 });
        expect(await backend.isStarting('j')).toBe(true);
    });

    it('returns true for initialising job', async () => {
        const backend = new InMemoryBackend({
            loginHost: 'test',
            username: 'test',
            projectDir: '/tmp',
        });
        backend.setJob('j', { status: 'initialising', model: 'm', serverPort: 8000 });
        expect(await backend.isStarting('j')).toBe(true);
    });

    it('returns false for running job', async () => {
        const backend = new InMemoryBackend({
            loginHost: 'test',
            username: 'test',
            projectDir: '/tmp',
        });
        backend.setJob('j', { status: 'running', model: 'm', serverPort: 8000 });
        expect(await backend.isStarting('j')).toBe(false);
    });

    it('returns false for missing job', async () => {
        const backend = new InMemoryBackend({
            loginHost: 'test',
            username: 'test',
            projectDir: '/tmp',
        });
        expect(await backend.isStarting('ghost')).toBe(false);
    });
});
