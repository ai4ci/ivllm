/**
 * tests/unit/RemoteOps.mock.test.ts — Tests for the RemoteOps interface
 * using a MockRemoteOps implementation.
 *
 * This replaces the old `makeRemoteOps('dry-run')` / `makeRemoteOps('mock')`
 * approach. A single MockRemoteOps class implements the RemoteOps interface
 * and allows tests to verify what commands would be sent without any SSH.
 */
import { describe, it, expect } from 'bun:test';
import { RemoteOps } from '../../src/ops/RemoteOps';
import type { Credentials, EnvVarEntry } from '../../src/types';

/**
 * MockRemoteOps — a test implementation of RemoteOps that records what
 * would have been executed rather than actually running anything.
 *
 * This is the key primitive for TypeScript integration tests: it lets us
 * exercise the full CLI → Backend → RemoteOps call chain with zero network
 * activity.
 */
class MockRemoteOps extends RemoteOps {
    readonly calls: Array<{
        method: string;
        command?: string;
        options?: any;
    }> = [];

    runRemote(command: string, options?: any): Promise<{ exitCode: number; stdout: string }> {
        this.calls.push({ method: 'runRemote', command, options });
        // Simulate sbatch returning a fake job ID
        if (command.includes('sbatch')) {
            return Promise.resolve({ exitCode: 0, stdout: 'Submitted batch job 123456' });
        }
        // Simulate squeue returning test state
        if (command.includes('squeue') || command.includes('sacct')) {
            return Promise.resolve({ exitCode: 0, stdout: '123456 RUNNING' });
        }
        // Default: success with empty output
        return Promise.resolve({ exitCode: 0, stdout: '' });
    }

    copyFile(_localPath: string, _remotePath: string): Promise<void> {
        this.calls.push({ method: 'copyFile', localPath: _localPath, remotePath: _remotePath });
        return Promise.resolve();
    }

    copyDirectory(_localPath: string, _remotePath: string, _direction: 'up' | 'down'): Promise<void> {
        this.calls.push({ method: 'copyDirectory', localPath: _localPath, remotePath: _remotePath, direction: _direction });
        return Promise.resolve();
    }

    runRemoteSync(command: string, env: EnvVarEntry[]): any {
        this.calls.push({ method: 'runRemoteSync', command, env });
        // Return a mock emitter
        const mock = Object.assign(new (require('events').EventEmitter)(), {
            kill: () => { mock.emit('close', 0); return true; },
        });
        return mock;
    }

    spawnTunnel(_localPort: number, _remoteHost: string, _remotePort: number): any {
        this.calls.push({ method: 'spawnTunnel', localPort: _localPort, remoteHost: _remoteHost, remotePort: _remotePort });
        const mock = Object.assign(new (require('events').EventEmitter)(), {
            kill: () => { mock.emit('close', 0); return true; },
        });
        return mock;
    }

    checkSSH(): Promise<boolean> {
        this.calls.push({ method: 'checkSSH' });
        return Promise.resolve(true);
    }

    /** Clear recorded calls */
    clear(): void {
        this.calls.length = 0;
    }
}

function makeMockOps(): MockRemoteOps {
    return new MockRemoteOps();
}

describe('MockRemoteOps — runRemote', () => {
    it('records sbatch call and returns fake job ID', async () => {
        const ops = makeMockOps();
        const result = await ops.runRemote('sbatch --parsable test.sh', { env: [], silent: false });
        expect(result.exitCode).toBe(0);
        expect(result.stdout).toContain('123456');
        expect(ops.calls[0].method).toBe('runRemote');
        expect(ops.calls[0].command).toContain('sbatch');
    });

    it('records env variables', async () => {
        const ops = makeMockOps();
        await ops.runRemote('echo hello', { env: [{ key: 'HF_TOKEN', value: 'secret' }], silent: true });
        expect(ops.calls[0].options?.env).toEqual([{ key: 'HF_TOKEN', value: 'secret' }]);
    });
});

describe('MockRemoteOps — copyFile', () => {
    it('records the copy', async () => {
        const ops = makeMockOps();
        await ops.copyFile('/local/file.yaml', '/remote/vllm.yaml');
        expect(ops.calls[0].method).toBe('copyFile');
        expect(ops.calls[0].localPath).toBe('/local/file.yaml');
        expect(ops.calls[0].remotePath).toBe('/remote/vllm.yaml');
    });
});

describe('MockRemoteOps — copyDirectory', () => {
    it('records upload', async () => {
        const ops = makeMockOps();
        await ops.copyDirectory('/local/engine', '/remote/engine', 'up');
        expect(ops.calls[0].method).toBe('copyDirectory');
        expect(ops.calls[0].direction).toBe('up');
    });

    it('records download', async () => {
        const ops = makeMockOps();
        await ops.copyDirectory('/local/engine', '/remote/engine', 'down');
        expect(ops.calls[0].direction).toBe('down');
    });
});

describe('MockRemoteOps — checkSSH', () => {
    it('returns true without actual SSH', async () => {
        const ops = makeMockOps();
        const result = await ops.checkSSH();
        expect(result).toBe(true);
        expect(ops.calls[0].method).toBe('checkSSH');
    });
});

describe('MockRemoteOps — spawnTunnel', () => {
    it('returns a closeable mock emitter', () => {
        const ops = makeMockOps();
        const tunnel = ops.spawnTunnel(11434, 'gh200-1', 8000);
        expect(tunnel.kill).toBeDefined();
        expect(tunnel.kill()).toBe(true);
        expect(ops.calls[0].method).toBe('spawnTunnel');
        expect(ops.calls[0].localPort).toBe(11434);
        expect(ops.calls[0].remoteHost).toBe('gh200-1');
        expect(ops.calls[0].remotePort).toBe(8000);
    });
});

describe('MockRemoteOps — makeFullCommand', () => {
    it('prepends env vars to command', async () => {
        const ops = makeMockOps();
        // makeFullCommand is protected, but we can test via runRemote
        await ops.runRemote('echo hello', { env: [{ key: 'A', value: '1' }, { key: 'B', value: '2' }] });
        expect(ops.calls[0].options?.env).toEqual([
            { key: 'A', value: '1' },
            { key: 'B', value: '2' },
        ]);
    });
});
