import { describe, test, expect } from 'bun:test';
import { makeV3Paths, parseV3Lockfile } from '../src/job';

describe('makeV3Paths', () => {
  test('builds correct paths for a job', () => {
    const paths = makeV3Paths('/projects/XXXX', 'my-job');
    expect(paths.engineDir).toBe('/projects/XXXX/engine');
    expect(paths.jobsDir).toBe('/projects/XXXX/engine/jobs');
    expect(paths.jobDir).toBe('/projects/XXXX/engine/jobs/my-job');
    expect(paths.lockfilePath).toBe('/projects/XXXX/engine/jobs/my-job/status.json');
    expect(paths.logPath).toBe('/projects/XXXX/engine/jobs/my-job/vllm.0.log');
    expect(paths.configPath).toBe('/projects/XXXX/engine/jobs/my-job/vllm.yaml');
    expect(paths.strippedConfigPath).toBe('/projects/XXXX/engine/jobs/my-job/vllm.stripped.yaml');
    expect(paths.scriptPath).toBe('/projects/XXXX/engine/jobs/my-job/slurm.sh');
    expect(paths.cachePath).toBe('/projects/XXXX/engine/jobs/my-job/jit-cache.tar.gz');
    expect(paths.libDir).toBe('/projects/XXXX/engine/lib');
  });

  test('handles jobs with hyphens and dots', () => {
    const paths = makeV3Paths('/p/test', 'my-model.v2');
    expect(paths.jobDir).toContain('my-model.v2');
    expect(paths.lockfilePath).toContain('my-model.v2');
  });

  test('handles project dir without trailing slash', () => {
    const paths = makeV3Paths('/projects/XXXX/', 'job');
    // Should not produce double slashes
    expect(paths.engineDir).not.toContain('//');
  });
});

describe('parseV3Lockfile', () => {
  test('parses valid lockfile', () => {
    const raw = JSON.stringify({
      status: 'running',
      jobName: 'test-job',
      model: 'test-model',
      serverPort: 54321,
      requestedTime: '2026-07-15T12:00:00Z',
      idleTimeout: 30,
      slurmJobId: '123456',
      computeHostname: 'nid00123',
      vllmPid: 98765,
    });
    const result = parseV3Lockfile(raw);
    expect(result).not.toBeNull();
    expect(result!.status).toBe('running');
    expect(result!.jobName).toBe('test-job');
    expect(result!.model).toBe('test-model');
    expect(result!.serverPort).toBe(54321);
    expect(result!.slurmJobId).toBe('123456');
  });

  test('parses lockfile with minimal fields', () => {
    const raw = JSON.stringify({
      status: 'pending',
      jobName: 'new-job',
      model: 'new-model',
      serverPort: 49152,
      requestedTime: '2026-07-15T12:00:00Z',
      idleTimeout: 30,
    });
    const result = parseV3Lockfile(raw);
    expect(result).not.toBeNull();
    expect(result!.status).toBe('pending');
    expect(result!.computeHostname).toBeUndefined();
  });

  test('parses lockfile with cancel status', () => {
    const raw = JSON.stringify({
      status: 'cancel',
      jobName: 'cancel-job',
      model: 'm',
      serverPort: 50000,
      requestedTime: '2026-07-15T12:00:00Z',
      idleTimeout: 30,
    });
    const result = parseV3Lockfile(raw);
    expect(result).not.toBeNull();
    expect(result!.status).toBe('cancel');
  });

  test('parses lockfile with failure info', () => {
    const raw = JSON.stringify({
      status: 'failed',
      jobName: 'fail-job',
      model: 'm',
      serverPort: 50000,
      requestedTime: '2026-07-15T12:00:00Z',
      idleTimeout: 30,
      reason: 'GPU error',
      exitCode: 42,
      stopTime: '2026-07-15T13:00:00Z',
    });
    const result = parseV3Lockfile(raw);
    expect(result).not.toBeNull();
    expect(result!.status).toBe('failed');
    expect(result!.reason).toBe('GPU error');
    expect(result!.exitCode).toBe(42);
  });

  test('returns null for empty input', () => {
    expect(parseV3Lockfile('')).toBeNull();
    expect(parseV3Lockfile('  ')).toBeNull();
  });

  test('returns null for malformed JSON', () => {
    expect(parseV3Lockfile('not json')).toBeNull();
    expect(parseV3Lockfile('{"status": 123}')).toBeNull(); // status must be string
    expect(parseV3Lockfile('{"status": "running"}')).toBeNull(); // missing jobName
  });

  test('returns null for completely invalid input', () => {
    expect(parseV3Lockfile('null')).toBeNull();
    expect(parseV3Lockfile('{}')).toBeNull();
  });
});
