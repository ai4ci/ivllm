import { describe, it, expect, beforeEach, afterEach } from 'bun:test';
import { existsSync, mkdirSync, writeFileSync, rmSync, readdirSync } from 'fs';
import { join } from 'path';
import { listJobConfigs, JOB_CONFIG_DIR } from '../src/vllm-config.ts';

function writeTestConfig(jobName: string, content: string): string {
  if (!existsSync(JOB_CONFIG_DIR)) {
    mkdirSync(JOB_CONFIG_DIR, { recursive: true });
  }
  const filePath = join(JOB_CONFIG_DIR, `${jobName}.yaml`);
  writeFileSync(filePath, content, 'utf-8');
  return filePath;
}

function cleanupTestConfigs(): void {
  if (!existsSync(JOB_CONFIG_DIR)) return;
  const files = readdirSync(JOB_CONFIG_DIR);
  for (const f of files) {
    if (f.startsWith('ivllm-test-') && f.endsWith('.yaml')) {
      rmSync(join(JOB_CONFIG_DIR, f));
    }
  }
}

describe('listJobConfigs', () => {
  beforeEach(() => {
    cleanupTestConfigs();
  });

  afterEach(() => {
    cleanupTestConfigs();
  });

  it('returns empty array when no configs exist', () => {
    // Ensure no test configs are present
    const entries = listJobConfigs().filter((e) =>
      e.jobName.startsWith('ivllm-test-'),
    );
    expect(entries.length).toBe(0);
  });

  it('lists stored configs with parsed metadata', () => {
    writeTestConfig(
      'ivllm-test-single',
      `
model: Qwen/Qwen2.5-0.5B-Instruct
max-model-len: 256
tensor-parallel-size: 4
pipeline-parallel-size: 1
`,
    );
    const entries = listJobConfigs().filter((e) =>
      e.jobName.startsWith('ivllm-test-'),
    );
    expect(entries.length).toBe(1);
    expect(entries[0]!.jobName).toBe('ivllm-test-single');
    expect(entries[0]!.model).toBe('Qwen/Qwen2.5-0.5B-Instruct');
    expect(entries[0]!.tensorParallelSize).toBe(4);
    expect(entries[0]!.pipelineParallelSize).toBe(1);
  });

  it('lists multiple configs sorted alphabetically', () => {
    writeTestConfig('ivllm-test-zulu', 'model: org/zulu\n');
    writeTestConfig('ivllm-test-alpha', 'model: org/alpha\n');
    const entries = listJobConfigs().filter((e) =>
      e.jobName.startsWith('ivllm-test-'),
    );
    expect(entries.length).toBe(2);
    expect(entries[0]!.jobName).toBe('ivllm-test-alpha');
    expect(entries[1]!.jobName).toBe('ivllm-test-zulu');
  });

  it('handles malformed config gracefully', () => {
    writeTestConfig('ivllm-test-broken', 'not: valid: yaml: [');
    const entries = listJobConfigs().filter((e) =>
      e.jobName.startsWith('ivllm-test-'),
    );
    expect(entries.length).toBe(1);
    expect(entries[0]!.jobName).toBe('ivllm-test-broken');
    expect(entries[0]!.model).toBeUndefined();
  });

  it('ignores non-yaml files in the config directory', () => {
    if (!existsSync(JOB_CONFIG_DIR))
      mkdirSync(JOB_CONFIG_DIR, { recursive: true });
    writeFileSync(
      join(JOB_CONFIG_DIR, 'ivllm-test-config.json'),
      '{}',
      'utf-8',
    );
    writeFileSync(
      join(JOB_CONFIG_DIR, 'ivllm-test-readme.txt'),
      'hello',
      'utf-8',
    );
    writeTestConfig('ivllm-test-valid', 'model: org/model\n');
    const entries = listJobConfigs().filter((e) =>
      e.jobName.startsWith('ivllm-test-'),
    );
    expect(entries.length).toBe(1);
    expect(entries[0]!.jobName).toBe('ivllm-test-valid');
    // Cleanup non-yaml test files
    rmSync(join(JOB_CONFIG_DIR, 'ivllm-test-config.json'));
    rmSync(join(JOB_CONFIG_DIR, 'ivllm-test-readme.txt'));
  });
});
