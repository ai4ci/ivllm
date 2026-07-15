import { describe, it, expect, beforeEach, afterEach } from 'bun:test';
import {
  writeFileSync,
  unlinkSync,
  readFileSync,
  existsSync,
  rmSync,
  mkdirSync,
} from 'fs';
import { join } from 'path';
import { tmpdir, homedir } from 'os';
import {
  parseVllmConfig,
  parseVllmConfigWithMetadata,
  stripMetadata,
  stripIvllmKeys,
  IVLLM_ONLY_KEYS,
  jobConfigPath,
  saveJobConfig,
  parseEnvVars,
} from '../src/vllm-config.ts';
import yaml from 'js-yaml';

function writeTmp(content: string): string {
  const path = join(tmpdir(), `ivllm-test-${Date.now()}.yaml`);
  writeFileSync(path, content, 'utf-8');
  return path;
}

describe('parseVllmConfig', () => {
  it('extracts model from YAML', () => {
    const path = writeTmp(
      'model: Qwen/Qwen2.5-0.5B-Instruct\nmax-model-len: 8192\n',
    );
    try {
      expect(parseVllmConfig(path).model).toBe('Qwen/Qwen2.5-0.5B-Instruct');
    } finally {
      unlinkSync(path);
    }
  });

  it('extracts tensor-parallel-size (kebab-case) from YAML', () => {
    const path = writeTmp(
      'tensor-parallel-size: 4\nmodel: some/model\nmax-model-len: 8192\n',
    );
    try {
      expect(parseVllmConfig(path).tensorParallelSize).toBe(4);
    } finally {
      unlinkSync(path);
    }
  });

  it('extracts tensor_parallel_size (underscore) from YAML', () => {
    const path = writeTmp(
      'tensor_parallel_size: 8\nmodel: some/model\nmax-model-len: 8192\n',
    );
    try {
      expect(parseVllmConfig(path).tensorParallelSize).toBe(8);
    } finally {
      unlinkSync(path);
    }
  });

  it('extracts pipeline-parallel-size (kebab-case) from YAML', () => {
    const path = writeTmp(
      'pipeline-parallel-size: 2\nmodel: some/model\nmax-model-len: 8192\n',
    );
    try {
      expect(parseVllmConfig(path).pipelineParallelSize).toBe(2);
    } finally {
      unlinkSync(path);
    }
  });

  it('extracts pipeline_parallel_size (underscore) from YAML', () => {
    const path = writeTmp(
      'pipeline_parallel_size: 2\nmodel: some/model\nmax-model-len: 8192\n',
    );
    try {
      expect(parseVllmConfig(path).pipelineParallelSize).toBe(2);
    } finally {
      unlinkSync(path);
    }
  });

  it('returns pipelineParallelSize=1 when not present', () => {
    const path = writeTmp('model: some/model\nmax-model-len: 8192\n');
    try {
      expect(parseVllmConfig(path).pipelineParallelSize).toEqual(1);
    } finally {
      unlinkSync(path);
    }
  });

  it('throws error when model not present', () => {
    const path = writeTmp('max-model-len: 8192\n');
    try {
      expect(() => parseVllmConfig(path)).toThrowError();
    } finally {
      unlinkSync(path);
    }
  });

  it('returns tensorParallelSize=1 when not present', () => {
    const path = writeTmp('model: some/model\nmax-model-len: 8192\n');
    try {
      expect(parseVllmConfig(path).tensorParallelSize).toEqual(1);
    } finally {
      unlinkSync(path);
    }
  });

  it('parses a realistic multi-line config', () => {
    const path = writeTmp(
      'model: Qwen/Qwen2.5-0.5B-Instruct\n' +
        'tensor-parallel-size: 4\n' +
        'max-model-len: 8192\n' +
        'gpu-memory-utilization: 0.90\n',
    );
    try {
      const result = parseVllmConfig(path);
      expect(result.model).toBe('Qwen/Qwen2.5-0.5B-Instruct');
      expect(result.tensorParallelSize).toBe(4);
    } finally {
      unlinkSync(path);
    }
  });

  it('throws a helpful error for a missing file', () => {
    expect(() => parseVllmConfig('/nonexistent/path.yaml')).toThrow();
  });

  it('throws a helpful error for invalid YAML', () => {
    const path = writeTmp('{{invalid: yaml: [\n');
    try {
      expect(() => parseVllmConfig(path)).toThrow();
    } finally {
      unlinkSync(path);
    }
  });

  it('extracts min-vllm-version (kebab-case) from YAML', () => {
    const path = writeTmp(
      'min-vllm-version: "0.9.1"\nmodel: some/model\nmax-model-len: 8192\n',
    );
    try {
      expect(parseVllmConfig(path).minVllmVersion).toBe('0.9.1');
    } finally {
      unlinkSync(path);
    }
  });

  it('returns minVllmVersion=0.15.0 when not present', () => {
    const path = writeTmp('model: some/model\nmax-model-len: 8192\n');
    try {
      expect(parseVllmConfig(path).minVllmVersion).toEqual('0.15.0');
    } finally {
      unlinkSync(path);
    }
  });

  it('extracts max-model-len from YAML', () => {
    const path = writeTmp('model: test\nmax-model-len: 131072\n');
    try {
      expect(parseVllmConfig(path).maxModelLen).toBe(131072);
    } finally {
      unlinkSync(path);
    }
  });

  it('throws error when maxModelLen when not present', () => {
    const path = writeTmp('model: some/model\n');
    try {
      expect(() => parseVllmConfig(path)).toThrowError();
    } finally {
      unlinkSync(path);
    }
  });

  it('extracts enable-auto-tool-choice: true from YAML', () => {
    const path = writeTmp(
      'enable-auto-tool-choice: true\nmodel: some/model\nmax-model-len: 8192\n',
    );
    try {
      expect(parseVllmConfig(path).enableAutoToolChoice).toBe(true);
    } finally {
      unlinkSync(path);
    }
  });

  it('returns enableAutoToolChoice=true when not present', () => {
    const path = writeTmp('model: some/model\nmax-model-len: 8192\n');
    try {
      expect(parseVllmConfig(path).enableAutoToolChoice).toBeTrue();
    } finally {
      unlinkSync(path);
    }
  });

  it('derives enableReasoning: true from presence of reasoning-parser in YAML', () => {
    const path = writeTmp(
      'reasoning-parser: qwen3\nmodel: some/model\nmax-model-len: 8192\n',
    );
    try {
      expect(parseVllmConfig(path).enableReasoning).toBe(true);
    } finally {
      unlinkSync(path);
    }
  });

  it('returns enableReasoning=false when reasoning-parser is absent', () => {
    const path = writeTmp('model: some/model\nmax-model-len: 8192\n');
    try {
      expect(parseVllmConfig(path).enableReasoning).toBeFalse();
    } finally {
      unlinkSync(path);
    }
  });

  it('does not derive enableReasoning from enable-reasoning key (not a valid vLLM key)', () => {
    const path = writeTmp(
      'enable-reasoning: true\nmodel: some/model\nmax-model-len: 8192\n',
    );
    try {
      expect(parseVllmConfig(path).enableReasoning).toBeFalse();
    } finally {
      unlinkSync(path);
    }
  });
});

describe('stripIvllmKeys', () => {
  it('removes min-vllm-version from the output YAML', () => {
    const path = yaml.load(
      'model: some/model\nmin-vllm-version: "0.9.1"\ntensor-parallel-size: 4\n',
    ) as Record<string, unknown>;
    const result = yaml.dump(stripIvllmKeys(path));
    expect(result).not.toContain('min-vllm-version');
  });

  it('preserves all non-ivllm keys', () => {
    const path = yaml.load(
      'model: Qwen/Qwen2.5-0.5B-Instruct\n' +
        'min-vllm-version: "0.9.1"\n' +
        'tensor-parallel-size: 2\n' +
        'max-model-len: 32768\n',
    ) as Record<string, unknown>;
    const result = yaml.dump(stripIvllmKeys(path));
    expect(result).toContain('Qwen/Qwen2.5-0.5B-Instruct');
    expect(result).toContain('tensor-parallel-size');
    expect(result).toContain('max-model-len');
  });

  it('is a no-op when no ivllm-only keys are present', () => {
    const path = yaml.load(
      'model: some/model\ntensor-parallel-size: 4\n',
    ) as Record<string, unknown>;
    const result = yaml.dump(stripIvllmKeys(path));
    expect(result).toContain('some/model');
    expect(result).toContain('tensor-parallel-size');
  });

  it('IVLLM_ONLY_KEYS contains min-vllm-version', () => {
    expect(IVLLM_ONLY_KEYS.has('min-vllm-version')).toBe(true);
  });

  it('IVLLM_ONLY_KEYS contains env', () => {
    expect(IVLLM_ONLY_KEYS.has('env')).toBe(true);
  });

  it('removes env from the output YAML', () => {
    const path = yaml.load(
      'model: some/model\n' +
        'env:\n' +
        '  FOO: bar\n' +
        'tensor-parallel-size: 4\n',
    ) as Record<string, unknown>;
    const result = yaml.dump(stripIvllmKeys(path));
    expect(result).not.toContain('env:');
    expect(result).not.toContain('FOO');
    expect(result).toContain('some/model');
    expect(result).toContain('tensor-parallel-size');
  });
});

describe('env field in VllmConfig', () => {
  it('parses env block from YAML into VllmConfig.env', () => {
    const path = writeTmp(
      'model: some/model\n' +
        'max-model-len: 1234\n' +
        'env:\n' +
        "  VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS: '1'\n" +
        "  VLLM_USE_DEEP_GEMM_FP8: '1'\n",
    );
    try {
      const cfg = parseVllmConfig(path);
      expect(cfg.env).toBeDefined();
      // Find the specific environment entries in the array
      const profilerEnv = cfg.env.find(
        (e) => e.key === 'VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS',
      );
      const gemmEnv = cfg.env.find((e) => e.key === 'VLLM_USE_DEEP_GEMM_FP8');

      // Assert on their values
      expect(profilerEnv?.value).toBe('1');
      expect(gemmEnv?.value).toBe('1');
    } finally {
      unlinkSync(path);
    }
  });

  it('returns empty env object when env block is absent', () => {
    const path = writeTmp(
      'model: some/model\ntensor-parallel-size: 4\nmax-model-len: 8192\n',
    );
    try {
      expect(parseVllmConfig(path).env.length).toEqual(0);
    } finally {
      unlinkSync(path);
    }
  });

  it('returns empty env object when env block is empty', () => {
    const path = writeTmp('model: some/model\nenv: {}\nmax-model-len: 8192\n');
    try {
      const cfg = parseVllmConfig(path);
      expect(cfg.env.length).toEqual(0);
    } finally {
      unlinkSync(path);
    }
  });
});

describe('parseEnvVars', () => {
  it('returns array of {key, value} from env block', () => {
    const path = writeTmp(
      'model: some/model\n' + 'env:\n' + '  FOO: bar\n' + "  BAZ: '123'\n",
    );
    try {
      const vars = parseEnvVars(path);
      expect(vars).toHaveLength(2);
      expect(vars).toContainEqual({ key: 'FOO', value: 'bar' });
      expect(vars).toContainEqual({ key: 'BAZ', value: '123' });
    } finally {
      unlinkSync(path);
    }
  });

  it('returns empty array when no env block', () => {
    const path = writeTmp('model: some/model\nmax-model-len: 8192\n');
    try {
      expect(parseEnvVars(path)).toEqual([]);
    } finally {
      unlinkSync(path);
    }
  });

  it('returns empty array when env block is empty', () => {
    const path = writeTmp('model: some/model\nenv: {}\nmax-model-len: 8192\n');
    try {
      expect(parseEnvVars(path)).toEqual([]);
    } finally {
      unlinkSync(path);
    }
  });
});

describe('jobConfigPath', () => {
  it('returns path under ~/.config/ivllm/<name>.yaml', () => {
    const expected = join(homedir(), '.config', 'ivllm', 'qwen36.yaml');
    expect(jobConfigPath('qwen36')).toBe(expected);
  });

  it('uses the job name verbatim', () => {
    const expected = join(homedir(), '.config', 'ivllm', 'my-job.yaml');
    expect(jobConfigPath('my-job')).toBe(expected);
  });
});

describe('idleTimeout', () => {
  it('defaults to 30 when not specified', () => {
    const p = writeTmp(yaml.dump({ model: 'test', 'max-model-len': 4096 }));
    const cfg = parseVllmConfig(p);
    unlinkSync(p);
    expect(cfg.idleTimeout).toBe(30);
  });

  it('reads idle-timeout from config', () => {
    const p = writeTmp(
      yaml.dump({ model: 'test', 'max-model-len': 4096, 'idle-timeout': 60 }),
    );
    const cfg = parseVllmConfig(p);
    unlinkSync(p);
    expect(cfg.idleTimeout).toBe(60);
  });

  it('accepts -1 for never timeout', () => {
    const p = writeTmp(
      yaml.dump({ model: 'test', 'max-model-len': 4096, 'idle-timeout': -1 }),
    );
    const cfg = parseVllmConfig(p);
    unlinkSync(p);
    expect(cfg.idleTimeout).toBe(-1);
  });

  it('strips idle-timeout from raw config', () => {
    const p = writeTmp(
      yaml.dump({ model: 'test', 'max-model-len': 4096, 'idle-timeout': 30 }),
    );
    const cfg = parseVllmConfig(p);
    unlinkSync(p);
    expect(cfg.raw['idle-timeout']).toBeUndefined();
  });
});

describe('parseVllmConfigWithMetadata', () => {
  it('returns metadata when present', () => {
    const p = writeTmp(
      yaml.dump({
        metadata: {
          version: '1.0',
          author: 'alice',
          lifecycle: 'maturing',
          targetVllmVersion: '0.22.0',
          description: 'test config',
        },
        model: 'test-model',
        'max-model-len': 4096,
      }),
    );
    const { config, metadata } = parseVllmConfigWithMetadata(p);
    unlinkSync(p);
    expect(metadata).not.toBeNull();
    expect(metadata!.version).toBe('1.0');
    expect(metadata!.author).toBe('alice');
    expect(metadata!.lifecycle).toBe('maturing');
    expect(metadata!.targetVllmVersion).toBe('0.22.0');
    expect(metadata!.description).toBe('test config');
    expect(config.model).toBe('test-model');
  });

  it('returns null metadata when no metadata block', () => {
    const p = writeTmp(
      yaml.dump({ model: 'test-model', 'max-model-len': 4096 }),
    );
    const { config, metadata } = parseVllmConfigWithMetadata(p);
    unlinkSync(p);
    expect(metadata).toBeNull();
    expect(config.model).toBe('test-model');
  });
});

describe('stripMetadata', () => {
  it('removes metadata block from YAML', () => {
    const raw = yaml.dump({
      metadata: { version: '1.0', author: 'alice' },
      model: 'test-model',
      'max-model-len': 4096,
      'tensor-parallel-size': 2,
    });
    const stripped = stripMetadata(raw);
    const parsed = yaml.load(stripped) as Record<string, unknown>;
    expect(parsed['metadata']).toBeUndefined();
    expect(parsed['model']).toBe('test-model');
    expect(parsed['tensor-parallel-size']).toBe(2);
  });

  it('preserves all non-metadata keys', () => {
    const raw = yaml.dump({
      model: 'test-model',
      'max-model-len': 4096,
      'gpu-memory-utilization': 0.9,
    });
    const stripped = stripMetadata(raw);
    const parsed = yaml.load(stripped) as Record<string, unknown>;
    expect(parsed['metadata']).toBeUndefined();
    expect(parsed['model']).toBe('test-model');
    expect(parsed['gpu-memory-utilization']).toBe(0.9);
  });
});
