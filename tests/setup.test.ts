import { describe, it, expect } from 'bun:test';
import { renderSetupScript } from '../src/templates/setup.ts';
import { parseJobId, parseJobState } from '../src/slurm.ts';
import { ProcessState, type Credentials } from '../src/types.ts';
import { makeRemoteOps } from '../src/remote-ops.ts';
import { makeSimplePaths } from '../src/job.ts';

describe('renderSetupScript', () => {
  const creds: Credentials = {
    loginHost: 'test.example.com',
    username: 'test-user',
    projectDir: '/projects/p',
    defaultLocalPort: 11434,
    hfToken: 'hf_mytoken',
  };

  const base: ProcessState = {
    sessionName: 'test',
    ops: makeRemoteOps(creds, 'dry-run'),
    paths: makeSimplePaths(creds, '0.19.1'),
    vllmVersion: '0.19.1',
  };

  it('requests 1 GPU so --torch-backend=auto can detect CUDA via nvidia-smi', () => {
    const script = renderSetupScript(base, '/tmp/remote.log');
    expect(script).toContain('#SBATCH --gpus=1');
  });

  it('uses --torch-backend=auto for CUDA torch wheel selection', () => {
    const script = renderSetupScript(base, '/tmp/remote.log');
    expect(script).toContain('--torch-backend=auto');
  });

  it('installs HPC SDK to /projects/p/ivllm/nvhpc', () => {
    const script = renderSetupScript(base, '/tmp/remote.log');
    expect(script).toContain('/projects/p/ivllm/nvhpc');
    expect(script).toContain('nvhpc_2026_263_Linux_aarch64_cuda_multi');
  });

  it('skips HPC SDK install if /projects/p/ivllm/nvhpc already exists', () => {
    const script = renderSetupScript(base, '/tmp/remote.log');
    expect(script).toContain('/projects/p/ivllm/nvhpc');
    // idempotency guard
    expect(script).toMatch(/if \[ ! -d.*nvhpc/);
  });

  it('sets NVHPC_ROOT and LD_LIBRARY_PATH with compat path first', () => {
    const script = renderSetupScript(base, '/tmp/remote.log');
    expect(script).toContain(
      'NVHPC_ROOT=/projects/p/ivllm/nvhpc/Linux_aarch64/26.3',
    );
    expect(script).toContain('$NVHPC_ROOT/cuda/12.9/compat');
    // compat must appear before lib64 in LD_LIBRARY_PATH
    const idx1 = script.indexOf('cuda/12.9/compat');
    const idx2 = script.indexOf('cuda/12.9/lib64');
    expect(idx1).toBeLessThan(idx2);
  });

  it('loads gcc-native/14.2 module before pip install', () => {
    const script = renderSetupScript(base, '/tmp/remote.log');
    expect(script).toContain('module load gcc-native/14.2');
  });

  it('creates versioned venv at /projects/p/ivllm/<version>', () => {
    const script = renderSetupScript(base, '/tmp/remote.log');
    expect(script).toContain('/projects/p/ivllm/0.19.1');
  });

  it('skips venv install if versioned dir already exists', () => {
    const script = renderSetupScript(base, '/tmp/remote.log');
    expect(script).toMatch(/if \[ ! -d.*\/projects\/p\/ivllm\/0\.19\.1/);
  });

  it('installs vllm using version-specific cu129 wheels', () => {
    const script = renderSetupScript(base, '/tmp/remote.log');
    expect(script).toContain('wheels.vllm.ai/0.19.1/cu129');
    expect(script).not.toContain('nightly');
    expect(script).not.toContain('cu130');
  });

  it('sets UV_CACHE_DIR to $LOCALDIR (per-user in-job scratch, not shared $PROJECTDIR)', () => {
    const script = renderSetupScript(base, '/tmp/remote.log');
    expect(script).toContain('UV_CACHE_DIR=$LOCALDIR/uv_cache');
    expect(script).not.toContain('UV_CACHE_DIR=$PROJECTDIR');
  });

  it('sets g+w on $PROJECTDIR/ivllm so all project members can create versioned venvs', () => {
    const script = renderSetupScript(base, '/tmp/remote.log');
    expect(script).toContain('chmod g+w /projects/p/ivllm');
  });

  it('sets g+w on $PROJECTDIR/hf so all project members can share cached model weights', () => {
    const script = renderSetupScript(base, '/tmp/remote.log');
    expect(script).toContain('chmod g+w /projects/p/hf');
  });

  it('installs exact vllm version', () => {
    const script = renderSetupScript(base, '/tmp/remote.log');
    expect(script).toContain('vllm==0.19.1');
  });

  it('does not reference singularity', () => {
    const script = renderSetupScript(base, '/tmp/remote.log');
    expect(script).not.toContain('singularity');
  });

  it('includes IVLLM_SETUP_SUCCESS marker', () => {
    const script = renderSetupScript(base, '/tmp/remote.log');
    expect(script).toContain('IVLLM_SETUP_SUCCESS');
  });

  it('does not use --pty flag', () => {
    const script = renderSetupScript(base, '/tmp/remote.log');
    expect(script).not.toContain('--pty');
  });

  it('does not use /tmp for temporary files (Isambard policy: use $LOCALDIR)', () => {
    const script = renderSetupScript(base, '/tmp/remote.log');
    expect(script).not.toContain('/tmp/nvhpc');
    expect(script).not.toMatch(/\btmp\b.*nvhpc/);
  });

  it('uses $LOCALDIR for temporary HPC SDK download and extraction', () => {
    const script = renderSetupScript(base, '/tmp/remote.log');
    expect(script).toContain('$LOCALDIR');
    expect(script).toMatch(
      /\$LOCALDIR.*nvhpc\.tar\.gz|nvhpc\.tar\.gz.*\$LOCALDIR/,
    );
  });

  it('cleans up the extracted HPC SDK installer directory after install', () => {
    const script = renderSetupScript(base, '/tmp/remote.log');
    expect(script).toMatch(/rm -rf.*nvhpc_2026_263_Linux_aarch64_cuda_multi/);
  });

  it('uses a different version when specified', () => {
    const script = renderSetupScript(
      {
        ...base,
        vllmVersion: '0.10.0',
        paths: makeSimplePaths(creds, '0.10.0'),
      },
      '/tmp/remote.log',
    );
    expect(script).toContain('/projects/p/ivllm/0.10.0');
    expect(script).toContain('vllm==0.10.0');
    expect(script).not.toContain('0.19.1');
  });
});

describe('parseJobId', () => {
  it('parses job ID from sbatch success output', () => {
    expect(parseJobId('Submitted batch job 12345\n')).toBe('12345');
  });

  it('parses job ID with no trailing newline', () => {
    expect(parseJobId('Submitted batch job 99999')).toBe('99999');
  });

  it('returns null for unrecognised output', () => {
    expect(parseJobId('sbatch: error: ...')).toBeNull();
  });
});

describe('parseJobState', () => {
  it('returns completed for COMPLETED state', () => {
    expect(parseJobState('COMPLETED')).toBe('completed');
  });

  it('returns running for RUNNING state', () => {
    expect(parseJobState('RUNNING')).toBe('running');
  });

  it('returns running for PENDING state', () => {
    expect(parseJobState('PENDING')).toBe('running');
  });

  it('returns failed for FAILED state', () => {
    expect(parseJobState('FAILED')).toBe('failed');
  });

  it('returns failed for TIMEOUT state', () => {
    expect(parseJobState('TIMEOUT')).toBe('failed');
  });

  it('returns failed for CANCELLED state', () => {
    expect(parseJobState('CANCELLED by 1000')).toBe('failed');
  });

  it('returns null for empty/unknown output', () => {
    expect(parseJobState('')).toBeNull();
  });
});
