import { describe, it, expect } from 'bun:test';
import { renderInferenceScript } from '../src/templates/inference.ts';
import {
  SessionState,
  type Credentials,
  type InferenceJobOptions,
  type ServeOptions,
} from '../src/types.ts';
import { writeFileSync } from 'fs';
import { parseStartArgs } from '../src/job.ts';
import { makeRemoteOps } from '../src/remote-ops.ts';
import { makePaths } from '../src/job.ts';
import { makeLocalOps } from '../src/local-ops.ts';

const creds: Credentials = {
  loginHost: 'test.example.com',
  username: 'test-user',
  projectDir: '/projects/p',
  defaultLocalPort: 11434,
  hfToken: 'HFTOKEN',
};

writeFileSync(
  '/tmp/test.yaml',
  `# Test vllm-config

model: Qwen/Qwen2.5-0.5B-Instruct
tensor-parallel-size: 1
max-model-len: 32768
# Native context: 32,768 tokens (full native context retained)
gpu-memory-utilization: 0.90
dtype: bfloat16
enable-auto-tool-choice: true
tool-call-parser: hermes
enable-prefix-caching: true
min-vllm-version: "0.19.1"
# min-vllm-version: ivllm checks this before submitting the job; stripped before passing to vLLM.
# Environment variables to aid startup/coordination
env:
  TEST_ENV: "test-value"
`,
);

const args: string[] = ['testJob', '--config', '/tmp/test.yaml', '--dry-run'];

const job: InferenceJobOptions = await parseStartArgs(args, creds);

const vllmVersion = '0.10.10';
const paths = makePaths(
  creds,
  job.jobName,
  job.configYaml.model,
  job.cacheKey,
  vllmVersion,
);

const base = new SessionState({
  startArgs: job,
  localOps: makeLocalOps(job.localPort, true),
  ops: makeRemoteOps(creds, 'dry-run'),
  vllmVersion: '0.10.10',
  paths: paths,
  sessionName: job.jobName,
});

// function updateState(
//   opts: Partial<InferenceJobOptions>,
//   ss: SessionState = base,
// ): SessionState {
//   ss.startArgs = {
//     ...ss.startArgs,
//     ...opts,
//   };
//   return ss;
// }
//
// function updateServe(
//   opts: Partial<ServeOptions>,
//   ss: SessionState = base,
// ): SessionState {
//   ss.startArgs.configYaml = {
//     ...ss.startArgs.configYaml,
//     ...opts,
//   };
//   return ss;
// }

function updateState(
  opts: Partial<InferenceJobOptions>,
  ss: SessionState = base,
): SessionState {
  const startArgs = {
    ...ss.startArgs,
    ...opts,
  };
  return { ...base, startArgs: startArgs };
}

function updateServe(
  opts: Partial<ServeOptions>,
  ss: SessionState = base,
): SessionState {
  const ss2 = new SessionState(ss);
  ss2.startArgs.configYaml = {
    ...ss2.startArgs.configYaml,
    ...opts,
  };
  return ss2;
}

describe('renderInferenceScript', () => {
  it('sets SBATCH job name', () => {
    expect(renderInferenceScript(base)).toContain(
      '\n#SBATCH --job-name=testJob',
    );
  });

  it('sets SBATCH GPU count', () => {
    expect(renderInferenceScript(updateState({ gpuCount: 4 }))).toContain(
      '\n#SBATCH --gpus=4',
    );
  });

  it('omits --exclusive for fractional GPU requests', () => {
    expect(renderInferenceScript(updateState({ gpuCount: 2 }))).not.toContain(
      '\n#SBATCH --exclusive',
    );
  });

  // it("scales --mem per GPU (120GB) for fractional requests", () => {
  //   expect(
  //     renderInferenceScript({ ...base, gpuCount: 1 }),
  //   ).toContain("#SBATCH --mem=120G");
  //   expect(
  //     renderInferenceScript({ ...base, gpuCount: 2 }),
  //   ).toContain("#SBATCH --mem=240G");
  //   expect(
  //     renderInferenceScript({ ...base, gpuCount: 3 }),
  //   ).toContain("#SBATCH --mem=360G");
  // });

  it('scales --cpus-per-task per GPU (64) for fractional requests', () => {
    expect(renderInferenceScript(updateState({ gpuCount: 2 }))).toContain(
      '\n#SBATCH --cpus-per-gpu=64',
    );
  });

  it('sets SBATCH time limit', () => {
    expect(renderInferenceScript(base)).toContain('\n#SBATCH --time=8:00:00');
  });

  it('activates the versioned venv from /projects/p', () => {
    expect(renderInferenceScript(base)).toContain(
      'source "/projects/p/ivllm/0.10.10/bin/activate"',
    );
  });

  it('sets CUDA_HOME and adds nvcc to PATH for Ray worker kernel compilation', () => {
    const script = renderInferenceScript(base);
    expect(script).toContain('CUDA_HOME=$NVHPC_ROOT/cuda/12.9');
    expect(script).toContain('PATH=$CUDA_HOME/bin:$PATH');
  });

  it('sets CPATH to include NVHPC math_libs headers for cublasLt.h', () => {
    const script = renderInferenceScript(base);
    expect(script).toContain('CPATH=$NVHPC_ROOT/math_libs/12.9/include:');
  });

  it('relocates $HOME to local tmpfs for fast JIT cache', () => {
    const script = renderInferenceScript(base);
    expect(script).toContain('HOME="$LOCALDIR/user_home"');
    expect(script).toContain('mkdir -p "$HOME"');
  });

  it('does not use $SCRATCHDIR symlinks for JIT caches anymore', () => {
    const script = renderInferenceScript(base);
    expect(script).not.toContain('$SCRATCHDIR/');
    expect(script).not.toContain('ln -sfn "$FLASHINFER_JIT_CACHE_DIR"');
    expect(script).not.toContain('ln -sfn "$DG_JIT_CACHE_DIR"');
    expect(script).not.toContain('ln -sfn "$TRITON_CACHE_DIR"');
    expect(script).not.toContain('ln -sfn "$TORCHINDUCTOR_CACHE_DIR"');
    expect(script).not.toContain('ln -sfn "$VLLM_CACHE_ROOT"');
  });

  it('saves JIT cache tarball to shared storage after health check passes', () => {
    const script = renderInferenceScript(base);
    expect(script).toContain('JIT CACHE SAVE');
    expect(script).toContain('tar czf');
    expect(script).toContain('CACHE_FILE');
    expect(script).toContain('-C "$LOCALDIR" user_home');
    expect(script).toContain('disown');
  });

  it('runs cache save in background after vLLM is healthy', () => {
    const script = renderInferenceScript(base);
    const healthIdx = script.indexOf('vLLM is ready');
    const saveIdx = script.indexOf('JIT CACHE SAVE');
    expect(healthIdx).toBeGreaterThan(-1);
    expect(saveIdx).toBeGreaterThan(-1);
    expect(saveIdx).toBeGreaterThan(healthIdx);
  });

  it('renders user env vars as export lines in single-node script', () => {
    const st = updateServe({
      env: [
        { key: 'VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS', value: '1' },
        { key: 'VLLM_USE_DEEP_GEMM_FP8', value: '1' },
      ],
    });
    const script = renderInferenceScript(st);
    expect(script).toContain('export VLLM_USE_DEEP_GEMM_FP8="1"');
  });

  it('does not render env exports when envVars is empty (single-node)', () => {
    const script = renderInferenceScript(updateServe({ env: [] }));
    // Should not have a "User-supplied environment variables" comment or bare exports
    expect(script).not.toContain('# User-supplied environment variables');
  });

  it('does not render env exports when envVars is empty (multi-node)', () => {
    const script = renderInferenceScript(
      updateServe(
        {
          env: [],
        },
        updateState({ gpuCount: 8 }),
      ),
    );
    expect(script).not.toContain('# User-supplied environment variables');
  });

  it('sets CC=gcc and CXX=g++ for JIT compilation with gcc-native module', () => {
    const script = renderInferenceScript(base);
    expect(script).toContain('export CC=gcc');
    expect(script).toContain('export CXX=g++');
  });

  it('loads gcc-native module for C++20 host compiler support', () => {
    const script = renderInferenceScript(base);
    expect(script).toContain('module load brics/nccl gcc-native');
  });

  it('sets LD_LIBRARY_PATH with cuda/12.9/compat first', () => {
    const script = renderInferenceScript(base);
    expect(script).toContain('$NVHPC_ROOT/cuda/12.9/compat');
    const idxCompat = script.indexOf('cuda/12.9/compat');
    const idxLib64 = script.indexOf('cuda/12.9/lib64');
    expect(idxCompat).toBeLessThan(idxLib64);
  });

  it('does not reference singularity', () => {
    expect(renderInferenceScript(base)).not.toContain('singularity');
  });

  it('does not reference cu130', () => {
    expect(renderInferenceScript(base)).not.toContain('cu130');
  });

  it('sets HF_HOME', () => {
    expect(renderInferenceScript(base)).toContain('HF_HOME="/projects/p/hf"');
  });

  it('sets umask 0002 for shared group-writable files', () => {
    expect(renderInferenceScript(base)).toContain('umask 0002');
  });

  it('sets HF_HUB_OFFLINE=1 to prevent API calls when model is already cached', () => {
    expect(renderInferenceScript(base)).toContain('export HF_HUB_OFFLINE=1');
  });

  it('changes into the job work directory before starting vllm serve', () => {
    const script = renderInferenceScript(base);
    expect(script).toContain('cd "$WORK_DIR"');
    const idxCd = script.indexOf('cd "$WORK_DIR"');
    const idxServe = script.indexOf('vllm serve');
    expect(idxCd).toBeLessThan(idxServe);
  });

  it('single-node: trap on_exit EXIT is not followed by prose text on the same line', () => {
    // Regression: the exit trap block was concatenated with a comment fragment,
    // causing bash to treat comment words as invalid signal names.
    const script = renderInferenceScript(base);
    const trapLine = script
      .split('\n')
      .find((l) => l.trimStart().startsWith('trap on_exit EXIT'));
    expect(trapLine).toBeDefined();
    expect(trapLine!.trim()).toBe('trap on_exit EXIT');
  });

  it('serves the correct model', () => {
    expect(renderInferenceScript(base)).toContain('Qwen/Qwen2.5-0.5B-Instruct');
  });

  it('uses the correct server port', () => {
    expect(renderInferenceScript(base)).toContain('--port 8000');
  });

  it('uses vllm serve --config (no model positional arg on command line)', () => {
    const script = renderInferenceScript(base);
    expect(script).toContain('vllm serve');
    expect(script).toContain('--config "$VLLM_CONFIG"');
    // model positional on the vllm CLI would conflict with YAML; not present
    expect(script).not.toContain('vllm serve Qwen');
  });

  it('does not pass --tensor-parallel-size on command line (comes from YAML config)', () => {
    expect(renderInferenceScript(base)).not.toContain('--tensor-parallel-size');
  });

  it('includes served-model-name flags for the model and job name', () => {
    const script = renderInferenceScript(base);
    // argparse collects multiple positional args from the space-separated list
    expect(script).toContain(
      '--served-model-name "Qwen/Qwen2.5-0.5B-Instruct" "qwen2.5-0.5b-instruct" "default" "testJob"',
    );
  });

  it('references the vllm config file from workDir', () => {
    expect(renderInferenceScript(base)).toContain(
      '/home/p/test-user/testJob/testJob.yaml',
    );
  });

  it('writes initialising status to job_details.json', () => {
    expect(renderInferenceScript(base)).toContain('"initialising"');
  });

  it('writes compute hostname to job_details.json', () => {
    expect(renderInferenceScript(base)).toContain('compute_hostname');
  });

  it('writes SLURM job ID to job_details.json', () => {
    expect(renderInferenceScript(base)).toContain('SLURM_JOB_ID');
  });

  it('updates status to running after health check passes', () => {
    const script = renderInferenceScript(base);
    expect(script).toContain('"running"');
  });

  it('updates status to failed when vLLM process dies during startup', () => {
    expect(renderInferenceScript(base)).toContain('"failed"');
  });

  it('waits indefinitely for health rather than enforcing a startup timeout', () => {
    const script = renderInferenceScript(base);
    expect(script).not.toContain('MAX_WAIT=');
    expect(script).not.toContain('Timed out waiting for vLLM');
    expect(script).not.toContain('"timeout"');
  });

  it('polls /health endpoint on localhost', () => {
    expect(renderInferenceScript(base)).toContain('localhost:8000/health');
  });

  it('does not use --pty flag', () => {
    expect(renderInferenceScript(base)).not.toContain('--pty');
  });

  it('respects a different server port', () => {
    const script = renderInferenceScript(updateState({ serverPort: 9000 }));
    expect(script).toContain('--port 9000');
    expect(script).toContain('localhost:9000/health');
  });

  it('respects a different gpu count in SBATCH directive', () => {
    const script = renderInferenceScript(updateState({ gpuCount: 8 }));
    expect(script).toContain('#SBATCH --nodes=2');
    expect(script).toContain('#SBATCH --gpus-per-node=4');
  });
});

const multiNodeBase = updateState(
  {
    gpuCount: 8,
  },
  base,
);

describe('renderInferenceScript (multi-node)', () => {
  it('sets --nodes=2 in SBATCH for 2-node job', () => {
    expect(renderInferenceScript(multiNodeBase)).toContain(
      '\n#SBATCH --nodes=2',
    );
  });

  it('sets umask 0002 for shared group-writable files', () => {
    expect(renderInferenceScript(multiNodeBase)).toContain('umask 0002');
  });

  it('requests GPUs per node in SBATCH for multi-node overlap compatibility', () => {
    const script = renderInferenceScript(multiNodeBase);
    expect(script).toContain('\n#SBATCH --gpus-per-node=4');
    expect(script).not.toContain('\n#SBATCH --gpus=8');
  });

  it('starts Ray head node', () => {
    const script = renderInferenceScript(multiNodeBase);
    expect(script).toMatch(/ray\s+start[\s\S]*--block[\s\S]*--head/);
  });

  it('starts Ray worker nodes via srun', () => {
    const script = renderInferenceScript(multiNodeBase);
    expect(script).toMatch(/ray\s+start[\s\S]*--block[\s\S]*--address/);
  });

  it('requests full node memory in SBATCH', () => {
    expect(renderInferenceScript(multiNodeBase)).toContain('\n#SBATCH --mem=0');
  });

  it('requests full node memory for all multi-node srun steps', () => {
    const script = renderInferenceScript(multiNodeBase);
    const memRequests = script.match(/--mem=0/g) ?? [];
    expect(memRequests.length).toBeGreaterThanOrEqual(5); // SBATCH + head + worker + status + serve
  });

  it('installs an EXIT trap so diagnostics are still collected after startup failures', () => {
    const script = renderInferenceScript(multiNodeBase);
    expect(script).toContain('on_exit()');
    expect(script).toContain('trap on_exit EXIT');
  });

  it('collects exit diagnostics explicitly before the scripted startup-failure exit', () => {
    const script = renderInferenceScript(multiNodeBase);
    expect(script).toContain('finalize_and_exit 1 "startup failure"');
    expect(script).toContain('collect_exit_diagnostics()');
  });

  it('wraps ray start commands in bash -c to guarantee venv PATH on compute nodes', () => {
    const script = renderInferenceScript(multiNodeBase);
    expect(script).toContain('bash -c "source');
    expect(script).not.toContain('env VLLM_HOST_IP');
  });

  it('sources the venv inside each bash -c ray start call', () => {
    const script = renderInferenceScript(multiNodeBase);
    // bash -c blocks may have leading whitespace/newlines before 'source'
    expect(script).toContain('bash -c "');
    const activateCount = (
      script.match(/bash -c "[\s\S]*?source[\s\S]*?\/bin\/activate/g) ?? []
    ).length;
    expect(activateCount).toBeGreaterThanOrEqual(3); // head, worker, vllm serve (plus ray status)
  });

  it('sets VLLM_HOST_IP inside bash -c for ray head', () => {
    const script = renderInferenceScript(multiNodeBase);
    expect(script).toContain('VLLM_HOST_IP=$HEAD_NODE_IP');
    expect(script).toContain('ray start');
    expect(script).toContain('--head');
  });

  it('sets VLLM_HOST_IP inside bash -c for ray workers', () => {
    const script = renderInferenceScript(multiNodeBase);
    expect(script).toContain('VLLM_HOST_IP=$WORKER_IP');
    expect(script).toContain('ray start');
    expect(script).toContain('--address');
  });

  it('runs vllm serve with --distributed-executor-backend ray', () => {
    expect(renderInferenceScript(multiNodeBase)).toContain(
      '--distributed-executor-backend ray',
    );
  });

  it('runs vllm serve via srun --overlap on the head node', () => {
    const script = renderInferenceScript(multiNodeBase);
    expect(script).toContain('srun --overlap');
  });

  it('changes into the job work directory before multi-node vllm serve', () => {
    const script = renderInferenceScript(multiNodeBase);
    expect(script).toContain('bash -c "\n');
    expect(script).toContain(`cd ${multiNodeBase.paths.remoteJobDir}`);
  });

  it('uses HEAD_NODE as the compute_hostname', () => {
    const script = renderInferenceScript(multiNodeBase);
    expect(script).toContain('HEAD_NODE');
    expect(script).toContain('compute_hostname');
  });

  it('loads the brics/nccl and gcc-native modules', () => {
    expect(renderInferenceScript(multiNodeBase)).toContain(
      'module load brics/nccl gcc-native',
    );
  });

  it('sets NVHPC_ROOT and LD_LIBRARY_PATH preamble before ray start', () => {
    const script = renderInferenceScript(multiNodeBase);
    expect(script).toContain(
      'NVHPC_ROOT=/projects/p/ivllm/nvhpc/Linux_aarch64/26.3',
    );
    expect(script).toContain('$NVHPC_ROOT/cuda/12.9/compat');
    // preamble must appear before ray start
    const idxNvhpc = script.indexOf('NVHPC_ROOT=');
    const idxRay = script.indexOf('ray start');
    expect(idxNvhpc).toBeLessThan(idxRay);
  });

  it('does not set deprecated Ray env vars removed in vLLM 0.19.1', () => {
    const script = renderInferenceScript(multiNodeBase);
    expect(script).not.toContain('VLLM_USE_RAY_SPMD_WORKER');
    expect(script).not.toContain('VLLM_USE_RAY_COMPILED_DAG');
    expect(script).not.toContain('VLLM_USE_RAY_SPMD_HEAD');
  });

  it('sets NCCL_CROSS_NIC=1 for multi-node NCCL comms', () => {
    const script = renderInferenceScript(multiNodeBase);
    expect(script).toContain('NCCL_CROSS_NIC=1');
  });

  it('sets VLLM_SKIP_CUSTOM_ALL_REDUCE=1 to bypass P2P and symmetric memory handshaking across nodes', () => {
    const script = renderInferenceScript(multiNodeBase);
    expect(script).toContain('export VLLM_SKIP_CUSTOM_ALL_REDUCE=1');
  });

  it('single-node template is unchanged for nodeCount=1', () => {
    const script = renderInferenceScript(base);
    expect(script).not.toContain('ray start');
    expect(script).not.toContain('--distributed-executor-backend');
  });
});
