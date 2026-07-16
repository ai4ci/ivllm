// tests/preload.ts
globalThis.__VERSION__ = '1.0.0-test';

import { beforeAll, afterAll } from 'bun:test';
import { mkdirSync, writeFileSync, rmSync, chmodSync } from 'node:fs';
import Bun from 'bun';

const SANDBOX_DIR = `${import.meta.dirname}/.hpc-sandbox`;
const WORKSPACE_DIR = `${SANDBOX_DIR}/workspace`;
const BIN_DIR = `${SANDBOX_DIR}/bin`;

let vllmServer: any;

beforeAll(async () => {
  console.log('🛠️  Initializing Complete HPC Bubblewrap Sandbox Engine...');

  // 1. Setup isolated directories
  mkdirSync(WORKSPACE_DIR, { recursive: true });
  mkdirSync(BIN_DIR, { recursive: true });

  // Expose these paths to your app logic so it knows where to "upload" files over SSH simulation
  process.env.HPC_MOCK_ACTIVE = 'true';
  process.env.HPC_WORKSPACE_PATH = WORKSPACE_DIR;

  // 2. Write a specialized 'sbatch' wrapper that calls Bubblewrap
  // This executes the script in a container that can only see vital system files and our fake bins
  const sbatchContent = `#!/usr/bin/env bash
    SCRIPT_NAME="$1"
    if [ -z "$SCRIPT_NAME" ]; then
        echo "sbatch: error: no script specified" >&2
        exit 1
        fi

        echo "Submitted batch job 998877"

        # Execute inside the bubblewrap sandbox
        bwrap \\
        --ro-bind /bin /bin \\
        --ro-bind /usr /usr \\
        --ro-bind /lib /lib \\
        --ro-bind /lib64 /lib64 \\
        --ro-bind /etc /etc \\
        --dev /dev \\
        --proc /proc \\
        --bind "${WORKSPACE_DIR}" /workspace \\
        --ro-bind "${BIN_DIR}" /hpc-bin \\
        --setenv PATH "/hpc-bin:/usr/local/bin:/usr/bin:/bin" \\
        --chdir /workspace \\
        bash "$SCRIPT_NAME"
        `;

  writeFileSync(`${BIN_DIR}/sbatch`, sbatchContent);
  chmodSync(`${BIN_DIR}/sbatch`, 0o755);

  // 3. Write a mock 'vllm' executable that forwards to our local background server
  // On an HPC, users call `vllm serve --model ...`. We can mock this command to make a curl lookup
  const vllmContent = `#!/usr/bin/env bash
COMMAND="$1"
if [ "$COMMAND" = "serve" ]; then
    echo "INFO: Starting mock vLLM engine..."
    # Ping the background Bun server to notify it that it was initialized
    curl -s -X POST http://localhost:8000/internal/register-model > /dev/null
    echo "INFO: vLLM Open-AI compatible server running at http://localhost:8000"
    # Keep running to simulate a long-lived cluster daemon process
    sleep infinity
fi
`;

  writeFileSync(`${BIN_DIR}/vllm`, vllmContent);
  chmodSync(`${BIN_DIR}/vllm`, 0o755);

  // Add the sandbox binaries to the local process env for standard test executions
  process.env.PATH = `${BIN_DIR}:${process.env.PATH}`;

  // 4. Boot Background Server to handle vLLM calls (e.g. tracking initialization status)
  vllmServer = Bun.serve({
    port: 8000,
    fetch(req) {
      const url = new URL(req.url);
      if (url.pathname === '/internal/register-model') {
        return new Response('OK');
      }
      if (url.pathname === '/v1/chat/completions') {
        return Bun.Response.json({
          choices: [{ message: { content: 'Mock API reply' } }],
        });
      }
      return new Response('Not Found', { status: 404 });
    },
  });
});

afterAll(() => {
  console.log('🧹 Tearing down HPC Sandbox environments...');
  if (vllmServer) vllmServer.stop();
  rmSync(SANDBOX_DIR, { recursive: true, force: true });
});
