import { writeFileSync, unlinkSync, existsSync } from 'fs';
import { join } from 'path';
import { tmpdir } from 'os';
import { loadCredentials, assertConfigured } from '../config.ts';
import { renderSetupScript } from '../templates/setup.ts';
import { makeRemoteOps } from '../remote-ops.ts';
import { ProcessState } from '../types.ts';
import { makeSimplePaths } from '../job.ts';

// TODO: --dry-run flag

/**
 *
 * @param args
 */
export async function cmdSetup(args: string[]): Promise<void> {
  // Handle help flag
  if (args.includes('--help') || args.includes('-h')) {
    console.log(`
Usage: ivllm setup <version>

Options:
  <version>             vLLM version to install (e.g. 0.19.1)
  --help, -h            Show this help message

Examples:
  ivllm setup 0.19.1
`);
    return;
  }

  const vllmVersion = args[0];
  if (!vllmVersion || vllmVersion.startsWith('--')) {
    console.error('Error: vLLM version is required.');
    console.error('Usage: ivllm setup <version>  (e.g. ivllm setup 0.19.1)');
    process.exit(1);
  }

  const config = loadCredentials();
  try {
    assertConfigured(config);
  } catch (e) {
    console.error('Error:', (e as Error).message);
    process.exit(1);
  }

  const ops = makeRemoteOps(config, 'real');
  const paths = makeSimplePaths(config, vllmVersion);
  const venvDir = paths.remoteProjectVllmVersionDir;
  const remoteSetupDir = `${paths.remoteHomeDir}/.config/ivllm/${vllmVersion}`;
  const remoteSetupScript = `${remoteSetupDir}/slurm.sh`;
  const remoteSetupLog = `${remoteSetupDir}/setup.log`;

  console.log(`=== ivllm setup (version ${__VERSION__}) ===`);
  console.log(`Login node  : ${config.loginHost}`);
  console.log(`vLLM        : ${vllmVersion}`);
  console.log(`Install dir : ${venvDir}`);
  console.log('');

  // ── 3. Build session state ────────────────────────────────────────────

  const sessionState = new ProcessState({
    sessionName: 'vllm install',
    vllmVersion,
    ops,
    paths,
  });

  // Pre-flight: check SSH connectivity
  ops.checkSSH();

  // Check if versioned venv already exists

  const { exitCode: venvCheck } = await ops.runRemote(`test -d ${venvDir}/bin`);
  if (venvCheck === 0) {
    console.log(`✓ vLLM ${vllmVersion} already installed at ${venvDir}`);
    console.log('  Delete the directory first to reinstall vllm.');
    console.log('  Running setup for other components.');
  }

  // Render and copy setup script to LOGIN
  const script = await renderSetupScript(sessionState, remoteSetupLog);
  const localTmp = join(tmpdir(), 'ivllm-setup.slurm.sh');
  writeFileSync(localTmp, script, 'utf-8');

  try {
    console.log(`Copying setup script to ${remoteSetupDir} on login node...`);
    await ops.runRemote(`mkdir -p ${remoteSetupDir}`, {
      env: [],
      silent: false,
    });
    await ops.copyFile(localTmp, remoteSetupScript);
    console.log('✓ Script copied');

    // Submit SLURM job
    console.log('Submitting SLURM setup job...');

    const remoteSrun = `srun \\
    --job-name="install_vllm" \\
    --nodes=1 \\
    --ntasks-per-node=1 \\
    --cpus-per-task=16 \\
    --gpus-per-node=1 \\
    --partition=interactive \\
    --reservation=interactive \\
    --interactive \\
    --mem=48G \\
    --time=01:00:00 \\
    --export=ALL \\
    bash "${remoteSetupScript}"
    `;

    const { exitCode: dlCode } = await ops.runRemote(remoteSrun, {
      env: [],
      silent: false,
    });

    if (dlCode !== 0) {
      console.error('✗ Setup job failed. Log output:');
      process.exit(1);
    }

    // Validate venv exists
    const { exitCode: finalCheck } = await ops.runRemote(
      `test -d ${paths.remoteProjectVllmVersionDir}/bin`,
    );
    if (finalCheck !== 0) {
      console.error(
        `✗ venv not found at ${paths.remoteProjectVllmVersionDir} after setup.`,
      );
      process.exit(1);
    }

    console.log(`✓ vLLM ${vllmVersion} installation complete`);
  } finally {
    if (existsSync(localTmp)) unlinkSync(localTmp);
  }
}
