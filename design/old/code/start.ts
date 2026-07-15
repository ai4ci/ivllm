import {} from 'fs';
import { loadCredentials, assertConfigured } from '../config.ts';
import { makeRemoteOps } from '../remote-ops.ts';
import { makeLocalOps } from '../local-ops.ts';
import { parseStartArgs } from '../job.ts';
import { runInferenceSession } from '../session-helper.ts';

/**
 *
 * @param args
 */
export async function cmdStart(args: string[]): Promise<void> {
  // Handle help flag
  const help = `
Usage: ivllm start <job> [options]

Options:
--config <file>       vLLM config YAML (contains model, parallelism and all serving options)
--local-port <n>      Local port to expose the API on (from ivllm config)
--gpus <n>            GPUs to request (overrides tensor-parallel-size × pipeline-parallel-size from YAML)
--time <hh:mm:ss>     SLURM time limit (default: 4:00:00)
--mock                Use mock vLLM server (no GPU needed -- for testing); requires --model
--dry-run             Preview generated scripts and scp commands without running anything
--no-launch           Skip assistant launch menu, show config snippet only
--create-cache        Submit model to slurm to build vllm caches and exit once vllm healthy.
--help, -h            Show this help message

Examples:
ivllm start my-job --config examples/qwen2.5-instruct.yaml
ivllm start test-job --mock --model Qwen/Qwen2.5-0.5B-Instruct --dry-run
`;
  if (args.includes('--help') || args.includes('-h')) {
    console.log(help);
    return;
  }

  const config = loadCredentials();
  try {
    assertConfigured(config);
  } catch (e) {
    console.error('Error:', (e as Error).message);
    process.exit(1);
  }

  let startArgs;
  try {
    console.clear();
    startArgs = await parseStartArgs(args, config);

    startArgs.isInteractive = false;
    const ops = makeRemoteOps(config, startArgs.dryRun);
    const localOps = makeLocalOps(startArgs.localPort, startArgs.dryRun);

    // Delegate to unified session pipeline
    await runInferenceSession(startArgs, ops, localOps);
  } catch (e) {
    console.error('Error:', (e as Error).message);
    console.log(help);
    process.exit(1);
  }
}
