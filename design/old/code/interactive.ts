import { loadCredentials, assertConfigured } from '../config.ts';
import { makeRemoteOps } from '../remote-ops.ts';
import { makeLocalOps } from '../local-ops.ts';
import { parseStartArgs } from '../job.ts';
import { runInferenceSession } from '../session-helper.ts';

/**
 *
 * @param args
 */
export async function cmdInteractive(args: string[]): Promise<void> {
  // Handle help flag
  const help = `
Usage: ivllm interactive <job> [options]

Options:
--config <file>       vLLM config YAML (contains model, parallelism and all serving options)
--local-port <n>      Local port to expose the API on (from ivllm config)
--gpus <n>            GPUs to request (overrides tensor-parallel-size × pipeline-parallel-size from YAML)
--time <hh:mm:ss>     SLURM time limit (default: 4:00:00)
--dry-run             Preview generated scripts and scp commands without running anything
--no-launch           Skip assistant launch menu, show config snippet only
--help, -h            Show this help message

Examples:
ivllm interactive my-job --config examples/qwen2.5-instruct.yaml
ivllm interactive my-job --gpus 4

Note: This runs vLLM directly via srun (no sbatch), so it blocks your terminal
until you type 'exit' or press Ctrl+C. Use 'ivllm start' for background job submission.
`;
  if (args.includes('--help') || args.includes('-h')) {
    console.log(help);
    return;
  }

  const credentials = loadCredentials();
  try {
    assertConfigured(credentials);
  } catch (e) {
    console.error('Error:', (e as Error).message);
    process.exit(1);
  }

  try {
    console.clear();
    const startArgs = await parseStartArgs(args, credentials);
    startArgs.isInteractive = true;
    const ops = makeRemoteOps(credentials, startArgs.dryRun);
    const localOps = makeLocalOps(startArgs.localPort, startArgs.dryRun);
    // Delegate to unified session pipeline (isInteractive: true → uses srun)
    await runInferenceSession(startArgs, ops, localOps);
  } catch (e) {
    console.error('Error:', (e as Error).message);
    console.log(help);
    process.exit(1);
  }
}
