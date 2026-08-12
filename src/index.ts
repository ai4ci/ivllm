#!/usr/bin/env bun
import { program } from 'commander';

import { getBackend } from './backends/backend-factory.ts';
import {
    loadCredentials,
    assertConfigured,
    saveCredentials,
} from './config.ts';
import { formatJobRow, formatJobTable, formatBenchStatus } from './utils.ts';
import type { CloseableEventEmitter } from './types.ts';
import { sleep } from 'bun';
import { isLocalPortInUse } from './local-ops.ts';
import path from 'path';
import fs from 'fs';

// Assign globally across Node.js/Browser using the universal globalThis object
const { version } = await import('../package.json');
(globalThis as any).__VERSION__ = version;

/**
 * CLI entry point — registers all commands with Commander and parses argv.
 *
 * Commands: `setup`, `cancel`, `config`, `connect`, `status`.
 */
async function main() {
    program.name('ivllm').version(version).description(`run llms on HPCs`);

    program
        .command('setup')
        .description('Install vLLM <version> on the HPC (one-off)')
        .argument('<version>', 'the vLLM version to install (e.g. 0.19.1)')
        .option(
            '--force',
            'Force reinstallation of the vLLM version even if it exists',
            false,
        )
        .action(cmdSetup);

    program
        .command('cancel')
        .description(
            `Cancel a running vLLM inference job.

Graceful cancel (default):
Writes "cancel" to the job's lockfile. The compute-side monitor detects the
request and shuts down vLLM cleanly, preserving logs and diagnostics.

Force cancel (--force):
Runs scancel on the SLURM job directly and updates the lockfile. Use this
when graceful shutdown fails or the monitor is unresponsive.`,
        )
        .argument(
            '<jobName>',
            'the short name of the job, e.g. qwen36, from `ivllm status`',
        )
        .option(
            '--force',
            'use slurm scancel directly instead of graceful cancel',
            false,
        )
        .option('--abort', 'abort the job capturing diagnostics', false)
        .action(cmdCancel);

    program
        .command('config')
        .description('configure user credentials for isambard')
        .option(
            '--login-host <host>',
            'SSH login node (e.g. XXXX.aip2.isambard)',
        )
        .option('--username <user>', 'HPC username (e.g. YYYY.XXXX)')
        .option('--project-dir <path>', 'HPC project dir (e.g. /projects/XXXX)')
        .option('--hf-token <token>', 'HuggingFace token for gated models')
        .action(cmdConfig);

    program
        .command('connect')
        .description(
            `Start or connect to a vLLM inference session.

If the job is already running, establishes an SSH tunnel.
If the job is stopped or failed, restarts it.
If the job doesn't exist, creates it and starts it.`,
        )
        .argument(
            '<jobName>',
            'the short name of the job, e.g. qwen36, from `ivllm status`',
        )
        .option(
            '--config <configFile>',
            'vLLM config YAML (required for first use)',
        )
        .option('--local-port [port]', 'Local port for API', '11434')
        .option(
            '--batch',
            'Submit to standard partition instead of interactive',
            false,
        )
        .option(
            '--time <duration>',
            'SLURM time limit as <hh:mm:ss>',
            '08:00:00',
        )
        .action(cmdConnect);

    program
        .command('status')
        .description('Show the status of llm jobs on the HPC')
        .argument(
            '[jobName]',
            'show status of specific job (if omitted, show all jobs)',
        )
        .action(cmdStatus);

    program
        .command('diagnostics')
        .description('Download diagnostics for a failed or crashed job')
        .argument('<jobName>', 'name of the job')
        .option('--out <path>', 'local destination directory')
        .action(cmdDiagnostics);

    const bench = program
        .command('bench')
        .description(
            '[experimental] Benchmark and compare vLLM configurations',
        );

    bench
        .command('submit')
        .description('Submit a set of vLLM configs as a benchmark comparison')
        .argument(
            '<comparison>',
            'name or config directory for this comparison run, e.g. qwen35',
        )
        .argument(
            '<configs...>',
            'one or more vLLM YAML config files with distinctive names, e.g. tp4-pp2.yaml, tp8.yaml, ... if non provided defaults to all yaml files in comparison directory',
        )
        .option('--time <duration>', 'SLURM time limit per job as <hh:mm:ss>')
        .action(cmdBenchSubmit);

    bench
        .command('status')
        .description('Check progress of a benchmark comparison')
        .argument(
            '<comparison>',
            'comparison name or config directory from `ivllm bench submit`',
        )
        .action(cmdBenchStatus);

    bench
        .command('results')
        .description('Fetch results of a completed benchmark comparison')
        .argument('<comparison>', 'comparison name or config directory')
        .argument(
            '[outDir]',
            'local directory to copy results into (defaults to `<comparison>/result`)',
        )
        .action(cmdBenchResults);

    await program.parseAsync(process.argv);
}

/**
 * Cancel command handler.
 *
 * Loads credentials, creates the backend, and delegates to
 * {@link Backend.requestCancel} for graceful or forced cancellation.
 * @param jobName — Job name to cancel
 * @param options — `{ force: boolean }` — force kill via scancel
 * @param options.force
 */
async function cmdCancel(
    jobName: string,
    options: { force: boolean; abort: boolean },
): Promise<void> {
    // Load and validate credentials
    const config = loadCredentials();
    let logWatcher: CloseableEventEmitter | null = null;

    assertConfigured(config);
    const backend = getBackend(config);
    await backend.requestCancel(jobName, options.force, options.abort);

    // 1. Global Intercept Cleanup Handler
    const cleanupAndExit = async () => {
        console.log('\n\n[Ctrl+C] close cancel monitor...');

        if (logWatcher) {
            await logWatcher.close();
        }
        process.exit(0);
    };

    // // Register intercept vectors immediately before doing any heavy operations
    process.once('SIGINT', cleanupAndExit);
    process.once('SIGTERM', cleanupAndExit);

    logWatcher = await backend.watchLog(jobName, '0');

    // tail the log until the shutdown is complete (or user cancels tail)
    while (
        !(await backend.isStopped(jobName)) &&
        (await logWatcher.isAlive())
    ) {
        await sleep(1000);
    }

    await logWatcher.close();
}

/**
 * Config command handler.
 *
 * Loads current credentials, applies any provided options, and saves.
 * @param options — Optional credential fields to update:
 *   `loginHost`, `username`, `projectDir`, `hfToken`
 * @param options.loginHost
 * @param options.username
 * @param options.projectDir
 * @param options.hfToken
 */
async function cmdConfig(options: {
    loginHost?: string;
    username?: string;
    projectDir?: string;
    hfToken?: string;
}): Promise<void> {
    const config = loadCredentials();
    if (options.loginHost) {
        config.loginHost = options.loginHost;
        saveCredentials(config);
        console.log('Configuration saved.');
    } else if (options.username) {
        config.username = options.username;
        saveCredentials(config);
        console.log('Configuration saved.');
    } else if (options.projectDir) {
        config.projectDir = options.projectDir;
        saveCredentials(config);
        console.log('Configuration saved.');
    } else if (options.hfToken) {
        config.hfToken = options.hfToken;
        saveCredentials(config);
        console.log('Configuration saved.');
    } else {
        console.log(JSON.stringify(config, null, 2));
    }
}

/**
 * Setup command handler.
 *
 * Loads credentials, creates the backend, and delegates to
 * {@link Backend.setup} to install vLLM on the HPC.
 * @param vllmVersion — Version string (e.g. `'0.19.1'`)
 * @param options — `{ force: boolean }` — force reinstall
 * @param options.force
 */
async function cmdSetup(
    vllmVersion: string,
    options: { force: boolean },
): Promise<void> {
    const config = loadCredentials();
    assertConfigured(config);
    const backend = getBackend(config);
    await backend.setup(vllmVersion, options.force);
}

/**
 * Status command handler.
 *
 * Shows lockfile status for a specific job or all jobs.
 * @param jobName — Optional job name; if omitted, lists all jobs
 */
async function cmdStatus(jobName?: string) {
    const config = loadCredentials();
    assertConfigured(config);
    const backend = getBackend(config);
    if (jobName) {
        // TODO: better formatting of a single job.
        console.log(formatJobRow(await backend.getJobStatus(jobName)));
    } else {
        console.log(formatJobTable(await backend.getAllJobStatus()));
    }
}

/**
 * Connect command handler.
 *
 * If the job is running, establishes an SSH tunnel.
 * If the job is stopped/failed, starts it.
 * If the job is starting up, tails logs until running.
 * @param jobName — Job name
 * @param options — `{ port, timeLimit, configFile }`
 * @param options.localPort
 * @param options.time
 * @param options.config
 * @param options.batch
 */
async function cmdConnect(
    jobName: string,
    options: {
        localPort: string;
        time: string;
        config?: string;
        batch: boolean;
    },
): Promise<void> {
    // TODO: rething the user experience here. Maybe better to
    // have a specific start command and defer to it if the job
    // is not already running, rather than try and start it.
    console.clear();
    const config = loadCredentials();
    assertConfigured(config);
    const backend = getBackend(config);
    const localPort = parseInt(options.localPort);

    if (await isLocalPortInUse(localPort)) {
        console.log(`port ${localPort} is in use`);
        console.log(`try: fuser -k ${localPort}/tcp`);
        process.exit(1);
    }

    // Track active background processes for the global Ctrl+C cleanup anchor
    let logWatcher: CloseableEventEmitter | null = null;
    let tunnel: CloseableEventEmitter | null = null;

    // 1. Global Intercept Cleanup Handler
    const cleanupAndExit = async () => {
        console.log(
            '\n\n[Ctrl+C] Cancelling connection routine. Cleaning up local resources...',
        );

        try {
            if (logWatcher) {
                console.log('--> Stopping background log tailing...');
                await logWatcher.close();
            }
            if (tunnel) {
                console.log('--> Disconnecting active SSH tunnel...');
                await tunnel.close();
            }
        } catch (err) {
            console.error('Error during cleanup:', err);
        } finally {
            process.exit(0); // Only exit once all promises have fully resolved
        }
    };

    // // Register intercept vectors immediately before doing any heavy operations
    process.once('SIGINT', cleanupAndExit);
    process.once('SIGTERM', cleanupAndExit);

    try {
        if (await backend.isStartable(jobName)) {
            console.log(
                `[connect] job '${jobName}' is not active. requesting start...`,
            );
            await backend.requestStart(
                jobName,
                options.time,
                options.batch,
                options.config,
            );
            // Refresh to grab the updated transitional state
        }

        let pollIntervalMs = 2000;

        // 3. Conditional Background Log Tailing Setup
        if (await backend.isStarting(jobName)) {
            console.log(
                `[connect] job '${jobName}' is starting. attaching background log watcher...`,
            );

            // Spawn the log stream. Because we do not 'await' its string completion natively,
            // it runs concurrently in the background while the polling loop executes below.
            logWatcher = await backend.watchLog(jobName, '0', true);

            // tail the log until the shutdown is complete
            while (await backend.isStarting(jobName)) {
                await sleep(pollIntervalMs);
            }

            // Job no longer starting
            await logWatcher.close();
            logWatcher = null;
        }

        // Process is no longer starting.
        const status = await backend.getStatusFlag(jobName);

        if (status === 'cancel') {
            throw new Error(
                `[connect] ERROR: job has been cancelled, and is in process of shutting down.`,
            );
        }

        if (status === 'abort') {
            throw new Error(
                `[connect] ERROR: job has been aborted, and is in process of shutting down.`,
            );
        }

        // Process is no longer starting. May have failed
        if (status === '' || status === 'failed' || status === 'stopped') {
            throw new Error(
                `[connect] ERROR: job startup failed: The cluster reported status [${status}].`,
            );
        }

        // If its not starting, stopped, or cancelling it should be running
        // Unless some race condition
        if (status !== 'running') {
            const lockfile = await backend
                .getJobStatus(jobName)
                .catch(() => null);
            const status = lockfile?.status || 'unknown';
            throw new Error(
                `[connect] ERROR: job is in an inconsistent state: The cluster reported status [${status}].`,
            );
        }

        // 6. Establish and lock the SSH Tunnel
        tunnel = await backend.connect(jobName, localPort);
        const lockfile = await backend.getJobStatus(jobName).catch(() => null);
        const model = lockfile?.model;
        const slurm = lockfile?.slurmJobId;
        console.log(`
Launching claude
================

ANTHROPIC_BASE_URL="http://localhost:${localPort}" \\
ANTHROPIC_API_KEY="" \\
ANTHROPIC_AUTH_TOKEN="ollama" \\
ANTHROPIC_MODEL="${model}" \\
claude --model "${model}"

N.b. Some models do not work with claude code and fail with unsupported role.

Docker sandbox (unverified)
===========================

Sandboxing: the sandbox must be able to access localhost port ${localPort}

# 1. Initialize the explicit sandbox named "local-vllm-claude" using your current directory (.)
docker sandbox create --name="local-vllm-claude" shell .

# 2. Add the host firewall exemption rule targeting only this named sandbox
docker sandbox network proxy local-vllm-claude --allow-host localhost:${localPort}

# 3. Start your environment inside that named sandbox with your variables passed inline
docker sandbox run local-vllm-claude \
-e ANTHROPIC_BASE_URL="http://host.docker.internal:${localPort}" \
-e ANTHROPIC_API_KEY="placeholder" \
-e ANTHROPIC_AUTH_TOKEN="ollama" \
-e ANTHROPIC_MODEL="${model}" \
claude --model "${model}"

Launching copilot cli
=====================

COPILOT_PROVIDER_BASE_URL="http://localhost:${localPort}/v1" \\
COPILOT_MODEL="${model}" \\
copilot

Launching pi
============

use the vllm model selector plugin:
pi install https://github.com/ai4ci/pi-vllm
run pi and select the model with "/vllm" command.

VLLM crashes
============

A crash in vllm running remotely will not appear here. You might see a message in your
agent saying, e.g.:

  Error: EngineCore encountered an issue. See stack trace (above) for the root cause.

That means vllm has died - you can use 'ivllm status' to confirm and you'll need
to press Ctrl-C to close this tunnel.

SUCCESSFULLY CONNECTED:
=======================

[connect] job ${jobName} is ${status}.
SSH tunnel connected - OpenAI api: http://localhost:${localPort}/v1 ...

Tunnel will stay open until you press Ctrl-C ...
`);

        // Block CLI execution loop natively while the tunnel stays open
        await new Promise<void>((resolve) => {
            tunnel!.on('close', () => {
                console.log(`
SSH forwarding tunnel disconnected.

${model} is still running and you can reconnect immediately with:
'ivllm connect ${jobName}'.

It will time-out by itself if no one else is using it.

If you think it has crashed or you want to stop it you should run:

  ivllm cancel ${jobName}
  ivllm cancel ${jobName} --force

or log in to Isambard and:

  squeue --reservation=interactive --me
  squeue --me
  scancel ${slurm || '<slurm-id>'}

`);
                resolve();
            });
        });
    } finally {
        // Clean up listeners to prevent memory leaks if everything finished smoothly
        process.off('SIGINT', cleanupAndExit);
        process.off('SIGTERM', cleanupAndExit);
    }
}

/**
 * Diagnostics command handler.
 *
 * Downloads remote diagnostics for a failed or crashed job.
 * @param jobName — Job name
 * @param options — Command options
 * @param options.out — Optional local destination path
 */
async function cmdDiagnostics(
    jobName: string,
    options: { out?: string },
): Promise<void> {
    const config = loadCredentials();
    assertConfigured(config);
    const backend = getBackend(config);
    const localPath = await backend.fetchDiagnostics(jobName, options.out);
    console.log(`✓ Diagnostics saved to: ${localPath}`);
}

/**
 * Benchmark job submit handler.
 *
 * Submits a benchmarking job
 * @param comparison - a comparison name or directory - used to identify the comparison job
 * @param configs — an array of vllm configurations to benchmark job-1.yaml, may be empty
 * @param options.time — A hours to run each comparison e.g. "02:00:00"
 */
async function cmdBenchSubmit(
    comparison: string,
    configs: string[],
    options: { time?: string },
): Promise<void> {
    const config = loadCredentials();
    if (configs.length == 0) {
        if (fs.existsSync(comparison)) {
            configs = fs.globSync('*.yaml', { cwd: comparison });
            if (configs.length == 0) {
                throw new Error(`no vllm yaml files found in ${comparison}`);
            } else {
                comparison = path.basename(comparison);
                console.log(`found ${configs.length} configs in ${comparison}`);
            }
        } else {
            throw new Error(
                'no vllm config files given and comparison is not a directory',
            );
        }
    }
    assertConfigured(config);
    const backend = getBackend(config);
    await backend.requestBenchmark(comparison, configs, options.time);
    console.log(`✓ Benchmaring run submitted: ${comparison}`);
}

/**
 * Benchmark job check status.
 *
 * Downloads status for a benchmarking.
 * @param comparison - a comparison name or directory - used to identify the comparison job
 *
 */
async function cmdBenchStatus(comparison: string): Promise<void> {
    const config = loadCredentials();
    assertConfigured(config);
    const backend = getBackend(config);
    const result = await backend.getBenchmarkStatus(comparison);
    comparison = path.basename(comparison);
    if (!result.complete) {
        console.log(`Comparison '${comparison}' is in progress:`);
        console.log(formatBenchStatus(result));
    } else {
        console.log(`Comparison '${comparison}' is complete:`);
        console.log(
            `use ivllm bench results ${comparison} <out-dir> to retrieve`,
        );
    }
}

/**
 * Benchmark job fetch results.
 *
 * Downloads status and results for a benchmarking job .
 * @param comparison - a comparison name or directory - used to identify the comparison job
 * @param outDir - a directory to write results to - defaults to <comparison>/result
 */
async function cmdBenchResults(
    comparison: string,
    outDir?: string,
): Promise<void> {
    const config = loadCredentials();
    assertConfigured(config);
    const backend = getBackend(config);
    if (!outDir) {
        outDir = path.join(comparison, 'result');
    }
    fs.mkdirSync(outDir);
    comparison = path.basename(comparison);
    const result = await backend.fetchBenchmarkResults(comparison, outDir);
    if (!result.ready) {
        console.log(`Comparison '${comparison}' is not finished yet:`);
        console.log(formatBenchStatus(result.status));
        process.exitCode = 1; // distinguishable from a real error for scripting
        return;
    }
    console.log(`✓ Results saved to: ${result.path}`);
}

main();
