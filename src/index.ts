#!/usr/bin/env bun
import { program } from 'commander';

import { IsambardBareMetalBackend } from './backends/IsambardBareMetalBackend.ts';
import { loadCredentials, assertConfigured, saveConfig } from './config.ts';
import { formatJobRow, formatJobTable } from './utils.ts';

// Assign globally across Node.js/Browser using the universal globalThis object
const { version } = await import('../package.json');
(globalThis as any).__VERSION__ = version;

async function main() {
    program.name('ivllm').version(version).description(`run llms on HPCs`);

    const setup = program
        .command('setup')
        .description('Install vLLM <version> on the HPC (one-off)')
        .argument('<version>', 'the vLLM version to install (e.g. 0.19.1)')
        .option(
            '--force',
            'Force reinstallation of the vLLM version even if it exists',
            false,
        )
        .action(cmdSetup);

    const cancel = program
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
            'Use slurm scancel directly instead of graceful cancel',
            false,
        )
        .action(cmdCancel);

    const config = program
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

    const connect = program
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
        .option('--dry-run', 'Preview without connecting to HPC', false)
        .action(cmdConnect);

    const status = program
        .command('status')
        .description('Show the status of llm jobs on the HPC')
        .argument(
            '[jobName]',
            'show status of specific job (if omitted, show all jobs)',
        )
        .action(cmdStatus);

    await program.parseAsync(process.argv);
}

async function cmdCancel(jobName: string, force: boolean): Promise<void> {
    // Load and validate credentials
    const config = loadCredentials();
    assertConfigured(config);
    const backend = new IsambardBareMetalBackend(config);
    backend.requestCancel(jobName, force);
}

async function cmdConfig(
    host?: string,
    user?: string,
    path?: string,
    token?: string,
): Promise<void> {
    const config = loadCredentials();
    if (host) config.loginHost = host!;
    if (user) config.username = user!;
    if (path) config.projectDir = path!;
    if (token) config.hfToken = token!;
    saveConfig(config);
    console.log('Configuration saved.');
}

async function cmdSetup(vllmVersion: string, force: boolean): Promise<void> {
    const config = loadCredentials();
    assertConfigured(config);
    const backend = new IsambardBareMetalBackend(config);
    backend.setup(vllmVersion, force);
}

async function cmdStatus(jobName?: string) {
    const config = loadCredentials();
    assertConfigured(config);
    const backend = new IsambardBareMetalBackend(config);
    if (jobName) {
        // TODO: better formatting of a single job.
        formatJobRow(await backend.getJobStatus(jobName));
    } else {
        formatJobTable(await backend.getAllJobStatus());
    }
}

async function cmdConnect(
    jobName: string,
    port: string,
    timeLimit: string,
    configFile?: string,
): Promise<void> {
    // TODO: rething the user experience here. Maybe better to
    // have a specific start command and defer to it if the job
    // is not already running, rather than try and start it.
    const config = loadCredentials();
    assertConfigured(config);
    const backend = new IsambardBareMetalBackend(config);
    const localPort = parseInt(port);

    if (!(await backend.isRunning(jobName))) {
        if (await backend.isStartable(jobName)) {
            await backend.requestStart(jobName, timeLimit, true, configFile);
        } else {
            if (await backend.isStarting(jobName)) {
                await backend.watchLog(
                    jobName,
                    '0',
                    '[startup] Startup complete',
                );
            } else {
                throw new Error(
                    `job ${jobName} is not in a startable state (maybe it is in the middle of shutting down)`,
                );
            }
        }
    }

    if (await backend.isRunning(jobName)) {
        await backend.connect(jobName, localPort);
    } else {
        throw new Error(
            `job ${jobName} was not running (after attempted start)`,
        );
    }
}

main();
