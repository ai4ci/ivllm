import type { SessionState, RemoteMonitor, JobDetails } from './types.ts';

import { createInterface } from 'readline';
import { shutdown, sleep, timestamp } from './session-helper.ts';

import { parseJobDetails } from './job.ts';
import { pollJobStatus, getSlurmQueueState } from './slurm.ts';
import { printSlurmLog } from './session-helper.ts';

import { launchAssistant } from './commands/agent.ts';

const HEARTBEAT_INTERVAL_MS = 15_000;
const POLL_INTERVAL_MS = 5_000;
const SLURM_POLL_INTERVAL_MS = 60_000;

export const detachSession: RemoteMonitor = {
  start: async (sessionState: SessionState) => {
    console.log(
      `\nUnmonitored batch job submitted: to cancel: ivllm stop ${sessionState.startArgs.jobName}...\n`,
    );
    return;
  },
};

/**
 *
 * @param sessionState
 * @param startArgs
 * @param opts
 */
export const monitorSession: RemoteMonitor = {
  start: async (sessionState: SessionState) => {
    const opts = sessionState.startArgs!;
    const { model, maxModelLen, enableAutoToolChoice, enableReasoning } =
      opts.configYaml;
    // const jobName = sessionState.startArgs.jobName;

    console.log("\nMonitoring job status (Ctrl+C or type 'exit' to stop)...\n");

    const rl = createInterface({ input: process.stdin, terminal: false });
    rl.on('line', (line) => {
      if (line.trim().toLowerCase() === 'exit') {
        rl.close();
        shutdown(sessionState, 'user requested exit');
      }
    });

    let lastStatus = 'pending';
    let lastSlurmQueueState = '';
    let logLineOffset = 0;
    let lastSlurmPollTime = 0;

    while (!sessionState.shuttingDown) {
      await sleep(POLL_INTERVAL_MS);
      const { stdout } = await sessionState.ops.runRemote(
        `cat ${sessionState.paths.remoteJobLockFile} 2>/dev/null`,
      );
      const details = parseJobDetails(stdout);

      if (!details) {
        if (Date.now() - lastSlurmPollTime >= SLURM_POLL_INTERVAL_MS) {
          lastSlurmPollTime = Date.now();
          const slurmState = await pollJobStatus(
            sessionState.ops,
            sessionState.slurmJobId!,
          );
          if (slurmState === 'failed') {
            await printSlurmLog(sessionState);
            // await printCrashDiagnostics(sessionState);
            await shutdown(sessionState, 'SLURM job failed unexpectedly', 1);
            return;
          }
        }
        continue;
      }

      if (details.status === 'pending') {
        if (Date.now() - lastSlurmPollTime >= SLURM_POLL_INTERVAL_MS) {
          lastSlurmPollTime = Date.now();
          const queueState = await getSlurmQueueState(
            sessionState.ops,
            sessionState.slurmJobId!,
          );
          if (queueState) {
            const msg =
              queueState.state === 'PENDING'
                ? `  [${timestamp()}] Waiting in SLURM queue (${queueState.reason})`
                : `  [${timestamp()}] SLURM state: ${queueState.state}`;
            if (msg !== lastSlurmQueueState) {
              console.log(msg);
              lastSlurmQueueState = msg;
            }
          }
        }
      }

      if (details.status !== lastStatus) {
        if (details.status === 'initialising') {
          console.log(
            `  [${timestamp()}] Job allocated — vLLM is starting up...`,
          );
        } else if (details.status !== 'pending') {
          console.log(`  [${timestamp()}] Status: ${details.status}`);
        }
        lastStatus = details.status;
      }

      if (!opts.isInteractive && details.status === 'initialising') {
        const slurmLogPath = sessionState.paths.remoteJobLogFile;
        const { stdout: newLines } = await sessionState.ops.runRemote(
          `tail -n +${logLineOffset + 1} ${slurmLogPath} 2>/dev/null`,
        );
        if (newLines.trim()) {
          const lines = newLines.split('\n').filter((l) => l.trim());
          for (const line of lines) {
            console.log(`  | ${line}`);
          }
          logLineOffset += lines.length;
        }
      }

      if (details.status === 'failed' || details.status === 'timeout') {
        if (details.error) console.error(`  Error: ${details.error}`);
        await printSlurmLog(sessionState);
        // await printCrashDiagnostics(sessionState);
        await shutdown(sessionState, `vLLM ${details.status}`, 1);
        return;
      }

      if (details.status === 'running') {
        rl.close();
        await onRunning(details);
        return;
      }
    }

    /**
     *
     * @param details
     */
    async function onRunning(details: JobDetails): Promise<void> {
      const computeHost = details.compute_hostname!;
      console.log(`\n✓ vLLM is running on ${computeHost}:${opts.serverPort}`);

      const tunnel = sessionState.ops.spawnTunnel(
        opts.localPort,
        computeHost,
        opts.serverPort,
      );
      sessionState.tunnel = tunnel;
      tunnel.on('exit', (code) => {
        if (!sessionState.shuttingDown)
          shutdown(
            sessionState,
            `SSH tunnel exited unexpectedly (code ${code})`,
            1,
          );
      });

      await sleep(2000);

      console.log(
        `\n🚀 OpenAI API endpoint: http://localhost:${opts.localPort}/v1`,
      );
      console.log(`   Model: ${details.model ?? model}`);

      const heartbeatTimer = setInterval(async () => {
        let alive = false;
        let tests = 20;
        while (!alive && tests > 0) {
          alive =
            alive ||
            (await sessionState.localOps.checkLocalHealth(opts.localPort));
          tests -= 1;
          if (!alive) {
            console.warn(`Dropped a heartbeat. (retries: ${tests})`);
            await sleep(20_000);
          }
        }

        if (!alive) {
          if (!sessionState.shuttingDown) {
            console.error(`\nHeartbeat failed`);
            await printSlurmLog(sessionState);
            // await printCrashDiagnostics(sessionState);
            shutdown(sessionState, 'vLLM heartbeat failed', 1);
          }
        }
      }, HEARTBEAT_INTERVAL_MS);

      sessionState.heartbeatTimer = heartbeatTimer;

      await launchAssistant({
        model: details.model ?? model,
        localPort: opts.localPort,
        maxModelLen,
        toolCall: enableAutoToolChoice,
        reasoning: enableReasoning,
        shutdown: (reason: string, exitCode = 0) =>
          shutdown(sessionState, reason, exitCode),
      });
    }
  },
};
