import type {
    CloseableEventEmitter,
    LockfileV3,
    Credentials,
    EnvVarEntry,
} from '../types';
import { Backend } from './Backend';
import { SshRemoteOps } from '../ops/SshRemoteOps';
import path from 'path';
import fs from 'fs';
import { homedir } from 'os';
import { sleep } from 'bun';
import { isLocalPortInUse } from '../local-ops';

export class IsambardBareMetalBackend extends Backend {
    ops: SshRemoteOps;
    remoteEngine: string;
    envs: EnvVarEntry[];
    bootstrapped = false;

    constructor(creds: Credentials) {
        super(creds);
        this.ops = new SshRemoteOps(creds);
        this.remoteEngine = `${creds.projectDir}/engine`;
        this.envs = [{ key: 'IVLLM_PROJECTDIR', value: `${creds.projectDir}` }];
        if (creds.hfToken) {
            this.envs = this.envs.concat([
                { key: 'HF_TOKEN', value: `${creds.hfToken}` },
            ]);
        }
    }

    async bootstrap(): Promise<void> {
        await this.ops.checkSSH();
        if (!this.bootstrapped) {
            const currentDir = import.meta.dir;
            const enginePath = path.resolve(currentDir, '../engine');
            await this.ops.copyDirectory(
                enginePath,
                this.creds.projectDir,
                'up',
            );
            this.bootstrapped = true;
        }
    }

    async setup(version: string, force?: boolean): Promise<void> {
        await this.bootstrap();

        const { stdout, exitCode } = await this.ops.runRemote(
            `${this.remoteEngine}/ivllm-setup.sh -v "${version}"${force ? ' -f' : ''}`,
            { env: this.envs, silent: false },
        );

        if (exitCode !== 0)
            throw new Error(
                `setup request failed (exit ${exitCode}): ${stdout}`,
            );
    }

    async connect(
        job: string,
        localPort: number,
    ): Promise<CloseableEventEmitter> {
        await this.bootstrap();
        const lp = localPort;

        if (await isLocalPortInUse(lp)) {
            throw new Error(`Local port ${lp} is in use`);
        }

        let jobStatus: LockfileV3;

        for (;;) {
            jobStatus = await this.getJobStatus(job);
            const s = jobStatus.status;
            if (s == 'failed' || s == 'stopped' || s == 'cancel') {
                throw new Error(
                    `Could not connect to ${job} which is in state ${s}`,
                );
            }
            if (s == 'running') break;
            if (s == 'pending') console.log('Waiting for job to start');
            if (s == 'initialising') console.log('vLLM is starting up');
            sleep(10_000);
        }

        if (jobStatus.computeHostname) {
            return this.ops.spawnTunnel(
                lp,
                jobStatus.computeHostname,
                jobStatus.serverPort,
            );
        } else {
            throw new Error(
                `Could not connect to ${job} as there is no hostname in the lockfile`,
            );
        }
    }

    async requestCancel(job: string, force: boolean): Promise<void> {
        await this.bootstrap();
        const { stdout, exitCode } = await this.ops.runRemote(
            `${this.remoteEngine}/ivllm-cancel.sh -j "${job}"${force ? ' -f' : ''}`,
            { env: this.envs, silent: false },
        );

        if (exitCode !== 0)
            throw new Error(
                `cancel request failed job ${job} (exit ${exitCode}): ${stdout}`,
            );
    }

    async requestStart(
        job: string,
        maxTime: string = '08:00:00',
        monitor: boolean,
        config?: string,
    ): Promise<void> {
        await this.bootstrap();

        if (config) {
            if (fs.existsSync(config)) {
                const remoteConfig = `${this.creds.projectDir}/engine/${job}/vllm.yaml`;
                await this.ops.copyFile(config, remoteConfig);
            } else {
                throw new Error(`no configuration file found at: ${config}`);
            }
        }

        const { stdout, exitCode } = await this.ops.runRemote(
            `${this.remoteEngine}/ivllm-serve.sh -j "${job}" -t "${maxTime}"`,
            { env: this.envs, silent: false },
        );

        if (exitCode !== 0)
            throw new Error(
                `startup request failed for job ${job} (exit ${exitCode}): ${stdout}`,
            );

        if (monitor)
            await this.ops.runRemote(
                `${this.remoteEngine}/ivllm-show-log.sh -j "${job}" -m "[startup] Startup complete"`,
                { env: this.envs, silent: false },
            );
    }

    async getAllJobStatus(): Promise<LockfileV3[]> {
        await this.bootstrap();

        const { stdout, exitCode } = await this.ops.runRemote(
            `${this.remoteEngine}/ivllm-status.sh -p`,
            { env: this.envs, silent: true },
        );

        if (exitCode !== 0)
            throw new Error(
                `could not access job status (exit ${exitCode}): ${stdout}`,
            );

        let jobs: LockfileV3[] = [];
        try {
            const parsed = JSON.parse(stdout || '[]');
            if (Array.isArray(parsed)) {
                jobs = parsed
                    .map((j) => this.parseV3Lockfile(JSON.stringify(j)))
                    .filter((j): j is LockfileV3 => j !== null);
            }
        } catch {
            // ignore parse errors — show empty list
        }

        return jobs;
    }

    async watchLog(
        job: string,
        node?: string,
        until?: string,
    ): Promise<CloseableEventEmitter> {
        await this.bootstrap();

        return this.ops.runRemoteSync(
            `${this.remoteEngine}/ivllm-show-log.sh -j "${job}" -n "${node ?? '0'}"${until ? ` -m "${until}"` : ''}`,
            this.envs,
        );
    }

    async fetchDiagnostics(job: string, localDest?: string): Promise<string> {
        await this.bootstrap();

        const remoteDiagDir = `${this.remoteEngine}/diagnostics/${job}`;
        const targetDir =
            localDest ||
            path.join(homedir(), '.config', 'ivllm', 'diagnostics', job);

        console.log(`Downloading diagnostics for job '${job}'...`);
        await this.ops.copyDirectory(targetDir, remoteDiagDir, 'down');

        return targetDir;
    }
}
