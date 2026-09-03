import type {
    CloseableEventEmitter,
    LockfileV3,
    Credentials,
    EnvVarEntry,
    BenchmarkStatus,
} from '../types';
import { Backend } from './Backend';
import { SshRemoteOps } from '../ops/SshRemoteOps';
import path from 'path';
import fs from 'fs';
import { homedir } from 'os';
import { sleep } from 'bun';
import { isLocalPortInUse } from '../local-ops';
// import { compareVersions } from '../utils';

export class IsambardBareMetalBackend extends Backend {
    ops: SshRemoteOps;
    envs: EnvVarEntry[];
    bootstrapped = false;
    remoteHome?: string;

    constructor(creds: Credentials) {
        super(creds);
        this.ops = new SshRemoteOps(creds);
        this.envs = [{ key: 'IVLLM_PROJECTDIR', value: `${creds.projectDir}` }];
        if (creds.hfToken) {
            this.envs = this.envs.concat([
                { key: 'HF_TOKEN', value: `${creds.hfToken}` },
            ]);
        }
    }

    async bootstrap(): Promise<void> {
        await this.ops.checkSSH();
        if (this.bootstrapped) return;

        const currentDir = import.meta.dir;
        const enginePath = path.resolve(currentDir, '../engine');
        const remoteEngine = await this.getRemoteEngine();
        await this.ops.copyDirectory(
            `${enginePath}/`,
            `${remoteEngine}/`,
            'up',
        );

        // DISABLED: script directory user specific so version checks not needed.
        // now will always rsync but overhead to this is quite low given number of files
        //         const localVersion = (globalThis as any).__VERSION__ as string;
        //         const remoteVersion = await this.getRemoteEngineVersion();
        //
        //         if (remoteVersion && compareVersions(localVersion, remoteVersion) < 0) {
        //             throw new Error(`
        // This ivllm client is version ${localVersion}, but the engine deployed at
        // ${this.creds.projectDir} is version ${remoteVersion}.
        // Please upgrade your local ivllm install to ${remoteVersion} or later before continuing.
        // (i.e. do a git pull)
        // `);
        //         }
        //
        //         if (
        //             !remoteVersion ||
        //             compareVersions(localVersion, remoteVersion) > 0
        //         ) {
        //             const currentDir = import.meta.dir;
        //             const enginePath = path.resolve(currentDir, '../engine');
        //             const remoteEngine = await this.getRemoteEngine();
        //             await this.ops.copyDirectory(
        //                 `${enginePath}/`,
        //                 `${remoteEngine}/`,
        //                 'up',
        //             );
        //             // copy contents
        //             await this.setRemoteEngineVersion(localVersion);
        //         }

        this.bootstrapped = true;
    }

    async setup(version: string, force?: boolean): Promise<void> {
        await this.bootstrap();
        const remoteEngine = await this.getRemoteEngine();

        const { stdout, exitCode } = await this.ops.runRemote(
            `${remoteEngine}/ivllm-setup.sh -v "${version}"${force ? ' -f' : ''}`,
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
            if (s == 'warmup') console.log('vLLM is warming up');
            await sleep(10_000);
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

    override async getStatusFlag(job: string): Promise<string> {
        const remoteEngine = await this.getRemoteEngine();
        const out = await this.ops.runRemote(
            `bash -c "source ${remoteEngine}/lib/utils.sh; get_job_status_setting '${job}' '.status'"`,
            { env: this.envs, silent: true },
        );
        return out.stdout;
    }

    async requestCancel(
        job: string,
        force: boolean,
        abort: boolean,
    ): Promise<void> {
        const remoteEngine = await this.getRemoteEngine();
        await this.bootstrap();
        const { stdout, exitCode } = await this.ops.runRemote(
            `${remoteEngine}/ivllm-cancel.sh -j "${job}"${force ? ' -f' : ''}${abort ? ' -a' : ''}`,
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
        batch: boolean,
        config?: string,
    ): Promise<void> {
        await this.bootstrap();
        const remoteEngine = await this.getRemoteEngine();

        if (config) {
            if (fs.existsSync(config)) {
                const remoteConfig = `${this.creds.projectDir}/engine/jobs/${job}/vllm.yaml`;
                console.log(`uploading config ${config} to ${remoteConfig}`);
                await this.ops.copyFile(config, remoteConfig);
            } else {
                throw new Error(`no configuration file found at: ${config}`);
            }
        }

        const { stdout, exitCode } = await this.ops.runRemote(
            `${remoteEngine}/ivllm-serve.sh -j "${job}" -t "${maxTime}"${batch ? ' -b' : ''}`,
            { env: this.envs, silent: false },
        );

        if (exitCode !== 0)
            throw new Error(
                `startup request failed for job ${job} (exit ${exitCode}): ${stdout}`,
            );

        // N.b. monitoring of job output is a secondary task and requires
        // coordinating process to start up a watchLog and kill it off when
        // appropriate.
    }

    async getAllJobStatus(): Promise<LockfileV3[]> {
        await this.bootstrap();
        const remoteEngine = await this.getRemoteEngine();

        const { stdout, exitCode } = await this.ops.runRemote(
            `${remoteEngine}/ivllm-status.sh -p`,
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
        start?: boolean,
    ): Promise<CloseableEventEmitter> {
        await this.bootstrap();
        const remoteEngine = await this.getRemoteEngine();

        return this.ops.runRemoteSync(
            `${remoteEngine}/ivllm-show-log.sh -j "${job}" -n "${node ?? '0'}"${start ? ` -a` : ''}`,
            this.envs,
        );
    }

    async fetchDiagnostics(job: string, localDest?: string): Promise<string> {
        await this.bootstrap();

        const remoteDiagDir = `${this.creds.projectDir}/engine/diagnostics/${job}`;
        const targetDir =
            localDest ||
            path.join(homedir(), '.config', 'ivllm', 'diagnostics', job);

        console.log(`Downloading diagnostics for job '${job}'...`);
        await this.ops.copyDirectory(targetDir, remoteDiagDir, 'down');

        return targetDir;
    }

    private async getRemoteEngine(): Promise<string> {
        return `${await this.getRemoteHome()}/.local/bin`;
    }

    private async getRemoteHome(): Promise<string> {
        if (!this.remoteHome) {
            const { stdout } = await this.ops.runRemote('echo $HOME', {
                env: this.envs,
                silent: true,
            });
            this.remoteHome = stdout.trim();
        }
        return this.remoteHome;
    }

    // private async getRemoteEngineVersion(): Promise<string> {
    //     const { stdout } = await this.ops.runRemote(
    //         `cat "${await this.getRemoteHome()}/.config/ivllm/version" 2>/dev/null || true`,
    //         { env: this.envs, silent: true },
    //     );
    //     return stdout.trim();
    // }
    //
    // private async setRemoteEngineVersion(version: string): Promise<void> {
    //     await this.ops.runRemote(
    //         `umask 002 && mkdir -p "${await this.getRemoteHome()}/.config/ivllm" && echo "${version}" > "${await this.getRemoteHome()}/.config/ivllm/version"`,
    //         { env: this.envs, silent: true },
    //     );
    // }

    private remoteComparisonDir(comparison: string): string {
        return `${this.getRemoteHome()}/ivllm/benchmark/${comparison}`;
    }

    override async requestBenchmark(
        comparison: string,
        configs: string[],
        time?: string,
    ): Promise<void> {
        await this.bootstrap();
        const remoteEngine = await this.getRemoteEngine();
        const remoteDir = this.remoteComparisonDir(comparison);

        // Upload every config — same copyFile call requestStart() already uses
        // for a single config, just once per file here.
        for (const config of configs) {
            if (!fs.existsSync(config)) {
                throw new Error(`no configuration file found at: ${config}`);
            }
            const remoteConfig = `${remoteDir}/${path.basename(config)}`;
            await this.ops.copyFile(config, remoteConfig);
        }

        // Detached launch — see ivllm-bench.sh's own "Fire-and-forget client
        // contract" comment block for why (no GPU work of its own, can run for
        // hours, must survive this SSH command returning).
        const timeEnv = time ? ` IVLLM_BENCH_TIME="${time}"` : '';
        const launch =
            `nohup${timeEnv} ${remoteEngine}/ivllm-bench.sh -c "${remoteDir}" ` +
            `> "${remoteDir}/orchestrator.log" 2>&1 < /dev/null & disown; echo started`;

        const { stdout, exitCode } = await this.ops.runRemote(launch, {
            env: this.envs,
            silent: false,
        });

        if (exitCode !== 0 || !stdout.includes('started')) {
            throw new Error(
                `benchmark submit failed for '${comparison}' (exit ${exitCode}): ${stdout}`,
            );
        }
    }

    override async getBenchmarkStatus(
        comparison: string,
    ): Promise<BenchmarkStatus> {
        await this.bootstrap();
        const remoteDir = this.remoteComparisonDir(comparison);
        const { stdout, exitCode } = await this.ops.runRemote(
            `cat "${remoteDir}/benchmarking_status.json"`,
            { env: this.envs, silent: true },
        );
        if (exitCode !== 0) {
            throw new Error(
                `no status found for comparison '${comparison}' — check the name, or that submit succeeded`,
            );
        }
        return JSON.parse(stdout) as BenchmarkStatus;
    }

    override async fetchBenchmarkResults(
        comparison: string,
        localDest: string,
    ): Promise<
        | { ready: true; path: string }
        | { ready: false; status: BenchmarkStatus }
    > {
        const status = await this.getBenchmarkStatus(comparison);
        if (!status.complete) {
            return { ready: false, status };
        }
        const remoteDir = `${this.remoteComparisonDir(comparison)}/results`;
        await this.ops.copyDirectory(localDest, remoteDir, 'down');
        return { ready: true, path: localDest };
    }
}
