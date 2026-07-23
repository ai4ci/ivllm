import { join, isAbsolute, dirname } from 'node:path';
import fs from 'node:fs';
import url from 'node:url';
import { EventEmitter } from 'node:events';
import { spawn } from 'child_process';
import { SandboxManager } from 'vsbx';
import type {
    RunRemoteOptions,
    RunRemoteResult,
    CloseableEventEmitter
} from '../types';
import {
    RemoteOps
} from './RemoteOps'

export class SandboxRemoteOps extends RemoteOps {
    private sandboxDir: string;
    private workspaceDir: string;
    private binDir: string;

    constructor(sandboxDir: string) {
        super();
        this.sandboxDir = sandboxDir;
        this.workspaceDir = join(sandboxDir, "workspace");
        this.binDir = join(sandboxDir, "bin");

        // 1. Physically scaffold directories on the host
        fs.mkdirSync(this.workspaceDir, { recursive: true });
        fs.mkdirSync(this.binDir, { recursive: true });
        // 2. Provision local cluster utilities (sbatch, vllm, etc.)
        const mockDir = join(dirname(url.fileURLToPath(import.meta.url)),"mocks");
        fs.cpSync(mockDir, this.binDir, { recursive: true })

        // 3. Initialize the stateful sandbox wrapper instance
        // All properties configured here persist for the lifetime of this object instance.
        SandboxManager.initalize(
            {
                // Read-Write bindings preserve state modification between runRemote executions
                rwBindings: [
                    { hostPath: this.workspaceDir, sandboxPath: "/workspace" }
                ],
                // Read-Only hooks inject our mock orchestrations seamlessly
                roBindings: [
                    { hostPath: this.binDir, sandboxPath: "/hpc-bin" }
                ],
                // Isolate environmental manipulation
                env: {
                    PATH: `/hpc-bin:${process.env.PATH}`,
                    PROJECTDIR: '/workspace',

                },
                allowNetwork: true // Set to false if you want absolute local isolation
            }
        );
    }

    /**
     * Translates a relative/absolute path passed by your app logic
     * to the physical isolated location inside the host's testing workspace.
     */
    private resolvePath(remotePath: string): string {
        if (isAbsolute(remotePath)) {
            // Strip leading '/' to sand-box it into our folder safely
            return join(this.workspaceDir, remotePath.substring(1));
        }
        return join(this.workspaceDir, remotePath);
    }

    async runRemote(
        command: string,
        options: RunRemoteOptions = { env: [], silent: true },
    ): Promise<RunRemoteResult> {
        return new Promise((resolve, reject) => {
            const fullCommand = this.makeFullCommand(command, options.env);
            const sandboxedCommand = await SandboxManager.wrapWithSandbox(fullCommand);
            const proc = spawn(
                sandboxedCommand,
                {
                    shell: true,
                    stdio: ['ignore', 'pipe', 'inherit']
                },
            );

            let stdout = '';

        proc.stdout?.on('data', (chunk: Buffer) => {
            stdout += chunk.toString();
            if (!options.silent) console.log(chunk.toString());
        });

            proc.on('error', reject);
            proc.on('close', (code) =>
            resolve({ exitCode: code ?? 1, stdout: stdout.trim() }),
            );
        }
    }

    /**
     * Cleans up the host system resources and resets the state
     * after the testing loop finishes.
     */
    async destroy(): Promise<void> {
        console.log("🧹 Tearing down Sandbox Remote Ops lifecycle context...");
        // 1. Force-remove the sandboxed workspace files and generated mock binaries
        fs.rmSync(this.sandboxDir, { recursive: true, force: true });
        SandboxManager.reset();
    }

    async copyFile(localPath: string, remotePath: string): Promise<void> {
        const targetHostPath = this.resolvePath(remotePath);
        fs.mkdirSync(dirname(targetHostPath), { recursive: true });
        fs.copyFileSync(localPath, targetHostPath);
    }

    tailRemoteLog(remotePath: string, prefix = ''): { stop: () => void } {
        const targetHostPath = this.resolvePath(remotePath);

        // Ensure file exists so tail doesn't immediately exit with an error
        if (!fs.existsSync(targetHostPath)) {
            fs.mkdirSync(dirname(targetHostPath), { recursive: true });
            fs.writeFileSync(targetHostPath, '');
        }

        // Spawn a local native tail process pointing at our sandbox file
        const proc = spawn(
            'tail',
            ['-f', '-n', '+1', targetHostPath],
            { stdio: 'inherit' }
        );

        return {
            stop: () => {
                proc.kill();
            },
        };
    }

    spawnTunnel(
        localPort: number,
        remoteHost: string,
        remotePort: number,
    ): CloseableEventEmitter {
        const pid=998877
        const mockSshTunnel = Object.assign(
            new EventEmitter(), {
                stdin: Object.assign(new EventEmitter(), { write: () => true }),
                stdout: Object.assign(new EventEmitter(), {
                    pipe: (destination: EventEmitter) => destination,
                }),
                stderr: Object.assign(new EventEmitter(), {
                    pipe: (destination: EventEmitter) => destination,
                }),
                // 2. Add required identity properties so it looks like a running process
                pid: pid,
                exitCode: null, // Stays null so it looks permanently alive
                kill: () => {
                    console.log(`  [dry-run] Shutting down tunnel ${localPort}:${remoteHost}:${remotePort}`);
                    mockSshTunnel.emit('close', 0);
                    return true;
                },
            });
        return mockSshTunnel;
    }

    async matchVllmVersion(minVllmVersion: string): Promise<string> {
        // Simulate standard cluster modules queries or manual package listings
        // Returns a valid mock match satisfying >= minVllmVersion
        return `0.4.2-mock-isambard`;
    }

    async checkSSH(): Promise<boolean> {
        return true;
    }

}

