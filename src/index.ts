#!/usr/bin/env bun
import { cmdSetup } from './commands/setup.ts';
import { cmdConnect } from './commands/connect.ts';
import { cmdCancel } from './commands/cancel.ts';
import { cmdStatus } from './commands/status.ts';
import { cmdList } from './commands/list.ts';
import { cmdAgent } from './commands/agent.ts';
import { cmdConfig } from './commands/config.ts';

// Assign globally across Node.js/Browser using the universal globalThis object
const { version } = await import('../package.json');
(globalThis as any).__VERSION__ = version;

const [, , command, ...args] = process.argv;

const USAGE = `
Usage: ivllm <command> [options]

Commands:
  setup <version>         Install vLLM <version> on the HPC (one-off)
  connect <job>           Start or connect to an inference session
  cancel <job>            Cancel a running job
  list                    List stored vLLM job configs
  status [job]            Show status of a job (or all jobs)
  config                  Show or set configuration
  agent                   Launch AI assistant connected to local vLLM server

Options:
  --version, -v           Show version

Run 'ivllm <command> --help' for command-specific options.

For command-specific help, run:
  ivllm connect --help    Connect options
  ivllm cancel --help     Cancel options
  ivllm setup --help      Setup options
  ivllm agent --help      Agent options (including --port)
  ivllm config --help     Config options
`.trim();

switch (command) {
  case '--version':
  case '-v':
    console.log(`ivllm ${__VERSION__}`);
    process.exit(0);
  case 'setup':
    await cmdSetup(args);
    break;
  case 'connect':
    await cmdConnect(args);
    break;
  case 'cancel':
    await cmdCancel(args);
    break;
  case 'status':
    await cmdStatus(args);
    break;
  case 'list':
    await cmdList(args);
    break;
  case 'config':
    await cmdConfig(args);
    break;
  case 'agent':
    await cmdAgent(args);
    break;
  default:
    console.log(USAGE);
    process.exit(command ? 1 : 0);
}
