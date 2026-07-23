# isambard-vllm (`ivllm`)

A CLI tool for managing vLLM inference jobs on [Isambard AI](https://www.isambard.ac.uk/) HPC from your local machine. It submits SLURM jobs, downloads models on the login node, establishes a forward SSH tunnel, and exposes an OpenAI-compatible API on `localhost` — so you can point any agent harness (e.g. OpenCode) straight at your HPC GPU allocation.

```
http://localhost:11434/v1   ←→   ssh tunnel   ←→   vLLM on COMPUTE node
```

---

## Prerequisites

- **Bun** ≥ 1.3 installed locally (the native TypeScript runtime). If you already have **Node.js / npm** installed, you can install Bun globally in seconds by running:
  ```bash
  npm install -g bun
  ```
  Otherwise, install via shell script (`curl -fsSL https://bun.sh/install | bash`) or Homebrew (`brew install oven-sh/bun/bun`).
- A working SSH connection to the Isambard AI login node, with credentials cached in an SSH agent (key-based auth, no interactive password prompts)
- A HuggingFace account and access token for gated models (stored via `ivllm config --hf-token`). Hugging Face access token can be created from the [Access Token](https://huggingface.co/settings/tokens) page

---

## Installation

```bash
# Clone the repository
git clone https://github.com/ai4ci/isambard-vllm.git
cd isambard-vllm

# Option A: Using Bun (Recommended)
bun install
bun link

# Option B: Using npm
npm install
npm link
```

> If `bun link` doesn't put the binary on your PATH, add `~/.bun/bin` to your shell's `PATH`.

<details>
<summary>
Click to see how to do so
</summary>
   
> **zsh**
> ```zsh
> echo 'export PATH="$HOME/.bun/bin:$PATH"' >> ~/.zshrc
> source ~/.zshrc
> ```
>
> **bash**
> ```bash
> echo 'export PATH="$HOME/.bun/bin:$PATH"' >> ~/.bashrc
> source ~/.bashrc
> ```
> 
</details>

---

## Configuration

Run once to configure your connection details where XXXX is your project ID, and YYYY is your user id:

```bash
ivllm config --login-host <login-node>   # e.g. XXXX.aip2.isambard
ivllm config --username <hpc-username>   # e.g. YYYY.XXXX
ivllm config --project-dir <path>        # HPC project dir, e.g. /projects/XXXX
ivllm config --local-port <port>         # default: 11434
ivllm config --hf-token <token>          # HuggingFace token for gated models
```

Settings are saved to `~/.config/ivllm/config.json`. Run `ivllm config` with no arguments to view current settings.

---

## Shared Project Architecture (Important for Teams)

`isambard-vllm` uses a **shared multi-user architecture**. All artifacts live on the shared parallel filesystem:

### 1. Done ONCE per Team (Shared across all project members)
* **vLLM Setup (`ivllm setup`):** Running `ivllm setup <version>` installs the virtual environment and GPU compilation toolchains into your shared allocation (`$PROJECTDIR/engine/`). 
  * Once *any* teammate runs `ivllm setup`, **no other team members need to run it** for that version. Everyone instantly shares the same optimized installation!
* **Model Downloads:** Model weights are stored in the shared Hugging Face cache at `$PROJECTDIR/engine/hf/`.
  * When any member runs `ivllm connect`, the tool checks this shared directory. If *any* teammate has already downloaded the model, it is **reused instantly by everyone**, avoiding duplicated disk space and preventing Hugging Face API rate-limiting (429) blocks.

### 2. Done ONCE per Individual User (on your local machine)
* **Tool Installation:** Each teammate runs the local installation (`bun install && bun link`) once to install the CLI tool locally.
* **Local Configuration (`ivllm config`):** Each teammate runs `ivllm config` once on their own local machine to save their personal HPC username (e.g. `YYYY.XXXX`), host, and project directory.

### 3. Shared Job Lifecycle
* **Any project member can `ivllm connect` to any running job** — no session ownership.
* **Any project member can `ivllm cancel` any job** — graceful shutdown via lockfile.
* Lockfiles use `umask 0002` and `chmod g+w` for group-writable permissions.

---

## Commands

| Command | Description |
|---------|-------------|
| `connect <job>` | Start or connect to a vLLM inference session. If running, establishes SSH tunnel. If stopped/failed, restarts. |
| `cancel <job>` | Request graceful shutdown (writes 'cancel' to lockfile). Use `--force` for hard kill via scancel. |
| `status [job]` | Show status of a job (or all jobs) |
| `config` | Show or set connection details (host, username, project dir, HF token) |
| `setup <version>` | Install vLLM `<version>` on the HPC (one-off, e.g. `ivllm setup 0.19.1`) |

Run `ivllm <command> --help` for command-specific options.

---

## Quickstart

### 1. Install vLLM on the HPC (one-off per project)

```bash
ivllm setup 0.19.1
```

This submits a SLURM job on a compute node to install the NVIDIA HPC SDK 26.3 (providing CUDA 12.9 forward compatibility) and the specified vLLM version into a shared versioned directory at `$PROJECT_DIR/ivllm/0.19.1/`. Progress is streamed to your terminal. Takes ~10–20 minutes on first run; skipped automatically if that version is already installed. To install a different version run `ivllm setup <version>` again.

### 2. vLLM config file (examples provided)

The LLM server is configured via a YAML config file (see the [vLLM docs](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html)). The config file specifies the model and all serving parameters. Example `vllm.yaml`:

```yaml
model: Qwen/Qwen2.5-7B-Instruct
tensor-parallel-size: 1
max-model-len: 32768
gpu-memory-utilization: 0.90
dtype: bfloat16
enable-auto-tool-choice: true
tool-call-parser: hermes
enable-prefix-caching: true
```

This file needs to be saved locally, and passed in the `--config` parameter. See later
for details on how to create this file.

Ready-to-use example configs for popular models are in the [`examples/`](examples/) directory:

| File | Model | Notes |
|------|-------|-------|
| [`qwen2.5-instruct.yaml`](examples/qwen2.5-instruct.yaml) | Qwen/Qwen2.5-0.5B-Instruct | Dense 0.5B, single node (minimal example) |
| [`qwen3.6-35b-a3b.yaml`](examples/qwen3.6-35b-a3b.yaml) | Qwen/Qwen3.6-35B-A3B | Hybrid MoE 35B, reasoning, single node |
| [`qwen3.5-long-context.yaml`](examples/qwen3.5-long-context.yaml) | Qwen/Qwen3.5-35B-A3B | Hybrid MoE 35B, long context, single node |
| [`gemma-4-31B-it.yaml`](examples/gemma-4-31B-it.yaml) | google/gemma-4-31B-it | Dense 31B multimodal, single node |
| [`gpt-oss-120b.yaml`](examples/gpt-oss-120b.yaml) | openai/gpt-oss-120b | MoE 117B MXFP4, single node |
| [`nemotron-3-super-120B-A12B-BF16.yaml`](examples/nemotron-3-super-120B-A12B-BF16.yaml) | nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16 | Dense 120B reasoning/tool model, single node, requires shared parser plugin |
| [`minimax-m2.5.yaml`](examples/minimax-m2.5.yaml) | MiniMaxAI/MiniMax-M2.5 | MoE 230B, multi-node |

### 3. Connect to a vLLM session

The `connect` command is the primary way to interact with a vLLM session. It handles all states:

- **New job**: Creates lockfile, submits via `sbatch`, tails logs, establishes tunnel when healthy
- **Running job**: Establishes SSH tunnel immediately
- **Stopped/failed job**: Restarts from the stopped state

```bash
ivllm connect qwen2 --config examples/qwen2.5-instruct.yaml
```

This will:
1. Check SSH connectivity and that the venv exists
2. Read the model name from `vllm.yaml` and download it to the shared HF cache on the login node
3. Submit a SLURM job via `sbatch`
4. Tail remote log files to monitor startup progress
5. Establish a forward SSH tunnel once vLLM is healthy
6. Print the local endpoint and exit

**N.B. Starting up even a simple model can take a few minutes.**

```
🚀 OpenAI API endpoint: http://localhost:11434/v1
   Model: Qwen/Qwen2.5-0.5B-Instruct
```

The config file is cached so you can restart with just the job name:

```bash
ivllm connect qwen2
```

#### `ivllm connect` options

| Flag | Description | Default |
|------|-------------|---------|
| `--config <file>` | vLLM config YAML (contains model, parallelism and all serving options) | required for first use |
| `--local-port [port]` | Local port to expose the API on | `11434` |
| `--batch` | Submit to standard non-interactive queue | interactive partition |
| `--dry-run` | Preview without connecting to HPC | off |

### 4. Lifecycle management

The system uses a detach/reattach model — the job runs on the compute node independently of your local client:

- **Disconnect**: Close terminal, `connect` exits, job keeps running on compute
- **Reconnect**: Run `ivllm connect <job>` again to re-establish the tunnel
- **Cancel**: Run `ivllm cancel <job>` for graceful shutdown, or `ivllm cancel --force <job>` for hard kill via scancel
- **Idle timeout**: Jobs shut down automatically after the configured idle timeout if no API activity

```bash
# Check status
ivllm status
ivllm status qwen2          # specific job

# Cancel gracefully
ivllm cancel qwen2

# Cancel forcefully (kills SLURM job directly)
ivllm cancel qwen2 --force
```

### 5. Launch an AI coding assistant

After starting vLLM, `ivllm start` and `ivllm interactive` offer to launch your AI coding assistant with the endpoint pre-configured. When the menu appears:

- **Layer 1 — target**: choose **OpenCode**, **GitHub Copilot**, **Claude Code**, **Pi**, change directory, or shut down `ivllm`
- **Layer 2 — wrapper**: choose **direct launch**, **scoder**, or **sbx** (only wrappers available on your machine are shown)
- **Layer 3 — action**: choose **launch now** or **show copy-paste command**

For every wrapper, `ivllm` prints the full shell-ready command before launching so you can copy, paste, and tweak it manually if needed.

You can also launch the assistant separately from a running session:

```bash
ivllm agent --port 11434
```

#### sbx prerequisite

If you launch through **sbx**, Docker Sandboxes must be allowed to reach the host-side `ivllm` endpoint first:

```bash
sbx policy allow network localhost:11434
```

Replace `11434` with your configured local port if different. `ivllm` does **not** edit global `sbx policy` rules automatically.

> **Manual configuration (legacy):** If you prefer to configure your assistant manually, add `opencode.json` to your project directory:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "isambard-vllm": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Isambard vLLM Server",
      "options": {
        "baseURL": "http://localhost:11434/v1",
        "apiKey": "EMPTY"
      },
      "models": {
        "Qwen/Qwen2.5-0.5B-Instruct": {
          "name": "Qwen2.5-0.5B-Instruct (Isambard)",
          "limit": {
            "context": 32768,
            "output": 8192
          }
        }
      }
    }
  }
}
```

Start up opencode and select the Qwen model from the `Isambard vLLM Server provider`.

When you have finished type "exit" at the terminal you started `ivllm` in and the isambard job will finish.

### 6. Check job status

```bash
ivllm status              # all known jobs
ivllm status qwen2        # specific job
```

Shows the lockfile status for each job (status, model, port, timestamps, reason).

### 7. Check job status

Not generally necessary as the local session from `ivllm start` or `ivllm interactive` will display the current status.

```bash
ivllm status           # all known jobs
ivllm status qwen2    # specific job
```

### 8. Cancel a job

If you need to stop a running job:

```bash
ivllm cancel qwen2        # graceful shutdown via lockfile
ivllm cancel qwen2 --force  # hard kill via scancel
```

Graceful cancel writes 'cancel' to the lockfile — the compute-side monitor detects it and shuts down cleanly. Force cancel runs `scancel` directly on the SLURM job.

---

## Generating a config with AI

`ivllm` ships an [Agent Skill](https://agentskills.io) for generating `vllm.yaml` files. If you are using an AI coding agent (Cursor, Claude, Windsurf, etc.), the skill will help the agent generate an optimised config for any HuggingFace model on Isambard AI hardware.

**Install the skill** run `bunx skills ai4ci/isambard-vllm`, or `bunx skills-npm` once ivllm is installed:

Once installed, ask your AI agent: *"Generate a vllm.yaml config for `Qwen/Qwen2.5-72B-Instruct`"*

---

## HuggingFace token

For gated models, store your token in the ivllm config:

```bash
ivllm config --hf-token hf_...
```

The token is saved to `~/.config/ivllm/config.json`. It is forwarded to the login node during the model download step and embedded in the setup SLURM script so the HPC can authenticate to HuggingFace. It is not stored in any shared or world-readable location. If not set, `ivllm` falls back to the `HF_TOKEN` environment variable.

---

## How it works

```
LOCAL                          LOGIN node                    COMPUTE node
------                         ----------                    ------------
ivllm connect <job>
  │─── ssh: check + bootstrap ──▶│
  │─── scp: engine/ ────────────▶│                             │
  │─── ssh: sbatch ─────────────▶│──── SLURM job ────────────▶│
  │                            │                             │ vLLM starts
  │                            │◀── writes status.json ──────│
  │◀── ssh: tail logs ──────────│──── tail remote log ──────▶│
  │                            │                             │
  │ (status: running)
  │─── ssh -L localPort:computeHost:serverPort ────────────▶│
  │
  └─── GET http://localhost:localPort/health (optional)
```

- The lockfile (`status.json`) lives on the shared parallel filesystem under `$PROJECTDIR/engine/jobs/<job>/`
- The CLI creates a `pending` lockfile, submits via `sbatch`, then tails logs
- The compute-side bash framework transitions: `pending → initialising → running` (or `failed`)
- Lifecycle ownership is on the **COMPUTE node** — the CLI can disconnect and reconnect freely
- The monitor triad (`monitor_startup`, `monitor_head`, `monitor_worker`) manages the job on compute
- All tunnelling is initiated by LOCAL; compute nodes cannot initiate outbound SSH connections on Isambard AI

---

## Dry run

Preview what `ivllm connect` would do without connecting to the HPC:

```bash
ivllm connect qwen2 --config vllm.yaml --dry-run
```

This prints the commands that would be sent without executing them.

---

## Development

```bash
bun test                     # all TypeScript + bash integration tests
bash tests/bash/run.sh       # bash tests only (unit + sandboxed)
bun run start                # run CLI directly
```

All tests use TDD. Test layout:
- `tests/unit/*.test.ts` — TypeScript unit tests (Backend, RemoteOps mock, local-ops, semver)
- `tests/integration/*.test.ts` — Full lifecycle via mock backend
- `tests/bash/unit/` — Pure bash logic tests
- `tests/bash/sandboxed/` — Bubblewrap-sandboxed bash tests with real jq/yq, mocked SLURM/vLLM
