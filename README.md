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
 
</details>

---

## Configuration

Run once to configure your connection details where XXXX is your project ID, and YYYY is your user id:

```bash
ivllm config --login-host <login-node>   # e.g. XXXX.aip2.isambard
ivllm config --username <hpc-username>   # e.g. YYYY.XXXX
ivllm config --project-dir <path>        # HPC project dir, e.g. /projects/XXXX
ivllm config --hf-token <token>          # HuggingFace token for gated models
```

Settings are saved to `~/.config/ivllm/config.json`. Run `ivllm config` with no arguments to view current settings.

---

## Quickstart

The typical workflow is:

1. **Configure once** (see [Configuration](#configuration) above)
2. **Discover** what's running: `ivllm status`
3. **Connect** to a job: `ivllm connect <job>`

```bash
# Step 1 — configure (one-off, per user)
ivllm config --login-host XXXX.aip2.isambard
ivllm config --username YYYY.XXXX
ivllm config --project-dir /projects/XXXX
ivllm config --hf-token hf_...

# Step 2 — see what's running, stopped, or failed
ivllm status
# ┌──────────┬──────────────────────────────────┬────────┬────────────┬─────────────┐
# │ Job      │ Model                            │ Status │ Port       │ Reason      │
# ├──────────┼──────────────────────────────────┼────────┼────────────┼─────────────┤
# │ qwen36   │ Qwen/Qwen3.6-35B-A3B-FP8         │ running│ 49153      │             │
# │ gemma4   │ google/gemma-4-31B-it            │ stopped│ 49154      │ idle timeout│
# └──────────┴──────────────────────────────────┴────────┴────────────┴─────────────┘

# Step 3a — attach to a running job (instant)
ivllm connect qwen36
# 🚀 OpenAI API endpoint: http://localhost:11434/v1
#    Model: Qwen/Qwen3.6-35B-A3B-FP8

# Step 3b — start a stopped/failed job (or a new one)
ivllm connect gemma4 --config examples/gemma-4-31B-it.yaml
# Waiting for job to start...
# vLLM is starting up...
# 🚀 OpenAI API endpoint: http://localhost:11434/v1
#    Model: google/gemma-4-31B-it
```

### How jobs shut down

Jobs shut down automatically — no manual intervention needed.

| Cause | Behaviour |
|-------|-----------|
| **Idle timeout** | After `idleTimeout` minutes (default: 30) with no API requests → `stopped` |
| **SLURM time limit** | Job reaches its `--time` limit → `stopped` |
| **vLLM crash** | Process dies unexpectedly → `failed` |
| **Manual cancel** | `ivllm cancel <job>` → graceful `stopped`; `ivllm cancel --force <job>` → hard kill |

### Reconnecting

Jobs run independently on the compute node — your local client can disconnect and reconnect freely.

```bash
# Disconnect: just close the terminal or Ctrl+C
# The job keeps running on compute

# Reconnect later:
ivllm connect qwen36   # attaches to the running job instantly
```

---

## Example Configs

LLM servers are configured via a YAML config file (see the [vLLM docs](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html)). The config specifies the model and all serving parameters:

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

16 ready-to-use configs are provided in [`examples/`](examples/):

| Model family | Examples |
|--------------|----------|
| Qwen | 2.5 (0.5B→3.6-35B), 3.5 (397B-A17B FP8, 3.5T Long Context), 3 Coder (Next Long Context) |
| DeepSeek | V4 (Flash, Pro) |
| Google | Gemma-4 (31B IT) |
| Other | GPT-OSS-120B, GLM-5.2-743B, MiniMax-M2.5, Nemotron-3-Super-120B |

To generate a config for an arbitrary model, see [Generating a config with AI](#generating-a-config-with-ai).

---

## Generating a config with AI

`ivllm` ships an [Agent Skill](https://agentskills.io) for generating `vllm.yaml` files. If you are using an AI coding agent (Cursor, Claude, Windsurf, etc.), the skill will help the agent generate an optimised config for any HuggingFace model on Isambard AI hardware.

**Install the skill** run `bunx skills ai4ci/isambard-vllm`, or `bunx skills-npm` once ivllm is installed:

Once installed, ask your AI agent: *"Generate a vllm.yaml config for `Qwen/Qwen2.5-72B-Instruct`"*

---

## Commands

| Command | Description |
|---------|-------------|
| `connect <job> [--config <file>]` | Connect to a running job, or start a new one. If running, establishes SSH tunnel immediately. If stopped/failed/never run, submits a SLURM job and waits. |
| `status [job]` | Show status of all jobs (or a specific one). Use to discover what's running, stopped, or failed. |
| `cancel <job> [--force]` | Cancel a running job. Graceful (default) writes `cancel` to the lockfile for clean shutdown. `--force` kills the SLURM job directly. |
| `config` | Show or set connection details (host, username, project dir, HF token). Run once per user. |
| `setup <version>` | **Admin:** Install vLLM `<version>` on the HPC. Creates a shared venv with CUDA toolchains. One-off per project version. |

Run `ivllm <command> --help` for command-specific options.

### `ivllm connect` options

| Flag | Description | Default |
|------|-------------|---------|
| `--config <file>` | vLLM config YAML (required for first use) | required for first use |
| `--local-port [port]` | Local port to expose the API on | `11434` |
| `--batch` | Submit to standard non-interactive queue | interactive partition |
| `--time <hh:mm:ss>` | SLURM time limit | `08:00:00` |
| `--dry-run` | Preview without connecting to HPC | off |

---

## Admin: `ivllm setup`

`ivllm setup` installs vLLM on the HPC. This is a **one-time per-project-admin** operation — once any team member runs it, everyone in the project shares the result.

```bash
ivllm setup 0.19.1
```

This submits a SLURM job on a compute node to install the NVIDIA HPC SDK 26.3 (providing CUDA 12.9 forward compatibility) and the specified vLLM version into a shared directory at `$PROJECTDIR/engine/<version>/`. Takes ~10–20 min on first run. Skipped automatically if that version is already installed.

---

## Shared Multi-User Architecture

`isambard-vllm` uses a **shared multi-user architecture**. All artifacts live on the shared parallel filesystem:

### Done ONCE per Team
* **vLLM Setup:** Once *any* teammate runs `ivllm setup`, **no other team members need to run it**. Everyone instantly shares the same optimized installation.
* **Model Downloads:** Model weights are stored in the shared Hugging Face cache at `$PROJECTDIR/model/hf/`. One download serves everyone — avoids duplicate disk space and HF API rate-limiting (429) blocks.

### Done ONCE per Individual User
* **Tool Installation:** Each teammate runs `bun install && bun link` once on their local machine.
* **Local Configuration:** Each teammate runs `ivllm config` once to save their HPC username, host, and project directory.

### Shared Job Lifecycle
* **Any project member can `connect` to any running job** — no session ownership.
* **Any project member can `cancel` any job** — graceful shutdown via lockfile.
* Lockfiles use `umask 0002` and `chmod g+w` for group-writable permissions.

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
