# MVP: isambard-vllm

Using HPC resources to run inferencing for public large language models. 

A command line tool to manage the lifecycle of llm inference jobs on a remote HPC (Isambard AI) from a local machine.
The goal is to have one or more ports on the local machine that are serving an OpenAI api endpoint from a compute node on an HPC, and which we can connect to through an agent harness e.g. OpenCode.
The process of establishing credentials for SSH is outside of the scope of thisd project and can be assumed.

* LOCAL: the local machine
* LOGIN: the HPC login node
* COMPUTE: the HPC compute node

## Prerequisites / Assumptions

* A working ssh connection to the HPC LOGIN, with any credentials cached in a ssh-agent.
* SLURM support on the HPC.
* The HPC has `jq`.
* huggingface token as $HF_TOKEN.

Each HPC inference job has a specified name, a model name, and a vllm configuration file.
MVP is single-node inference only.

## Workflow

1) vllm setup and installation script:
* This script is run on LOCAL, but remotely executes SLURM scripts copied onto the LOGIN node, using the COMPUTE nodes (so that the vllm installation is aware of the hardware configuration of the COMPUTE nodes).
* The SLURM script ensures that `vllm` is installed in the HPC using `uv` (see draft [design/old/setup-vllm.sh]).
* One installation of vLLM can be reused across HPC inference jobs.
* vllm symlinked in `.local/bin` on HPC? will need activation of venv...?
* failures reported to user on LOCAL machine (inc log files etc).

2) `ivllm start <job>` — session owner:
* `ivllm start` is a **long-running foreground process** that owns the full lifecycle of the inference session. It exits only when the session ends (user request, heartbeat failure, or SLURM timeout).
* Each inference job has a name. A single inference job is associated with a working directory on the HPC named after the job.
* Objective: start vllm with a specific model and vllm configuration options file (see [examples/gpt-oss-120b.yaml]).
* LOCAL connects to LOGIN and checks vllm is installed. Fails early if not.
* **Model download** (on LOGIN, before submitting the SLURM job):
  * Checks whether the model is already present in the shared HuggingFace cache (`$PROJECTDIR/hf`).
  * If not cached: runs `huggingface-cli download <model>` on LOGIN via SSH, with `HF_HOME=$PROJECTDIR/hf` and `HF_TOKEN` from LOCAL environment. Streams download progress to the user.
  * This avoids tying up a GPU COMPUTE allocation during what can be a lengthy download, and ensures the model is available before the SLURM job starts.
  * COMPUTE nodes use the same cache via `HF_HOME=$PROJECTDIR/hf` — no download occurs at job start.
* Creates `job_details.json` in the HPC job working directory with `status: "pending"`. If the file already exists, exits with error (lockfile — prevents duplicate jobs with the same name).
* LOCAL copies the vllm config file and a generated SLURM script to LOGIN via ssh, then submits the SLURM job and enters monitoring mode.
* **SLURM script responsibilities** (see draft [design/old/vllm-slurm.sh]):
  * On start: writes node hostname, SLURM job ID to `job_details.json`; sets `status: "initialising"`.
  * Starts vLLM with the config options and waits for it to become healthy.
  * Once healthy: writes the server port to `job_details.json`; sets `status: "running"`.
  * On failure or SLURM timeout: sets `status: "failed"` (or `"timeout"`) in `job_details.json`.
  * Logs to file in the HPC job working directory.
  * **No SSH tunnel logic** — tunnelling is entirely LOCAL's responsibility.
* **LOCAL monitoring mode**:
  * Polls `job_details.json` on LOGIN via SSH, reporting status changes to the user.
  * On `status: "running"`: spawns a child process for the forward SSH tunnel (`ssh -L <local_port>:<compute_hostname>:<server_port> <user>@<login_node>`).
  * Runs a periodic heartbeat: polls the vLLM `/health` endpoint through the tunnel.
  * If heartbeat fails or `status` becomes `"failed"`/`"timeout"`: reports the error, displays SLURM logs, then initiates shutdown sequence.
  * Accepts user input ("exit" or Ctrl+C) to initiate a clean shutdown.
* **Shutdown sequence** (triggered by any of the above):
  1. `scancel` the SLURM job via SSH to LOGIN.
  2. Kill the SSH tunnel child process.
  3. Remove `job_details.json` from the HPC (unlock).
  4. Exit.
* If LOCAL crashes without a clean shutdown, `ivllm stop <job>` performs the same shutdown sequence as a recovery tool.

## CLI Commands (MVP)

* `ivllm setup` — install vLLM on the HPC (one-off, shared across jobs). Runs a SLURM job on a COMPUTE node so vLLM is built against the correct hardware.
* `ivllm start <job> --model <model> --config <file> [--local-port <port>]` — long-running session owner: pre-downloads model on LOGIN if not cached, submits the SLURM inference job, monitors startup, establishes the SSH tunnel on a fixed local port (default: 11434), runs heartbeat, and tears everything down on exit. Prints the local connection URL (`http://localhost:<port>/v1`) once the tunnel is up.
* `ivllm status [job]` — display the current status of a job (or all known jobs) from `job_details.json` on LOGIN.
* `ivllm stop <job>` — recovery tool: cancels the SLURM job, kills any lingering tunnel, and removes the lockfile when `ivllm start` did not exit cleanly.

MVP assumes a single `ivllm start` process controlling a single vLLM server on a fixed local port. The implementation should be structured to support multiple concurrent jobs (each with its own port) as a future phase — see Future Considerations.

MVP is single-node inference only. Multi-node is a future phase.

## SSH Tunnelling

COMPUTE nodes on Isambard AI cannot initiate outbound SSH connections, so reverse tunnels (COMPUTE → LOGIN) are not possible. The correct approach is a **forward tunnel** initiated by LOCAL once the job is running:

```
ssh -L <local_port>:<compute_hostname>:<server_port> <user>@<login_node>
```

The SLURM script writes the compute node hostname and server port to `job_details.json`. LOCAL reads these values and establishes the forward tunnel.

The scripts in `design/old/` that use reverse tunnels (`vllm-slurm.sh`, `tunnel-test.sh`) are deprecated for this reason.

## Implementation

* The tutorial in [design/references/vllm-distributed.md] uses a pre-downloaded set of model weights. This is only available for GPT-OSS. We will not be able to use this.
* LOCAL CLI is implemented in Node.js + bun. LOGIN and COMPUTE scripts are bash.
* vLLM startup is slow. We need a connection test that mocks the vLLM server, so we can end-to-end test the monitoring workflow.
* Testing end to end with vLLM will need to be done with a lighter weight LLM like Qwen/Qwen2.5-0.5B-Instruct

## References

* [design/references/cuda.md] - using newer CUDA versions on isambard
* [design/references/vllm-serve-0.19.1] - vllm serve commarnd reference for v0.19.1
* [design/references/vllm-distributed.md] - describes the process of setting up inferencing on COMPUTE node and testing via a second COMPUTE node job on Isambard. This is useful for the details of how to interactively set up and start vllm. This is what we are trying to automate.
* [design/references/vllm-parallel.md] - describes the process of setting up vllm in parallel on general hardware.
* [design/references/storage.md] - info about where Isambard storage is.
