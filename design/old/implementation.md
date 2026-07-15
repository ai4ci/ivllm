# Implementation roadmap — isambard-vllm

## Phase 1 — Project scaffold and tooling

- [x] Initialise bun/Node.js project (`bun init`, `package.json`, TypeScript config)
- [x] CLI entry point with sub-command routing (`ivllm setup | start | status | stop`)
- [x] Configuration file (`~/.config/ivllm/config.json`): HPC login host, HPC username, venv path, default local port
- [x] SSH helper module: run a remote command on LOGIN, copy a file to LOGIN (wraps `ssh`/`scp` child processes)
- [x] Commit

## Phase 2 — `ivllm setup`

- [x] Generate setup SLURM script from template (based on `design/old/setup-vllm.sh`)
- [x] Copy script to LOGIN and submit via `sbatch`
- [x] Poll SLURM job status until complete or failed
- [x] Stream SLURM output log back to LOCAL in real time
- [x] Report success (vLLM version) or failure with log excerpt
- [x] Validate venv exists on HPC after setup
- [x] Commit

## Phase 3 — SLURM inference script

- [x] Write SLURM bash template for single-node vLLM inference
  - Activate venv
  - Write initial `job_details.json` (`status: "initialising"`, node hostname, SLURM job ID)
  - Start vLLM with config file, model, port
  - Poll `/health` until ready; update `job_details.json` (`status: "running"`, server port, model name)
  - On failure/timeout: update `job_details.json` (`status: "failed"` / `"timeout"`)
  - Log to file in job working directory
  - No SSH tunnel logic in SLURM script
- [x] Unit-testable template rendering (job name, config path, model, port substitution)
- [x] Commit

## Phase 4 — `ivllm start` — core session owner

- [x] Pre-flight: check venv exists on HPC; fail early if not
- [x] Model pre-download on LOGIN:
  - [x] Check HuggingFace cache at `$PROJECTDIR/hf` for the requested model
  - [x] If not cached: run `huggingface-cli download <model>` on LOGIN via SSH with `HF_HOME=$PROJECTDIR/hf` and `HF_TOKEN` forwarded; stream progress to user
- [x] Create `job_details.json` on HPC (`status: "pending"`); fail if already exists (lockfile)
- [x] Copy vllm config file and generated SLURM script to LOGIN
- [x] Submit SLURM job via `sbatch`; record SLURM job ID
- [x] Poll `job_details.json` on LOGIN; display status transitions to user
- [x] On `status: "running"`: spawn forward SSH tunnel child process
- [x] Print connection URL to user: `http://localhost:<port>/v1`
- [x] Heartbeat loop: poll `/health` through tunnel on configurable interval
- [x] Shutdown sequence (Ctrl+C, "exit" input, heartbeat failure, or SLURM failure):
  1. `scancel` SLURM job via SSH
  2. Terminate tunnel child process
  3. Remove `job_details.json` from HPC
- [x] Handle SIGINT/SIGTERM to trigger shutdown sequence
- [x] Commit

## Phase 5 — `ivllm status` and `ivllm stop`

- [x] `ivllm status [job]`: SSH to LOGIN, read `job_details.json` for named job (or all job working directories); display status table
- [x] `ivllm stop <job>`: recovery path — `scancel` by SLURM job ID from `job_details.json`, kill any lingering tunnel processes on LOCAL, remove `job_details.json`
- [x] Commit

## Phase 6 — Mock vLLM script and `ivllm start --dry-run`

- [x] Write a mock vLLM SLURM script template (`src/templates/mock-inference.ts`) that:
  - Uses the same template approach as `renderInferenceScript` — submitted as a SLURM batch job
  - Writes `job_details.json` correctly (same schema as real inference script)
  - Serves `/health` and `/v1/models` on the COMPUTE node via a lightweight bash HTTP server
  - Simulates a configurable startup delay before marking status `"running"`
- [x] Add `--mock` flag to `ivllm start`: substitutes `renderMockInferenceScript()` for `renderInferenceScript()`, otherwise identical flow
- [x] Implement `--dry-run` flag on `ivllm start` only:
  - SSH primitives are swapped for dry-run equivalents via a thin wrapper in `start.ts`
  - File copies (`scp`) write to a local temp directory instead of the remote LOGIN node
  - SSH remote commands are printed (with full command text) but not executed
  - SLURM job is not submitted
  - Key output for review: generated SLURM script and vLLM config file in local temp dir
  - Works with both real and mock modes (i.e. `--mock --dry-run` reviews the mock script)
- [x] Run `ivllm start <job> ... --dry-run` and review real inference script + config
- [x] Run `ivllm start <job> ... --mock --dry-run` and review mock SLURM script

## Phase 7 - Mock vLLM remote testing
- [x] End-to-end test: `ivllm start` against mock script, verify tunnel, heartbeat, clean shutdown
- [x] Test heartbeat failure path (mock server exits; verify LOCAL detects and shuts down cleanly)
- [x] Test lockfile behaviour (start same job twice)
- [x] Test `ivllm stop` recovery (simulate unclean exit, verify stop cleans up)
- [x] Commit

## Phase 8 — End-to-end test with real vLLM

- [x] Test `ivllm setup` on Isambard AI (resolve ADR-005 venv path question)
- [x] Confirm COMPUTE node has internet access
- [x] Test `ivllm start` with `Qwen/Qwen2.5-0.5B-Instruct` (lightweight model)
- [x] Verify OpenAI API endpoint accessible on LOCAL via tunnel
- [x] Resolve Unknowns: HuggingFace token, model cache location (`$HF_HOME` in `$PROJECTDIR`?)
- [x] Commit

---

## Post-MVP fixes and features

### Issue #1 — Exit before SLURM schedules

- [x] Moved readline setup to immediately after `sbatch` submission, so typing `exit` works during PENDING and initialising phases (not only after vLLM is running)

### Issue #2 — Launch feedback

- [x] While `job_details.json` shows `"pending"`, poll `squeue -j <id>` and print SLURM queue state (e.g. `PENDING Priority`, `RUNNING None`) so the user knows where they are in the queue
- [x] During `"initialising"`, incrementally tail the SLURM log and prefix each new line with `|` to show vLLM startup progress in real time
- [x] Added `parseSlurmQueueState` / `getSlurmQueueState` to `src/slurm.ts`

### Issue #3 — Multi-node inference via Ray

- [x] `resolveGpuCount` now returns `{ gpuCount, nodeCount }` where `nodeCount = ceil(gpuCount / gpusPerNode=4)`; no longer errors on configs with `pipeline-parallel-size > 1`
- [x] `renderInferenceScript` accepts `nodeCount`; when `nodeCount > 1` generates a Ray cluster bootstrap SLURM script:
  - Sets `#SBATCH --nodes=N`
  - Starts Ray head node and worker nodes via `srun bash -c "source venv && VLLM_HOST_IP=... ray start ..."`
  - Verifies cluster with `ray status`
  - Runs `vllm serve --distributed-executor-backend ray` on head node via `srun --overlap`
  - Sets required env vars: `VLLM_ALLREDUCE_USE_SYMM_MEM=0`, `NCCL_CROSS_NIC=1`, `NCCL_FORCE_FLUSH=0`
  - `compute_hostname` written to `job_details.json` is the Ray head node (tunnel target)
- [x] Added `module load brics/nccl gcc-native` to both single-node and multi-node templates
- [x] `start.ts` prints `⚠ Multi-node job: N nodes requested` when applicable
- [x] Updated `generate-vllm-config` skill: multi-node is now supported by `ivllm`

### Skills distribution

- [x] Migrated skill from `.github/skills/` to `skills/` following the `skills-npm` convention
- [x] Added `skills-npm` dev dependency and `prepare` script to `package.json`
- [x] Skill bundled in published npm package via `"files": ["skills"]`

### `generate-vllm-config` skill improvements

- [x] Step 2: check official vLLM recipes page before calculating from scratch
- [x] Expert parallelism (`--enable-expert-parallel`) guidance for MoE models
- [x] FP8 quantization guidance for memory-constrained models on H100
- [x] Tool calling: always include `enable-auto-tool-choice` + `tool-call-parser` for instruct/chat models; parser name table by model family
- [x] Prefix caching recommendation
- [x] Reasoning model parser names documented (`qwen3`, `deepseek_r1`, etc.)
- [x] Example config generated for `Qwen/Qwen2.5-0.5B-Instruct` in `examples/vllm.yaml`

### README updates

- [x] Removed `--model` from all non-mock examples (model is now in `vllm.yaml`)
- [x] Updated options table; added skill install instructions

---

## Phase F2 — CUDA forward compatibility via NVIDIA HPC SDK

One-time shared install of NVIDIA HPC SDK 26.3 into `$PROJECTDIR/ivllm/nvhpc/`.
Versioned vLLM venvs at `$PROJECTDIR/ivllm/<version>/` (cu130 wheels).
Enables CUDA 13.1 forward compatibility on Isambard AI (driver 565.57.01). See ADR-011.

Path layout (derived from `vllmVersion` config, not separately configurable):
- `$PROJECTDIR/ivllm/nvhpc/` — HPC SDK install (shared, installed once)
- `$PROJECTDIR/ivllm/<vllmVersion>/` — versioned vLLM venv (e.g. `0.19.1/`)

### F2.1 — Config changes

- [x] Remove `venvPath` from `Config` interface in `src/config.ts` (path is now derived from `vllmVersion`)
- [x] Remove `vllmVersion` from `Config` interface (see ADR-014 — version is now discovered at runtime)
- [x] Update `DEFAULTS` and config read/write to remove both `venvPath` and `vllmVersion`
- [x] Remove `--venv-path` and `--vllm-version` flags from `ivllm config` command
- [x] Write failing tests confirming `venvPath` and `vllmVersion` are absent from config schema
- [x] Confirm tests fail, implement, confirm pass

### F2.2 — Setup template

- [x] Rewrite `renderSetupScript` in `src/templates/setup.ts`:
  - Uses `--gpus=1` so `--torch-backend=auto` can detect CUDA via `nvidia-smi` (changed from CPU-only)
  - Phase A — HPC SDK install (skip if `$PROJECTDIR/ivllm/nvhpc` already exists):
    ```bash
    mkdir -p $PROJECTDIR/ivllm
    wget https://developer.download.nvidia.com/hpc-sdk/26.3/nvhpc_2026_263_Linux_aarch64_cuda_multi.tar.gz -O /tmp/nvhpc.tar.gz
    tar xpzf /tmp/nvhpc.tar.gz -C /tmp
    cd /tmp/nvhpc_2026_263_Linux_aarch64_cuda_multi
    NVHPC_SILENT=true NVHPC_INSTALL_DIR=$PROJECTDIR/ivllm/nvhpc NVHPC_INSTALL_TYPE=single ./install
    rm -f /tmp/nvhpc.tar.gz
    ```
  - Phase B — vLLM venv install (skip if `$PROJECTDIR/ivllm/<version>` already exists):
    ```bash
    module load gcc-native/14.2
    export NVHPC_ROOT=$PROJECTDIR/ivllm/nvhpc/Linux_aarch64/26.3
    export LD_LIBRARY_PATH=$NVHPC_ROOT/cuda/12.9/compat:$NVHPC_ROOT/cuda/12.9/lib64:$NVHPC_ROOT/compilers/lib:$NVHPC_ROOT/comm_libs/12.9/nccl/lib:$NVHPC_ROOT/comm_libs/12.9/nvshmem/lib:$NVHPC_ROOT/math_libs/12.9/lib64:${LD_LIBRARY_PATH:-}
    uv venv $PROJECTDIR/ivllm/<version> --python 3.12
    source $PROJECTDIR/ivllm/<version>/bin/activate
    uv pip install vllm==<version> --torch-backend=auto --extra-index-url https://wheels.vllm.ai/<version>/cu129
    ```
    Note: `cuda_multi` image used (provides both 12.9 and 13.1); `cu129` wheels used (not `cu130`)
  - Emit `IVLLM_SETUP_SUCCESS` marker on completion
- [x] Update `SetupScriptOptions` interface: remove `venvPath`; keep `vllmVersion` as the sole version field
- [x] Write failing tests in `tests/setup.test.ts`:
  - Script contains `nvhpc_2026_263_Linux_aarch64_cuda_multi`
  - Script contains `gcc-native/14.2`
  - Script contains `$PROJECTDIR/ivllm/nvhpc`
  - Script contains the versioned venv path (e.g. `$PROJECTDIR/ivllm/0.19.1`)
  - Script contains `NVHPC_ROOT` and the compat `LD_LIBRARY_PATH` (with `cuda/12.9/compat` first)
  - Script contains `uv pip install vllm==` and `wheels.vllm.ai/<version>/cu129`
  - Script does NOT contain `singularity` or `cu130`
- [x] Confirm tests fail, implement, confirm pass

### F2.3 — Setup command

- [x] Update `src/commands/setup.ts`:
  - Remove `venvPath` reference; pass `vllmVersion` to `renderSetupScript`
  - Success check: `test -d $PROJECTDIR/ivllm/<vllmVersion>/bin` on LOGIN
  - Update console output to show version and install path
- [x] Write failing tests for success check using versioned path
- [x] Confirm tests fail, implement, confirm pass

### F2.4 — Inference templates

- [x] Add `LD_LIBRARY_PATH` preamble and versioned venv activation to `renderInferenceScript`:
  ```bash
  export NVHPC_ROOT=$PROJECTDIR/ivllm/nvhpc/Linux_aarch64/26.3
  export LD_LIBRARY_PATH=$NVHPC_ROOT/cuda/13.1/compat:$NVHPC_ROOT/cuda/13.1/lib64:$NVHPC_ROOT/compilers/lib:$NVHPC_ROOT/comm_libs/13.1/nccl/lib:$NVHPC_ROOT/comm_libs/13.1/nvshmem/lib:$NVHPC_ROOT/math_libs/13.1/lib64:$LD_LIBRARY_PATH
  source $PROJECTDIR/ivllm/<vllmVersion>/bin/activate
  ```
- [x] Update single-node template: replace `venvPath` with `$PROJECTDIR/ivllm/<vllmVersion>`
- [x] Update multi-node template: same preamble before `ray start` calls and `vllm serve`
- [x] Remove `venvPath` from `InferenceScriptOptions`; add `vllmVersion: string`
- [x] Tests: `NVHPC_ROOT`, `cuda/12.9/compat`, versioned venv path, multi-node preamble order
- [x] Tests pass (180 total)

### F2.5 — vllm-config: `min-vllm-version` support

- [x] Add optional `min-vllm-version` field to `VllmConfig` interface in `src/vllm-config.ts`
- [x] `parseVllmConfig` reads and returns `minVllmVersion?: string`
- [x] Tests pass for `parseVllmConfig` with/without `min-vllm-version`

### F2.6 — AI coding assistant integration

- [x] `src/assistant.ts`: assistant detection, config generation, menu building, launch commands
  - [x] `binaryExists(name)` — checks PATH via `which`
  - [x] `getAvailableAssistants()` — returns assistant binaries filtered by PATH
  - [x] `getScoderAvailable()` — checks for `robchallen/scoder` on PATH
  - [x] `generateOpencodeConfig(opts)` — generates `opencode.json` structure with provider config, model entries, context/output limits
  - [x] `generateCopilotEnv(port, model)` — `COPILOT_PROVIDER_BASE_URL`, `COPILOT_MODEL`, `ANTHROPIC_BASE_URL`, `ANTHROPIC_API_KEY`, `CLAUDE_MODEL`
  - [x] `generateClaudeEnv(port, model)` — `ANTHROPIC_BASE_URL`, `ANTHROPIC_API_KEY`, `CLAUDE_MODEL`
  - [x] `buildAssistantMenuOptions(assistants, hasScoder)` — interleaves direct + scoder options
  - [x] `getLaunchCommand(assistant, useScoder)` — returns `{binary, args}` for tmux launch
- [x] `launchAssistantMenu()` in `src/commands/start.ts` — interactive menu loop with cwd display, change-dir `[-1]`, exit `0`
- [x] Runtime config generated for assistant launch
- [x] Launched in remote tmux window with `cd <cwd>` preamble, fallback to local tmux
- [x] `--no-launch` flag in CLI and `--mock` flag preserved for testing without GPU
- [x] 56 tests across 5 files: assistant (24), launch-assistant (12), assistant-menu (14), f26-integration (6), f26-menu (14)

### F2.6b — Launcher wrapper UX and sbx support

- [x] Write failing tests for wrapper-aware launch planning in `tests/assistant.test.ts` and `tests/launch-assistant.test.ts`:
  - [x] direct launch command renders shell-ready env exports + assistant binary
  - [x] scoder launch command renders `scoder --llm-port <port> <assistant>`
  - [x] sbx launch command renders `sbx exec -it -w <cwd> -e ... <sandbox> <agent>`
  - [x] OpenCode uses `OPENCODE_CONFIG_CONTENT` in all wrappers
  - [x] sandbox name is derived from agent + workspace basename when no existing sandbox is found
- [x] Refactor `src/assistant.ts` into wrapper-aware planning helpers:
  - [x] assistant types (`opencode`, `claude`, `copilot`)
  - [x] wrapper types (`none`, `scoder`, `sbx`)
  - [x] `generateAssistantEnv(assistant, opts)` shared entry point
  - [x] command planning helpers for display/launch plus sandbox naming and creation commands
- [x] Replace the flat launch menu in `src/commands/start.ts` with the 3-layer flow:
  - [x] layer 1 target menu
  - [x] layer 2 wrapper menu
  - [x] layer 3 action menu (`launch now`, `show command`, `back`)
- [x] Show the copy-paste command before auto-launch and allow “show command only” without starting the agent
- [x] Add sbx-specific helpers:
  - [x] detect `sbx` on PATH
  - [x] list existing sandboxes (`sbx ls --json`) and resolve by agent + workspace
  - [x] create a sandbox with `sbx create --name <agent>-<basename> <agent> <cwd>` when missing
- [x] Use `host.docker.internal:<port>` only for the `sbx` wrapper; preserve `localhost:<port>` for direct/scoder launches
- [x] Fail clearly when sbx launch prerequisites are missing (for example command not found or sandbox creation failure), but do not attempt to edit `sbx policy`
- [x] Confirm focused assistant tests pass, then run the full suite

### F2.7 — Dry-run verification

- [x] Run `ivllm start <job> --config <file> --dry-run` and verify generated SLURM script:
  - Contains `NVHPC_ROOT` and `LD_LIBRARY_PATH` with compat path first
  - Contains `source $PROJECTDIR/ivllm/<version>/bin/activate`
  - Does NOT contain `singularity` or `cu130`
- [x] Run `ivllm start <job> --mock --dry-run` and verify mock script is unchanged

### F2.8 — End-to-end testing on Isambard AI

- [x] Run `ivllm setup` — submit GPU job, verify HPC SDK at `$PROJECTDIR/ivllm/nvhpc/`
- [x] Verify versioned venv at `$PROJECTDIR/ivllm/<version>/` with vLLM installed
- [x] Run `ivllm start` with `Qwen/Qwen2.5-0.5B-Instruct` (single-node — passing)
- [x] Verify OpenAI API endpoint accessible on LOCAL via tunnel
- [x] Commit

---

## Phase F2.9 — Multi-node E2E debugging (Qwen3.5-397B-A17B-FP8)

Live debugging of 2-node `tp=4, pp=2` run. Each fix is iterative — one issue resolved reveals the next.

### Bugs fixed

- [x] **`srun env VAR=val cmd` permission denied**: `.local/bin/env` on login node inaccessible on compute nodes. Fixed: all `srun` steps use `bash -c "source venv && VAR=val cmd"`.
- [x] **`ray: command not found`**: `ray[default]` is not a hard vLLM dependency. Fixed: added `ray[default]` to setup install command.
- [x] **`--enable-reasoning` invalid vLLM flag**: Not a valid vLLM 0.19.1 option. Fixed: removed from example configs; `enableReasoning` in `vllm-config.ts` derived from presence of `reasoning-parser` key.
- [x] **`nvcc: command not found`**: Ray actors inherit `LD_LIBRARY_PATH` but not `PATH` or `CUDA_HOME`. Fixed: `CUDA_HOME` and `PATH=$CUDA_HOME/bin:$PATH` added to `NVHPC_PREAMBLE`.
- [x] **`nvcc warning: -std=c++20 not supported with host compiler`**: System gcc too old. Fixed: `module load brics/nccl gcc-native` added to both templates; `CC=gcc` and `CXX=g++` set explicitly.
- [x] **Deprecated Ray env vars**: `VLLM_USE_RAY_COMPILED_DAG`, `VLLM_USE_RAY_SPMD_WORKER`, `VLLM_USE_RAY_SPMD_HEAD` removed in vLLM 0.19.1. Fixed: removed from multi-node template.
- [x] **`cublasLt.h: No such file or directory`**: NVHPC stores math library headers in `math_libs/12.9/include/` not alongside CUDA SDK headers. Fixed: `CPATH=$NVHPC_ROOT/math_libs/12.9/include` added to `NVHPC_PREAMBLE`.
- [x] **`fused_moe_90` compile blocks Ray keepalive**: 182 nvcc invocations take ~25 min; Ray gRPC keepalive timeout fires. Fixed: redirected flashinfer JIT cache to Lustre (`FLASHINFER_JIT_CACHE_DIR`); cache persists across jobs.
- [x] **UV cache cross-filesystem copies**: uv cache on NFS home → venv on Lustre forces slow file copies. Fixed: `UV_CACHE_DIR=$LOCALDIR/uv_cache` (per-user in-job scratch). This also prevents permission conflicts when a second project member runs setup (see ADR-013).
- [x] **GDN prefill kernel NFS flock ESTALE**: `FLASHINFER_JIT_CACHE_DIR` not propagated to Ray actors by `ray_env.py`. Ray actors use default `~/.cache/flashinfer` (NFS) → `fcntl.flock` returns ESTALE → warmup fails → autotuner OOM kills actor → `Socket closed`. Fixed: symlink `~/.cache/flashinfer` → Lustre in preamble (see ADR-012).
- [x] **`NCCL_CROSS_NIC` / `NCCL_FORCE_FLUSH`**: Added per Isambard AI distributed inference guide.

### Current status

- [x] 2-node `Qwen3.5-397B-A17B-FP8` run with all fixes applied — Fully Tested, Verified, and Successfully Running!


## Phase F2.10 — Production hardening and multi-user support

### HuggingFace 429 rate limit (HF_HUB_OFFLINE)

- [x] `huggingface-hub` calls HF API to check cache freshness even when model is fully downloaded. On shared installations this triggers 429 rate-limit errors during job startup.
- [x] Fix: added `export HF_HUB_OFFLINE=1` to both single-node and multi-node inference SLURM templates (after `HF_HOME`). Safe because `ivllm start` always downloads the model on LOGIN before submitting the job.
- [x] TDD: wrote failing test; implemented; 246 tests passing (v0.2.19000)

### Multi-user uv cache permissions (`$LOCALDIR`)

- [x] First user creates `$PROJECTDIR/ivllm/uv_cache` with `drwxr-sr-x`; second user gets `Permission denied`.
- [x] Fix: changed `UV_CACHE_DIR` from `$PROJECTDIR/ivllm/uv_cache` to `$LOCALDIR/uv_cache` (per-user in-job scratch, wiped at job end). See ADR-013.
- [x] TDD: updated existing test; 246 tests passing (v0.2.20000)

### Group-write on shared install directory

- [x] `mkdir -p $PROJECTDIR/ivllm` creates the directory with user's umask (`drwxr-sr-x`). A second project member cannot create `ivllm/0.19.0/` inside it.
- [x] Fix: added `chmod g+w $PROJECTDIR/ivllm` immediately after `mkdir -p` in setup template.
- [x] TDD: added test; 247 tests passing (v0.2.21000)

### Remove `vllmVersion` from config; discover installed versions at start time (ADR-014)

- [x] `vllmVersion` removed from `Config` interface and defaults.
- [x] Added `semverGte` and `semverSort` to `src/semver.ts`.
- [x] Added `selectBestVersion(installed, min)` (exported for testing) and `listInstalledVersions` to `start.ts`; the latter SSHs `ls -d $PROJECTDIR/ivllm/*/bin` and parses version segments.
- [x] `ivllm start` discovers installed versions, filters by `min-vllm-version`, picks the highest.
- [x] `ivllm setup` takes vLLM version as positional arg (`ivllm setup 0.19.1`); errors if omitted.
- [x] Removed `--vllm-version` from `ivllm config`; updated USAGE.
- [x] TDD: new `tests/select-version.test.ts`; updated `tests/semver.test.ts`, `tests/config.test.ts`, `tests/remote-ops.test.ts`; 266 tests passing (v0.2.22000)

### HuggingFace token stored in config

- [x] `hfToken?: string` added to `Config` interface.
- [x] `ivllm config --hf-token <token>` saves token to `~/.config/ivllm/config.json`.
- [x] Setup SLURM script includes `export HF_TOKEN=<token>` when configured.
- [x] `ivllm start` model download uses `config.hfToken`; falls back to `HF_TOKEN` env var.
- [x] TDD: added tests in `tests/config.test.ts` and `tests/setup.test.ts`; 270 tests passing (v0.2.23000)


## Phase F2-alt — Singularity container support (on hold)

Preserved for future consideration. See ADR-010 for rationale.
Revisit if bare-metal pip install proves fragile across vLLM version updates,
or if multi-node container support via `brics/apptainer-multi-node` is validated with Ray.

### F2-alt.1 — Config changes

- [ ] Add `vllmImage: string` to `Config` interface (default: `"docker://vllm/vllm-openai:latest"`)
- [ ] Add `vllmImagePath: string` to `Config` interface (default: `"$PROJECTDIR/ivllm/images/vllm-openai.sif"`)
- [ ] Deprecate `venvPath` and `vllmVersion` fields (keep for backward compat but no longer used by setup/start)
- [ ] Update `DEFAULTS` in `src/config.ts`
- [ ] Add `--vllm-image` and `--vllm-image-path` flags to `ivllm config` command
- [ ] Update `src/commands/config.ts` help text

### F2-alt.2 — Setup template

- [ ] Rewrite `renderSetupScript` in `src/templates/setup.ts`:
  - CPU-only SLURM job (no `--gpus`, no GPU allocation needed for `singularity pull`)
  - `module load brics/apptainer-multi-node`
  - Create image directory: `mkdir -p $(dirname <imagePath>)`
  - `singularity pull --force <imagePath> <image>` (note: pull is slow, ~10–20 min)
  - Emit `IVLLM_SETUP_SUCCESS` marker on completion
- [ ] Update `SetupScriptOptions` interface: replace `venvPath`/`vllmVersion` with `vllmImage`/`vllmImagePath`
- [ ] Write failing tests in `tests/setup.test.ts`:
  - Script contains `singularity pull`
  - Script contains `brics/apptainer-multi-node`
  - Script does NOT contain `uv pip install` or `venv`
  - Script contains the image path and image source
- [ ] Confirm tests fail, then implement, then confirm tests pass

### F2-alt.3 — Setup command

- [ ] Update `src/commands/setup.ts`:
  - Pre-flight: check `.sif` exists at `vllmImagePath` → skip if present (idempotent); add `--force` flag to override
  - Pass `vllmImage`/`vllmImagePath` to `renderSetupScript`
  - Success check: `test -f <vllmImagePath>` on LOGIN (replaces venv check)
  - Update console output (remove venv references)
- [ ] Write failing tests in `tests/setup.test.ts` for idempotency and image path check
- [ ] Confirm tests fail, implement, confirm pass

### F2-alt.4 — Inference templates

- [ ] Add `vllmImagePath: string` to `InferenceScriptOptions` in `src/templates/inference.ts`
- [ ] Single-node template: replace `source <venv>/bin/activate && vllm serve ...` with:
  ```bash
  singularity run --nv \
    --env VLLM_ENABLE_CUDA_COMPATIBILITY=1 \
    --bind $PROJECTDIR \
    <vllmImagePath> \
    vllm serve --config <configPath> --host 0.0.0.0 --port <port>
  ```
- [ ] Multi-node template: same substitution for the `srun --overlap` vllm serve call;
  Ray head/worker `srun` calls also need `singularity exec --nv <image> /host/adapt.sh bash -c "ray start ..."` — requires `module load brics/apptainer-multi-node`
- [ ] Write failing tests in `tests/inference.test.ts`:
  - Single-node: contains `singularity run --nv`, `VLLM_ENABLE_CUDA_COMPATIBILITY=1`, `--bind $PROJECTDIR`
  - Single-node: does NOT contain `source` or `activate`
  - Multi-node: vllm serve line and ray start lines use singularity with `/host/adapt.sh`
- [ ] Confirm tests fail, implement, confirm pass

### F2-alt.5 — vllm-config: `min-vllm-version` support

- [ ] Add optional `min-vllm-version` field to `VllmConfig` interface in `src/vllm-config.ts`
- [ ] `parseVllmConfig` reads and returns `minVllmVersion?: string`
- [ ] Add `validateImageVersion(imagePath, minVersion, config)` helper:
  - Runs `singularity inspect --labels <imagePath>` on LOGIN via SSH
  - Parses `org.opencontainers.image.version` label
  - Compares semver; throws if image is too old
- [ ] Write failing tests for `parseVllmConfig` with `min-vllm-version` field
- [ ] Confirm tests fail, implement, confirm pass

### F2-alt.6 — Start command

- [ ] Update `src/commands/start.ts` pre-flight:
  - Replace venv existence check with `.sif` existence check: `test -f <vllmImagePath>`
  - If `min-vllm-version` set in yaml config, call `validateImageVersion` and fail early if not met
- [ ] HuggingFace pre-download via `singularity exec --bind $PROJECTDIR <imagePath> huggingface-cli download ...`
- [ ] Pass `vllmImagePath` from config into `renderInferenceScript`
- [ ] Write failing tests in `tests/start.test.ts` for new pre-flight check
- [ ] Confirm tests fail, implement, confirm pass

### F2-alt.7 — End-to-end testing on Isambard AI

- [ ] Run `ivllm setup` — submit CPU-only job, verify `.sif` created in `/projects/.../ivllm/images/`
- [ ] Run `ivllm start` with `Qwen/Qwen2.5-0.5B-Instruct` using container
- [ ] Verify OpenAI API endpoint accessible on LOCAL via tunnel
- [ ] Commit
