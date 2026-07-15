# Architecture Decision Records — isambard-vllm

## References

* [design/references/cuda.md] - using newer CUDA versions on isambard
* [design/references/vllm-serve-0.19.1] - vllm serve commarnd reference for v0.19.1
* [design/references/vllm-distributed.md] - describes the process of setting up inferencing on COMPUTE node and testing via a second COMPUTE node job on Isambard. This is useful for the details of how to interactively set up and start vllm. This is what we are trying to automate.
* [design/references/vllm-parallel.md] - describes the process of setting up vllm in parallel on general hardware.
* [design/references/storage.md] - info about where Isambard storage is.

## ADR-001: LOCAL CLI language — Node.js + bun

**Status**: Accepted

**Context**: The LOCAL CLI needs to manage async processes (SSH tunnel child process, heartbeat timer, stdin for user input), handle SSH subprocess execution, and potentially evolve into a lightweight API routing server.

**Decision**: Implement the LOCAL CLI in Node.js using bun as the runtime and package manager.

**Rationale**:
- Bun provides fast startup, built-in TypeScript support, and a single-binary distribution — good for a CLI tool.
- Node.js has mature primitives for managing child processes (`child_process.spawn`), async I/O, and timers, all needed for the session-owner pattern.
- Aligns with the future routing server direction (an HTTP server is trivial to add).
- LOGIN and COMPUTE scripts remain plain bash — no runtime dependency on the HPC side.

**Consequences**: Requires bun installed on LOCAL. Not natively portable to Windows (out of MVP scope).

---

## ADR-002: Session-owner pattern for `ivllm start`

**Status**: Accepted

**Context**: The vLLM inference session has a clear lifecycle: SLURM job submission → initialisation → running → shutdown. The SSH tunnel and heartbeat must be active for the duration and cleaned up reliably on exit.

**Decision**: `ivllm start` is a long-running foreground process that owns the entire session lifecycle. It does not exit until the session ends.

**Rationale**:
- Keeps tunnel management, heartbeat, and cleanup co-located in one process with straightforward signal handling.
- Avoids the complexity of a background daemon and IPC.
- Natural UX: the terminal running `ivllm start` shows live status; Ctrl+C or typing "exit" cleanly shuts down.
- `ivllm stop` exists purely as a recovery tool for unclean exits.

**Consequences**: The user must keep the terminal open for the duration of the session. A background/detach mode is a future consideration.

---

## ADR-003: `job_details.json` as the SLURM ↔ LOCAL communication channel

**Status**: Accepted

**Context**: LOCAL needs to know when vLLM is ready and what hostname/port to tunnel to. The SLURM job runs on a COMPUTE node with no direct connection back to LOCAL.

**Decision**: The SLURM script writes status updates and connection details to `job_details.json` in the job's working directory on the HPC (parallel filesystem, visible to both LOGIN and COMPUTE). LOCAL polls this file via SSH.

**Schema**:
```json
{
  "status": "pending" | "initialising" | "running" | "failed" | "timeout",
  "job_name": "<job>",
  "slurm_job_id": "<id>",
  "compute_hostname": "<hostname>",
  "server_port": 8000,
  "model": "<model-name>",
  "error": "<optional error message>"
}
```

**Rationale**:
- Simple, no additional infrastructure. The parallel filesystem (`$HOME` or `$PROJECTDIR`) is visible to all nodes.
- `jq` on the HPC makes atomic field updates straightforward in bash.
- Acts as a lockfile: existence of the file prevents duplicate jobs with the same name.

**Consequences**: LOCAL must poll via SSH (small overhead). File must be cleaned up on shutdown — handled by the shutdown sequence in `ivllm start` and as recovery in `ivllm stop`.

---

## ADR-004: Forward SSH tunnel from LOCAL

**Status**: Accepted

**Context**: COMPUTE nodes on Isambard AI cannot initiate outbound SSH connections, ruling out reverse tunnels.

**Decision**: LOCAL establishes a forward SSH tunnel once `job_details.json` reports `status: "running"`:
```
ssh -N -L <local_port>:<compute_hostname>:<server_port> <user>@<login_node>
```
This is spawned as a child process of `ivllm start` and killed as part of the shutdown sequence.

**Rationale**:
- The only viable tunnelling direction given HPC network constraints.
- Spawning as a child process ties tunnel lifetime to the `ivllm start` process.

**Consequences**: LOCAL must have SSH access to LOGIN (prerequisite, out of scope). The `ssh` binary must be available on LOCAL (standard on Linux/macOS).

---

## ADR-005: vLLM installation location on HPC

**Status**: Superseded by ADR-010

**Context**: vLLM must be installed once and reused across inference jobs. The `uv` venv created during setup must be activatable by the SLURM script.

**Decision (original)**: Install vLLM into a `uv` venv at a fixed path under `$HOME` or `$PROJECTDIR`.

**Why superseded**: The GH200 GPU driver on Isambard AI (565.57.01) only supports CUDA 12.7. Recent vLLM builds require CUDA 12.9+. pip installation of nightly vLLM either fails to build dependencies (e.g. `fastsafetensors`) or hits CUDA library version conflicts at runtime. The Isambard support team recommends using container images instead — see ADR-010.

---

## ADR-006: Fixed local port for MVP

**Status**: Accepted

**Context**: Multiple concurrent jobs with auto-assigned ports require a local registry and add complexity around OpenCode configuration.

**Decision**: MVP uses a single fixed local port (default: 11434, overridable with `--local-port`). No local registry in MVP.

**Rationale**:
- Keeps MVP scope minimal and testable.
- Port is parameterised throughout the implementation so the multi-job registry can be added in a future phase without refactoring.

**Consequences**: Running two jobs simultaneously in MVP is not supported (port conflict). Multi-job support is a tracked future phase.

---

## ADR-007: Model pre-download on LOGIN node

**Status**: Accepted

**Context**: vLLM will automatically download a model from HuggingFace if not cached, but this would occur during the SLURM job on a COMPUTE node. This wastes expensive GPU allocation time during download. COMPUTE nodes do have internet access.

**Decision**: `ivllm start` checks the shared HuggingFace cache (`$PROJECTDIR/hf`) on LOGIN before submitting the SLURM job. If the model is not cached, it runs `huggingface-cli download <model>` on LOGIN via SSH, streaming progress to the user. The SLURM script sets `HF_HOME=$PROJECTDIR/hf` so vLLM uses the pre-populated cache.

**Rationale**:
- LOGIN nodes have internet access and are not metered against GPU allocations.
- `$PROJECTDIR/hf` is shared parallel storage — one download serves all project members and all COMPUTE nodes.
- `huggingface-cli` is installed as a dependency of vLLM in the existing venv, so no extra setup is needed.
- Storing `HF_TOKEN` in `~/.config/ivllm/config.json` (via `ivllm config --hf-token`) avoids requiring the user to export an environment variable on every session; the env var is still accepted as a fallback.

**Consequences**: `ivllm start` requires a `--config` YAML containing `model:`. For gated models, run `ivllm config --hf-token <token>` once to persist the token. The token is used inline in the SSH download command and embedded in the setup SLURM script. Cache check is a simple directory existence test (`$PROJECTDIR/hf/hub/models--<org>--<name>`); a failed check falls through to `huggingface-cli download`. `HF_HUB_OFFLINE=1` is set in the inference SLURM script so the already-cached model is used without any HuggingFace API calls at inference time (prevents 429 rate-limit errors).

---

## ADR-008: vLLM config YAML as single source of truth for model and parallelism options

**Status**: Accepted

**Context**: `ivllm start` originally accepted `--model` and `--tensor-parallel-size` as CLI flags, which duplicated options that can also appear in the vLLM `--config` YAML. Passing conflicting values on both the CLI and in the YAML creates undefined behaviour.

**Decision**: All vLLM serving options (model, tensor-parallel-size, pipeline-parallel-size, max-model-len, etc.) are expressed exclusively in the vLLM config YAML. The `--model` and `--tensor-parallel-size` flags are removed from `ivllm start`. `ivllm start` parses the YAML locally (using `js-yaml`) to extract `model` (for the HuggingFace pre-download) and the parallelism sizes (to set `#SBATCH --gpus`). The SLURM script runs `vllm serve --config <file> --host 0.0.0.0 --port <port>` — the `host` and `port` flags are retained as explicit CLI overrides because they are infrastructure concerns (required for the SSH tunnel to work) rather than model configuration.

**Rationale**:
- A single config file is easier to audit, version, and share than a mix of CLI flags and YAML.
- Eliminates risk of conflicting values (e.g. `--tensor-parallel-size 4` on CLI vs `tensor-parallel-size: 2` in YAML).
- The vLLM YAML format already supports all options; users familiar with vLLM docs can use it directly.

**Consequences**: Users must include `model:` in their YAML config. `tensor-parallel-size` (and `pipeline-parallel-size`) in the YAML are used to derive the SLURM GPU allocation; `--gpus` remains as an explicit CLI override. The `--mock` mode retains `--model` as a CLI flag since it does not use a vLLM config file.

---

## ADR-009: Chat template support out of scope for MVP

**Status**: Accepted

**Context**: vLLM's `--chat-template` option accepts either a file path (a Jinja2 template file) or an inline single-line string. Some older models do not embed a chat template in their `tokenizer_config.json` and require one to be supplied explicitly.

**Decision**: Chat template file copying is out of scope for MVP. The single-line inline form is already supported at no cost (it is a plain YAML value in the config file). File-based templates are not supported — users who need them must copy the file to the HPC manually and reference its remote path in the YAML.

**Rationale**:
- Modern models (Llama 3, Qwen 2.5, Mistral, etc.) embed their chat template in the tokeniser config; vLLM picks it up automatically from the HuggingFace cache.
- The inline single-line form covers the remaining cases without requiring any additional file-copy logic.
- Adding file detection (is the `chat-template:` value a local path?) and an extra `scp` call adds complexity for an edge case that is unlikely to arise in practice on Isambard AI.

**Consequences**: If a user needs a file-based chat template, they must `scp` it to the HPC themselves and set `chat-template: /remote/path/template.jinja` in their YAML. This can be revisited if it becomes a recurring pain point during E2E testing.

---

## ADR-010: Singularity container for vLLM on HPC

**Status**: On hold — superseded by ADR-011. Preserved for future reference; Singularity remains a viable path for single-node and potentially multi-node if bare-metal limitations arise.

**Context**: The GH200 GPU driver on Isambard AI (565.57.01) supports CUDA 12.7 at most. Recent vLLM releases require CUDA 12.9+. pip installation of vLLM nightly fails:
- Build-time: `fastsafetensors` C++ extension fails to compile against system GCC
- Runtime: CUDA library version mismatches (`libnvJitLink`, `libcuda`) even with the forward-compat package

The Isambard support team recommends using Singularity/Apptainer with NGC container images. The official vLLM Docker images (`vllm/vllm-openai:<version>`) ship CUDA forward-compatibility libraries and support `VLLM_ENABLE_CUDA_COMPATIBILITY=1` to activate them automatically.

**Decision**: Replace pip/venv-based vLLM installation with a Singularity image:

1. **`ivllm setup`** submits a CPU-only SLURM job that runs `singularity pull` to convert the Docker image to a `.sif` file. The image is stored in shared project space (`/projects/<project>/ivllm/images/`). This is a one-time team operation, not per-user.

2. **SLURM inference template** runs:
   ```bash
   singularity run --nv \
     --env VLLM_ENABLE_CUDA_COMPATIBILITY=1 \
     <image.sif> vllm serve --config <config> --host 0.0.0.0 --port <port>
   ```
   instead of activating a venv.

3. **Version is specified** in `~/.ivllm/config.yaml` as `vllmImage` (default: `docker://vllm/vllm-openai:latest`). The per-job `vllm.yaml` config may specify `min-vllm-version: X.Y.Z` to reject an image that is too old; `ivllm start` reads the version label from the `.sif` and fails early if the requirement is not met.

4. **Image path** is configurable as `vllmImagePath` in `~/.ivllm/config.yaml`, defaulting to `/projects/<project>/ivllm/images/vllm-openai.sif`. Users who want a private image can point at `~/ivllm/images/` instead.

**Rationale**:
- Official vLLM images include CUDA compat libs; `VLLM_ENABLE_CUDA_COMPATIBILITY=1` handles `LD_LIBRARY_PATH` setup automatically — no manual library wrangling.
- Singularity/Apptainer is Isambard's recommended and supported container runtime for GPU jobs.
- `brics/apptainer-multi-node` module is available on Isambard for multi-node container workloads.
- Shared image in `/projects/` avoids each user pulling a multi-GB image independently.
- `singularity pull` is CPU-intensive and takes several minutes — must run on COMPUTE, not LOGIN.
- Versioned images pin the vLLM version, improving reproducibility; `min-vllm-version` in `vllm.yaml` provides a safety check.

**Consequences**:
- `ivllm setup` now submits a SLURM job rather than running pip install. It no longer has a venv to activate.
- SLURM templates (single-node and multi-node) replace `source .venv/bin/activate && vllm serve` with `singularity run --nv`.
- `ivllm start` validates the `.sif` exists (and meets `min-vllm-version` if specified) before submitting the inference job.
- `ivllm setup` is idempotent: if the `.sif` already exists and matches the configured version, it skips the pull.
- HuggingFace model cache (`$PROJECTDIR/hf`) and job working directories are bind-mounted into the container at runtime (`--bind $PROJECTDIR`).
- `huggingface-cli` used for pre-download (ADR-007) must be available outside the container on LOGIN — this may require a separate lightweight Python install for `huggingface_hub`. Alternatively, pre-download can run inside the container via `singularity exec`.

---

## ADR-011: NVIDIA HPC SDK bare-metal approach for CUDA forward compatibility

**Status**: Accepted (supersedes ADR-010)

**Context**: Two viable options were identified to run vLLM (which requires CUDA 12.9+) on Isambard AI (driver 565.57.01, max CUDA 12.7):

1. **Singularity containers** (ADR-010): clean single-node, but multi-node via Ray requires unproven `source /host/adapt.sh` pattern inside every `srun` task; large image (~10-15 GB); more `ivllm` code changes.
2. **NVIDIA HPC SDK bare-metal**: install the HPC SDK once to shared project space; set `LD_LIBRARY_PATH` to activate CUDA 12.9 forward compatibility; keep existing pip/venv + Ray architecture unchanged.

The Isambard AI documentation now explicitly documents the HPC SDK approach with the exact `LD_LIBRARY_PATH` to use, making it a low-risk, well-supported path. Multi-node Ray remains unchanged.

**Decision**: Use NVIDIA HPC SDK 26.3 (provides CUDA 12.9) installed once to shared project space. pip-install vLLM into a per-version shared venv using `cu129` wheels. The vLLM version is specified in `~/.ivllm/config.yaml`; the venv path is derived from it — no separate path config.

**Installation layout** (all under `$PROJECTDIR/ivllm/`, managed by `ivllm setup`):

```
$PROJECTDIR/ivllm/
  nvhpc/          ← NVIDIA HPC SDK 26.3 (shared, installed once)
  0.19.1/         ← uv venv with vLLM 0.19.1 (cu129)
  flashinfer_cache/  ← flashinfer JIT kernel cache (Lustre, persistent)
  uv_cache/          ← uv package cache (Lustre, enables hard links into venv)
```

The active venv path is always `$PROJECTDIR/ivllm/<vllmVersion>/`.

**`NVHPC_PREAMBLE` in all SLURM scripts** (set before `vllm serve` or `ray start`):
```bash
export NVHPC_ROOT=$PROJECTDIR/ivllm/nvhpc/Linux_aarch64/26.3
export CUDA_HOME=$NVHPC_ROOT/cuda/12.9
export PATH=$CUDA_HOME/bin:$PATH
export CPATH=$NVHPC_ROOT/math_libs/12.9/include:${CPATH:-}
export LD_LIBRARY_PATH=$NVHPC_ROOT/cuda/12.9/compat:$NVHPC_ROOT/cuda/12.9/lib64:$NVHPC_ROOT/compilers/lib:$NVHPC_ROOT/comm_libs/12.9/nccl/lib:$NVHPC_ROOT/comm_libs/12.9/nvshmem/lib:$NVHPC_ROOT/math_libs/12.9/lib64:${LD_LIBRARY_PATH:-}
export CC=gcc
export CXX=g++
export FLASHINFER_JIT_CACHE_DIR=$PROJECTDIR/ivllm/flashinfer_cache
# symlink ~/.cache/flashinfer → Lustre so Ray actors inherit it without env var propagation
ln -sfn $PROJECTDIR/ivllm/flashinfer_cache ~/.cache/flashinfer
```

**`ivllm setup`** submits a GPU SLURM job that:
1. Downloads the HPC SDK aarch64 tarball (~3 GB) and installs it to `$PROJECTDIR/ivllm/nvhpc/` (skipped if already present)
2. Sets up the compat `LD_LIBRARY_PATH` and `CUDA_HOME`
3. Loads `gcc-native` module (required to compile `fastsafetensors` and flashinfer JIT kernels)
4. Creates a uv venv at `$PROJECTDIR/ivllm/<vllmVersion>/` and pip-installs that specific vLLM version (cu129 wheels):
   ```bash
   UV_CACHE_DIR=$PROJECTDIR/ivllm/uv_cache uv pip install vllm==<version> ray[default] \
     --extra-index-url https://wheels.vllm.ai/<version>/cu129
   ```
   Note: `UV_CACHE_DIR` on Lustre enables hard links into the venv (same filesystem), avoiding slow NFS→Lustre copies.

**Per-job `vllm.yaml`** may specify `min-vllm-version: X.Y.Z`. `ivllm start` compares this against the configured `vllmVersion` and fails early if the installed version does not meet the minimum.

**Rationale**:
- HPC SDK approach is officially documented by Isambard AI with exact commands and `LD_LIBRARY_PATH`.
- Multi-node Ray architecture is unchanged — just env vars added to SLURM templates.
- Fewer `ivllm` code changes than container approach: only SLURM templates and setup script change.
- Shared project install (`$PROJECTDIR/ivllm/`) means one setup per team, not per user — same benefit as Singularity.
- Versioned venv directories allow multiple vLLM versions to coexist; upgrading is a second `ivllm setup` call.
- `vllmVersion` in config is the single source of truth; venv path is always derived — no path duplication.
- `cu129` wheels (CUDA 12.9) match the CUDA 12.9 forward compat environment provided by HPC SDK 26.3.
- Eliminates library ordering confusion: `compat` is first in `LD_LIBRARY_PATH` → `libcuda.so` from compat shadows the system stub.
- `CPATH` must include `math_libs/12.9/include` because NVHPC stores math library headers (cuBLAS, cuSPARSE, cublasLt) separately from the CUDA SDK headers — flashinfer JIT kernels include `cublasLt.h`.

**Consequences**:
- `venvPath` removed from `~/.ivllm/config.yaml`; `vllmVersion` was initially retained but is now also removed (see ADR-014).
- `ivllm setup` takes the vLLM version as a positional argument (`ivllm setup 0.19.1`); `ivllm start` discovers installed versioned venvs by listing `$PROJECTDIR/ivllm/*/bin` on the remote and selects the highest that satisfies `min-vllm-version` (see ADR-014).
- SLURM templates (single-node and multi-node) prepend the full NVHPC preamble before activating the venv and calling `vllm serve` or `ray start`.
- `ivllm setup` is idempotent: skips HPC SDK install if `$PROJECTDIR/ivllm/nvhpc` already exists; skips venv creation if `$PROJECTDIR/ivllm/<vllmVersion>` already exists.
- `fastsafetensors` and flashinfer JIT kernel build require `module load gcc-native` during setup and inference — added to both SLURM scripts.
- HuggingFace pre-download (ADR-007) uses `huggingface-cli` from the versioned venv on LOGIN.
- `ray[default]` must be explicitly installed alongside `vllm` — vLLM does not declare it as a hard dependency.
- `UV_CACHE_DIR` uses `$LOCALDIR` (per-user in-job scratch) rather than shared project space — see ADR-013.

---

## ADR-012: Multi-node Ray actor environment propagation via filesystem symlink

**Status**: Accepted

**Context**: vLLM's `ray_env.py` propagates only a fixed set of environment variable prefixes to Ray actors (`VLLM_*`, `NCCL_*`, `HF_*`, `UCX_*`, `LMCACHE_*`, plus `LD_LIBRARY_PATH` explicitly). Variables not in this list — including `FLASHINFER_JIT_CACHE_DIR`, `PATH`, `CPATH`, `CUDA_HOME`, `CC`, `CXX` — are not propagated.

When Ray actors on worker nodes call flashinfer's JIT build machinery, `FileLock` uses `fcntl.flock()` on a lock file in the flashinfer cache directory. The default cache is `~/.cache/flashinfer/` which is on NFS home on Isambard AI. NFS does not support `fcntl.flock` reliably and returns `ESTALE` (errno 116). This causes:
1. GDN prefill kernel warmup failure
2. The flashinfer autotuner runs without bounds during `determine_available_memory`
3. The Ray actor is OOM-killed
4. The actor's gRPC socket closes: `RpcError: Socket closed rpc_code: 14`

Setting `FLASHINFER_JIT_CACHE_DIR` to Lustre in the SLURM preamble fixes the login-node and head-node srun steps, but `ray_env.py` does not propagate `FLASHINFER_JIT_CACHE_DIR` to the worker actor processes.

**Decision**: Symlink `~/.cache/flashinfer` → `$PROJECTDIR/ivllm/flashinfer_cache` (Lustre) in the SLURM preamble. This runs before any Ray actor is spawned and ensures that all processes — regardless of how they were launched and regardless of env var propagation — resolve `~/.cache/flashinfer` to Lustre.

```bash
mkdir -p $PROJECTDIR/ivllm/flashinfer_cache ~/.cache
if [ -d ~/.cache/flashinfer ] && [ ! -L ~/.cache/flashinfer ]; then
  cp -r ~/.cache/flashinfer/. $PROJECTDIR/ivllm/flashinfer_cache/ 2>/dev/null || true
  rm -rf ~/.cache/flashinfer
fi
ln -sfn $PROJECTDIR/ivllm/flashinfer_cache ~/.cache/flashinfer
```

The `FLASHINFER_JIT_CACHE_DIR` env var is retained alongside the symlink as belt-and-braces.

**Rationale**:
- Symlink approach works for all consumer processes without requiring env var propagation.
- Lustre supports POSIX `fcntl.flock`; NFS home does not.
- The compiled kernel cache persists across SLURM jobs, eliminating the ~25-minute `fused_moe_90` recompile on every restart.
- Same pattern applies to `UV_CACHE_DIR` (on Lustre) to avoid NFS→Lustre cross-filesystem copies during venv install.

**Consequences**:
- The symlink is created at SLURM job startup on the head node. Worker nodes run in separate `srun` steps and have their own home directory view; the symlink on the head node does not automatically propagate to workers. However, Ray actor processes on the worker nodes inherit the srun daemon's environment. As long as the srun step that starts the Ray worker daemon also runs the preamble (which it does via `bash -c "source venv && ..."`), the symlink will be created on the worker node too.
- Stale lock files from previous failed runs should be cleaned up: `rm -rf ~/.cache/flashinfer/*.lock` before the next job.

**Amendment (June 2026)**:
To avoid group permission conflicts in shared multi-user/multi-model environments, the FlashInfer JIT cache path was relocated to use `$SCRATCHDIR` as the primary directory, with a fallback to the job's working directory (`$WORK_DIR/ivllm/flashinfer_cache`):
```bash
export FLASHINFER_JIT_CACHE_DIR=${SCRATCHDIR:-$WORK_DIR/ivllm}/flashinfer_cache
```
Using `$SCRATCHDIR` keeps compiled caches isolated per user and per project, uses Lustre (so locking/`flock` is fast and reliable), persists across runs (preventing long re-compilation wait times), and avoids using RAM-backed `$LOCALDIR` which would otherwise reduce compute node memory capacity.

Furthermore, DeepGEMM's JIT compilation of FP8 GEMM kernels at startup can hit multi-node race conditions on atomic directory renames under NFS home directories (`~/.deep_gemm`), failing with `Assertion error: runtime != nullptr`. To prevent similar JIT compilation lock and latency race conditions on PyTorch Triton kernels (`~/.triton`) or TorchInductor JIT artifacts (`~/.cache/torchinductor`), and because their standard environment variables can be stripped by vLLM's Ray worker launcher (`ray_env.py`), we apply the exact same symlinking and redirection pattern across all major JIT compilation engines:

```bash
# FlashInfer JIT
export FLASHINFER_JIT_CACHE_DIR=${SCRATCHDIR:-$WORK_DIR/ivllm}/flashinfer_cache
ln -sfn "$FLASHINFER_JIT_CACHE_DIR" ~/.cache/flashinfer

# DeepGEMM JIT
export DG_JIT_CACHE_DIR=${SCRATCHDIR:-$WORK_DIR/ivllm}/deep_gemm_cache
ln -sfn "$DG_JIT_CACHE_DIR" ~/.deep_gemm

# OpenAI Triton JIT
export TRITON_CACHE_DIR=${SCRATCHDIR:-$WORK_DIR/ivllm}/triton_cache
ln -sfn "$TRITON_CACHE_DIR" ~/.triton

# PyTorch TorchInductor JIT
export TORCHINDUCTOR_CACHE_DIR=${SCRATCHDIR:-$WORK_DIR/ivllm}/torchinductor_cache
ln -sfn "$TORCHINDUCTOR_CACHE_DIR" ~/.cache/torchinductor
```
This forces all JIT compiled kernels onto high-performance scratch space (Lustre), bypassing NFS locking/metadata caching latency and completely eliminating concurrent JIT initialization failures on multi-node runs.

Additionally, to prevent P2P IPC memory handle errors (`Cuda error ... 'invalid argument'`) during inter-node communication over HPE Slingshot interconnect, we bypass vLLM's custom all-reduce kernels during multi-node runs by exporting:
```bash
export VLLM_SKIP_CUSTOM_ALL_REDUCE=1
```
This forces vLLM to use standard, fully-supported NCCL collectives for all distributed communication, ensuring extremely robust multi-node model startup.

Additionally, to ensure reliable multi-node Ray log collection on job exits/failures, the `persist_ray_logs` diagnostic script was updated to use a standard non-login `bash -c` shell and to directly embed the literal `$RAY_DESTINATION` path rather than relying on environment propagation (`--export`), which gets stripped during nested or remote `srun` step invocations. This guarantees that diagnostics successfully execute and copy remote files back to the home directory even under strict Slurm job step cancellation contexts.

---

## ADR-013: Multi-user project space permissions for shared venv directory

**Status**: Accepted

**Context**: `$PROJECTDIR/ivllm/` is created by the first user who runs `ivllm setup`. On Isambard AI, `mkdir` creates directories with the user's umask (typically `0022` → `drwxr-sr-x`). The setgid bit on `$PROJECTDIR` ensures the group is inherited, but group write is not set. A second project member attempting to create their own versioned venv directory (e.g. `ivllm/0.19.0/`) gets `Permission denied`.

Additionally, `uv` uses a package cache to enable hard links into the venv (avoiding slow file copies). If `UV_CACHE_DIR` points to a shared location in `$PROJECTDIR`, the first user creates the cache directory with no group write, blocking other users from writing to it.

**Decision**:
1. The setup SLURM script runs `chmod g+w $PROJECTDIR/ivllm` immediately after `mkdir -p $PROJECTDIR/ivllm`, so all project members can create versioned venv subdirectories.
2. `UV_CACHE_DIR` is set to `$LOCALDIR/uv_cache` (per-user in-job scratch), not a shared location in `$PROJECTDIR`. `$LOCALDIR` is wiped at job end; the installed venv in `$PROJECTDIR/ivllm/<version>/` persists as normal.

**Rationale**:
- `chmod g+w` on the parent directory is a one-line fix that immediately unblocks all project members.
- Using `$LOCALDIR` for the uv cache sacrifices hard-link optimisation (cache on different filesystem than venv) but correctness and multi-user safety take priority. The uv cache's main value on Lustre was avoiding NFS→Lustre copies; since each user gets a fresh `$LOCALDIR` per job, the cache is ephemeral anyway.

**Consequences**:
- `$PROJECTDIR/ivllm/` is group-writable after any user's first `ivllm setup` run.
- uv downloads packages fresh into `$LOCALDIR/uv_cache` each setup run (no persistent shared cache). Installation time is slightly longer but remains correct.
- Removes the previous `UV_CACHE_DIR=$PROJECTDIR/ivllm/uv_cache` entry from the ADR-011 directory layout.

---

## ADR-014: Remove `vllmVersion` from personal config; discover installed versions at start time

**Status**: Accepted

**Context**: ADR-011 placed `vllmVersion` in `~/.config/ivllm/config.json` as the single source of truth for which vLLM version to use. This causes problems when multiple users share the same `$PROJECTDIR`:
- A second user with a different `vllmVersion` in their personal config (e.g. `0.19.0`) triggers a new setup run even though `0.19.1` is already installed and is compatible.
- `min-vllm-version` in the YAML config was already the semantic constraint — `vllmVersion` in personal config was redundant.

**Decision**:
1. Remove `vllmVersion` from `Config` interface and defaults.
2. `ivllm setup` takes the vLLM version as a positional argument: `ivllm setup 0.19.1`. There is no default — the user must be explicit.
3. `ivllm start` discovers installed versions at runtime by listing `$PROJECTDIR/ivllm/*/bin` on the remote. It selects the highest installed version that satisfies `min-vllm-version` from the YAML (or the highest overall if no minimum is set).
4. If no installed version satisfies the requirement, `ivllm start` fails with a clear message: `Run: ivllm setup <version>`.

**Rationale**:
- The remote install directory is the ground truth for what is actually available; personal config was a stale cache of that truth.
- Automatic version selection ensures that installing a newer vLLM version (`ivllm setup 0.20.0`) is immediately picked up by all users without config changes.
- `min-vllm-version` in the per-job YAML remains the right place to express feature requirements — it is close to the model config that depends on those features.

**Consequences**:
- `--vllm-version` flag removed from `ivllm config` command.
- `ivllm start` makes one additional SSH call at startup to list installed versions (negligible overhead; no Slurm API calls).
- Users who previously set `vllmVersion` in their config will find it ignored after upgrade; they should clean it out with `ivllm config` (or delete the key from `~/.config/ivllm/config.json` manually).

---

## ADR-015: AI coding assistant integration via interactive menu at `ivllm start`

**Status**: Accepted

**Context**: After vLLM reaches running state, users need to connect an AI coding assistant (OpenCode, Claude Code, GitHub Copilot) to the local endpoint. The first assistant launcher flattened the choices into a single menu and treated OpenCode as a file-writing workflow (`opencode.json` in the project). That does not scale cleanly to additional wrappers (for example `sbx`) or to per-session runtime overrides.

**Decision**: `ivllm start` presents a three-layer interactive launcher when the vLLM server is healthy:

1. **Layer 1 — target**: change directory, launch OpenCode, launch Copilot, launch Claude, or shut down `ivllm`.
2. **Layer 2 — wrapper**: choose direct launch, `scoder`, `sbx`, or go back.
3. **Layer 3 — action**: launch now, show copy-paste command, or go back.
4. Runtime configuration is injected per launch:
   - **opencode**: `OPENCODE_CONFIG_CONTENT=<json>` (highest practical runtime precedence; no project file write)
   - **copilot**: `COPILOT_PROVIDER_BASE_URL`, `COPILOT_MODEL`, plus compatibility env vars required by the CLI integration
   - **claude**: `ANTHROPIC_BASE_URL`, `ANTHROPIC_API_KEY`, `CLAUDE_MODEL`
5. The launcher always renders the full shell-ready command, including environment variables, before or alongside any automatic launch so the user can copy, paste, and customise it manually.
6. **scoder** remains a wrapper around the assistant command and receives the explicit `--llm-port <port>` argument.
7. **sbx** launches by reusing or creating a sandbox for the selected agent/workspace, then running the agent via `sbx exec -it -w <cwd> -e ... <sandbox> <agent>`.
8. `sbx` network policy remains a **user-managed prerequisite**: the user must have allowed `localhost:<ivllm-port>` before sandbox launch. `ivllm` must not edit global `sbx policy` state.
9. `--no-launch` still suppresses the menu entirely and shows the manual config snippet only.

For `sbx`, the endpoint inside the sandbox is `http://host.docker.internal:<port>` (or `/v1` for OpenCode). For direct and `scoder` launches, the endpoint remains `http://localhost:<port>`.

**Rationale**:
- Separating **assistant**, **wrapper**, and **action** keeps the UI scalable as more wrappers or agent types are added.
- Runtime env injection is a better fit than persistent config writes for per-session `ivllm` state (port, selected model, wrapper-specific hostname).
- `OPENCODE_CONFIG_CONTENT` matches OpenCode's config precedence better than writing `opencode.json` into the workspace.
- Showing the exact command lowers friction for advanced users without forcing `ivllm` to proxy arbitrary downstream agent flags.
- `sbx exec -e ...` is the correct boundary for per-launch sandbox env injection; these values should not be written into persistent sandbox state.

**Consequences**:
- `ivllm start` remains a long-running foreground process with an interactive launcher loop — the user must keep the terminal open.
- Launch code must generate wrapper-aware commands for three modes (direct, `scoder`, `sbx`) instead of a single binary+args pair.
- OpenCode launch no longer needs to modify workspace files for the normal path.
- `sbx` support depends on discovering/reusing a sandbox name derived from agent + workspace, and may need sandbox creation before the first launch.
- The user is responsible for `sbx policy allow network localhost:<port>`; if missing, `ivllm` should fail clearly rather than trying to edit policy.
