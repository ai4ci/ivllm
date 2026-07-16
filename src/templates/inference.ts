import { SACCT_DIAGNOSTICS_FORMAT } from '../slurm.ts';
import type { SessionState, Paths } from '../../src/types.ts';

/**
 * Generates the NVHPC environment setup preamble for SLURM scripts.
 *
 * Produces module loads, CUDA paths, compiler settings, NCCL tuning,
 * memory hooks, and multi-NIC configuration. This is the foundation
 * for all vLLM GPU workloads on Isambard.
 *
 * | Section | Purpose |
 * |---------|--------|
 * | Module loads | `brics/nccl` + `gcc-native` for JIT compilation |
 * | CUDA paths | `NVHPC_ROOT`, `CUDA_HOME`, `PATH`, `CPATH`, `LD_LIBRARY_PATH` |
 * | Compiler | `CC=gcc`, `CXX=g++` for flashinfer/torch.compile |
 * | CPU threads | `OMP_NUM_THREADS=16` (72-core GH200, 4 GPUs) |
 * | NCCL tuning | `NCCL_NET_GDR_LEVEL=5`, `FI_PROVIDER=cxi`, multi-NIC striping |
 * | Memory hooks | `FI_MR_CACHE_MONITOR=userfaultfd` (prevents Slingshot deadlocks) |
 * | CUDA limits | `CUDA_DEVICE_MAX_CONNECTIONS=1`, `NCCL_CUMEM_ENABLE=0` |
 * | PCIe tuning | `NCCL_IB_PCI_RELAXED_ORDERING=1` |
 * | vLLM overrides | `VLLM_SKIP_CUSTOM_ALL_REDUCE=1`, `VLLM_ENGINE_ITERATION_TIMEOUT_S=300` |
 * @param ss - Session state containing `{@link Paths}` for `{@link SimplePaths.nvhpcRoot}`
 * @returns Shell script preamble string
 * @see Paths
 * @see SimplePaths
 */
function renderNVHPCPreamble(ss: SessionState) {
  return `
module purge
module load brics/nccl gcc-native
# brics/nccl sets the correct NCCL for in and between node comms
# Force NCCL to use the aws-ofi-nccl libfabric plugin:
# export NCCL_NET="AWS Libfabric"
# Use the high speed network interface
# export NCCL_SOCKET_IFNAME="hsn"
# Print the NCCL version at startup
# export NCCL_DEBUG="VERSION"
# Use P2P when GPUs share the same NUMA node
# export NCCL_NET_GDR_LEVEL="PHB"
# Allow rings/trees to span multiple NICs
# export NCCL_CROSS_NIC="1"
# export NCCL_MIN_NCHANNELS="4"
# export NCCL_GDRCOPY_ENABLE="1"
# export NCCL_NET_FORCE_FLUSH="0"

# Libfabric (FI) tuning for Slingshot
# export FI_CXI_DEFAULT_CQ_SIZE="131072"
# export FI_CXI_DEFAULT_TX_SIZE="2048"
# export FI_CXI_DISABLE_NON_INJECT_MSG_IDC="1"
# export FI_HMEM_CUDA_USE_GDRCOPY="1"
# export FI_CXI_DISABLE_HOST_REGISTER="1"
# export FI_MR_CACHE_MONITOR="userfaultfd"
# export FI_CXI_RDZV_PROTO="alt_read"
# export FI_CXI_RDZV_THRESHOLD="0"
# export FI_CXI_RDZV_GET_MIN="0"
# export FI_CXI_RDZV_EAGER_SIZE="0"
# export FI_CXI_RX_MATCH_MODE="hybrid"

export NVHPC_ROOT=${ss.paths.nvhpcRoot}
export CUDA_HOME=$NVHPC_ROOT/cuda/12.9
export PATH=$CUDA_HOME/bin:$PATH

export GLOO_SOCKET_IFNAME=hsn0
export NCCL_SOCKET_IFNAME=hsn
# Force PyTorch's internal TensorPipe layer to follow Gloo to the exact index
export TP_SOCKET_IFNAME=hsn0

# NVHPC separates math library headers (cuBLAS, cuSPARSE) from the CUDA SDK headers.
# flashinfer JIT kernels include cublasLt.h which is in math_libs, not cuda/include.
export CPATH=$NVHPC_ROOT/math_libs/12.9/include:\${CPATH:-}

# HPC-specific paths kept first so Slingshot/aws-ofi-nccl wins over generic CUDA libs.
export LD_LIBRARY_PATH=\${LD_LIBRARY_PATH:-}:$NVHPC_ROOT/cuda/12.9/compat:$NVHPC_ROOT/cuda/12.9/lib64:$NVHPC_ROOT/compilers/lib:$NVHPC_ROOT/comm_libs/12.9/nccl/lib:$NVHPC_ROOT/comm_libs/12.9/nvshmem/lib:$NVHPC_ROOT/math_libs/12.9/lib64
export CUDA_VERSION=12.9

export VLLM_ENABLE_CUDA_COMPATIBILITY=1
export VLLM_CUDA_COMPATIBILITY_PATH=$NVHPC_ROOT/cuda/12.9/compat

# Use gcc from gcc-native module for JIT compilation (flashinfer, torch.compile).
export CC=gcc
export CXX=g++

# Prevent torch from over-subscribing CPU cores across parallel workers.
# GH200 has 72 cores; 16 threads/worker is safe for the 4-GPU-per-node case.
export OMP_NUM_THREADS=16

# Force NCCL to map over the Libfabric Cassini driver (Slingshot 11)
export NCCL_NET_GDR_LEVEL=5          # Enforce full GPUDirect RDMA across nodes
export FI_PROVIDER="cxi"             # Enforce Cray Cassini fabric provider
export FI_CXI_DEFAULT_CQ_SIZE=131072 # Expand Completion Queue size to prevent dropped frames

# === Libfabric CXI Buffer Optimisations ===
# These prevent Slingshot event-queue overflows and flow-control locks
# export FI_CXI_DEFAULT_TX_SIZE=16384
export FI_CXI_DEFAULT_TX_SIZE=1024 # as per isambard_containers
export FI_CXI_DISABLE_HOST_REGISTER=1
export FI_CXI_RX_MATCH_MODE=software

# Prevent Slingshot Memory Hooks Deadlocks
# HPE Slingshot uses 'memhooks' by default, which clashes with vLLM's memory allocation and hangs.
# Switching to userfaultfd guarantees stable collective communications.
export FI_MR_CACHE_MONITOR=userfaultfd

# Handle multi-NIC striping
# Each Isambard node has 4 separate Cassini NICs (one per GH200) operating at 200Gbps.
# These ensure NCCL spreads parallel communication across all 4 rails.
export NCCL_CROSS_NIC=1
export NCCL_MIN_NCHANNELS=4

# CRUCIAL FOR 4x GH200: Ensure full NVLink cross-mesh is forced for P2P collective transfers
export NCCL_P2P_LEVEL=SYS

# prevents parallel GPU worker processes from overlapping data transfers and causing a race condition or kernel hang during deep pipeline/tensor synchronizations
export CUDA_DEVICE_MAX_CONNECTIONS=1

# Prevents catastrophic virtual memory fragmentation inside the unified space
export NCCL_CUMEM_ENABLE=0

# Relaxed ordering tells the PCIe root complex that memory pages migrating
# between the LPDDR5 CPU memory and the HBM3 GPU memory don't need strict serial locks.
export NCCL_IB_PCI_RELAXED_ORDERING=1

# VLLM networking and compilation overrides:
export VLLM_FLASHINFER_ALLREDUCE_BACKEND=trtllm # Force standard NCCL for stability across Slingshot 11
export VLLM_ENGINE_ITERATION_TIMEOUT_S=300  # Prevent timeouts during multi-node graph setups
export VLLM_ALLREDUCE_USE_SYMM_MEM=0        # Disable broken experimental symmetric memory allocator

# This forces the DeepGEMM compiler to bypass dynamic device detection and compile directly for the GH200's native SM90 architecture (in theory):

export TRITON_CUDA_ARCH=90
export TRITON_PTXAS_PATH="$NVHPC_ROOT/cuda/12.9/bin/ptxas"
export DEEPGEMM_TARGET_ARCH="sm_90"

# Update this line from 9.0 to 9.0a
export TORCH_CUDA_ARCH_LIST="9.0a"

# Force NVCC to use 90a for FlashInfer's template compilation
export NVCC_APPEND_FLAGS="-arch=sm_90a"

# 4. Limit runtime combinatorics
export FLASHINFER_HEAD_DIMS="128"
export FLASHINFER_POS_ENCODING_MODES="0"

`;
}

/**
 * Generates the exit trap handler that prints diagnostic information when the script exits.
 *
 * Produces a bash trap that outputs model details, vLLM version, and cache directory
 * on script termination (normal or error). Used by `{@link renderSingleNodeScript}` and
 * `{@link renderMultiNodeScript}` to attach to the SLURM job.
 * @returns Exit trap bash code string
 * @see renderSingleNodeScript
 * @see renderMultiNodeScript
 */
function renderExitDiagnostics(): string {
  return `SLURM_ACCOUNTING_FILE="$WORK_DIR/slurm-accounting.txt"

persist_slurm_accounting() {
  if command -v sacct >/dev/null 2>&1; then
    sacct -j "$SLURM_JOB_ID" --format=${SACCT_DIAGNOSTICS_FORMAT} > "$SLURM_ACCOUNTING_FILE" 2>&1 || true
  fi
}`;
}

/**
 * Generates the work directory setup code for SLURM scripts.
 *
 * Creates the job working directory, sets up plugin symlinks, detects and handles
 * missing LOCALDIR in interactive allocations, and configures JIT cache directories
 * for flashinfer, deep_gemm, triton, and torchinductor.
 * @param paths - Session paths containing `{@link SimplePaths.remoteJobDir}`,
 *        `{@link SimplePaths.remoteProjectVllmPluginsDir}`, and `{@link Paths.remoteProjectJobCacheFile}`
 * @returns Work directory setup bash code string
 * @see Paths
 * @see SimplePaths
 */
function renderWorkDirSetup(paths: Paths): string {
  return `
WORK_DIR="${paths.remoteJobDir}"
mkdir -p "$WORK_DIR"
# Look for and link the plugins directory
if [ -d "${paths.remoteProjectVllmPluginsDir}" ]; then
  ln -sfn "${paths.remoteProjectVllmPluginsDir}" "${paths.remoteJobVllmPluginsDir}"
fi

# 1. Purge any contaminated or inherited LOCALDIR from parent environments
unset LOCALDIR

# 2. Extract the literal, currently executing user ID dynamically
CURRENT_UID=$(id -u)

# 3. Securely isolate node workspaces strictly by the current runner's context
export LOCALDIR="/local/user/$CURRENT_UID"
export NODE_LOCALDIR="$LOCALDIR/$(hostname -s)"
mkdir -p "$NODE_LOCALDIR"

echo "=== Interactive Deployment Context Resolved ==="
echo "Node only LOCALDIR is bound to: $NODE_LOCALDIR (Executing User UID: $CURRENT_UID)"

# Clean up ONLY this specific node's unique subfolder
echo "Pristine cleanup of isolated workspace for $(hostname -s)..."
rm -rf "\${NODE_LOCALDIR:?}"/* 2>/dev/null || true

# 1. Identify possible compiler cache location
TAR_FILE="${paths.remoteProjectJobCacheFile}"

# Direct all application caches to this isolated node location
export HOME="$NODE_LOCALDIR/user_home"
mkdir -p "$HOME"

# 3. Try to restore cached JIT compilations
if [ -f "$TAR_FILE" ]; then
  echo "Restoring JIT cache from shared storage..."
  # --no-same-permissions (or -m) forces tar to map files to the current user's umask
  tar xzf "$TAR_FILE" --no-same-permissions -C "$NODE_LOCALDIR" 2>/dev/null && echo "Cache restored" || echo "Cache corrupt — recompiling"
fi


export FLASHINFER_JIT_CACHE_DIR="$HOME/.cache/flashinfer"
export DG_JIT_CACHE_DIR="$HOME/.deep_gemm"
export TRITON_CACHE_DIR="$HOME/.triton"
export TORCHINDUCTOR_CACHE_DIR="$HOME/.cache/torchinductor"

export VLLM_CACHE_ROOT="$HOME/.cache/vllm"
`;
}

// Called after renderWorkDirSetup so OK to use variables from that.
/**
 * Generates the monitoring/watchdog code for the SLURM script.
 *
 * Produces a background loop that tracks memory usage and JIT cache growth.
 * In debug mode (`IVLLM_DEBUG=true`), it produces verbose logs including NCCL
 * initialization traces, Ray IPC messages, and process memory snapshots.
 * @returns Background monitor bash code string
 * @see renderSingleNodeScript
 * @see renderMultiNodeScript
 */
function renderMonitor(): string {
  const debug = !!process.env.IVLLM_DEBUG;
  if (debug) {
    return `
export VLLM_LOGGING_LEVEL=DEBUG
# to turn on more logging.
export VLLM_LOG_STATS_INTERVAL=1.
# to get log statistics more frequently for tracking running queue, waiting queue and cache hit states.

export VLLM_TRACE_FUNCTION=1
# to record all function calls for inspection in the log files to tell which function crashes or hangs. (WARNING: This flag will slow

# 1. Force NVIDIA NCCL to print full initialization, connection, and topology maps
export NCCL_DEBUG=TRACE
export NCCL_DEBUG_SUBSYS=INIT,COLL,ENV
export NCCL_DEBUG_FILE=$WORK_DIR/nccl_log.log

# 2. Force PyTorch Distributed to log process group handshakes and tracking metrics
export TORCH_CPP_LOG_LEVEL=INFO
export TORCH_DISTRIBUTED_DEBUG=INFO
export TORCH_SHOW_CPP_STACKTRACES=1

# 3. Force Ray to log every internal actor IPC execution message
export RAY_LOG_TO_DRIVER=1

echo "=== Shared Memory Allocation (Node $SLURM_NODEID) ==="
df -h /dev/shm
# Start a background resource monitor
(
  while true; do
    echo "--- Memory Snapshot at $(date) ---"
    # Filter by your Slurm Job ID, aggregate memory by process name, and sort
    ps -u "$USER" -o rss,comm | awk '
NR>1 { mem[$2] += $1; count[$2]++ }
END {
  for (cmd in mem) {
    printf "Cmd: %-15s | Count: %-2d | Total RAM: %.2f MB\\n", cmd, count[cmd], mem[cmd]/1024
  }
}' | sort -k8 -nr | head -n 5

# JIT Cache Growth Monitor
echo "--- JIT Cache Sizes at $(date) (Node $SLURM_NODEID) ---"
du -sh $FLASHINFER_JIT_CACHE_DIR $DG_JIT_CACHE_DIR $TRITON_CACHE_DIR $VLLM_CACHE_ROOT 2>/dev/null

sleep 5
done
) &
MONITOR_PID=$!
    `;
  } else {
    // A terse memory and disk usage monitor for startup
    return `
# Compact progress monitor
(
  while true; do
    printf "[%s-node %s] RAM: %s | Cache: fi=%sK dg=%sK ti=%sK vc=%sK\\n" \\
      "$(date +%H:%M:%S)" \\
      "$SLURM_NODEID" \\
      "$(ps -u $USER -o rss=,comm= | awk '{m[$2]+=$1} END{for(c in m) if(m[c]>1024) printf "%s=%dM ",c,m[c]/1024; else printf "%s=%dK ",c,m[c]}')" \\
      "$(du -sk $FLASHINFER_JIT_CACHE_DIR 2>/dev/null | cut -f1)" \\
      "$(du -sk $DG_JIT_CACHE_DIR 2>/dev/null | cut -f1)" \\
      "$(du -sk $TRITON_CACHE_DIR 2>/dev/null | cut -f1)" \\
      "$(du -sk $VLLM_CACHE_ROOT 2>/dev/null | cut -f1)"
    sleep 20
  done
) &
MONITOR_PID=$!
`;
  }
}

/**
 * Generates the exit trap handler for the SLURM script.
 *
 * Produces bash functions `collect_exit_diagnostics`, `finalize_and_exit`, and `on_exit`,
 * then registers the trap. Handles graceful shutdown including SLURM accounting capture.
 * @returns Exit trap bash code string
 * @see renderExitDiagnostics
 */
function renderExitTrap(): string {
  return `
collect_exit_diagnostics() {
  local reason="$1"
  echo "Collecting exit diagnostics ($reason)..."
  persist_slurm_accounting
  echo "Finished exit diagnostics ($reason)."
}

finalize_and_exit() {
  local exit_code="$1"
  local reason="$2"
  trap - EXIT
  collect_exit_diagnostics "$reason"
  exit "$exit_code"
}

on_exit() {
  EXIT_CODE=$?
  echo "shutdown monitor"
  if [ -n "\${MONITOR_PID:-}" ] && kill -0 "$MONITOR_PID" 2>/dev/null; then
    kill "$MONITOR_PID" 2>/dev/null; wait "$MONITOR_PID" 2>/dev/null
  fi
  echo "shutdown vllm..."
  if [ -n "\${VLLM_PID:-}" ] && kill -0 "$VLLM_PID" 2>/dev/null; then
    kill "$VLLM_PID" 2>/dev/null; wait "$VLLM_PID" 2>/dev/null
  fi
  trap - EXIT
  collect_exit_diagnostics "EXIT trap"
  exit $EXIT_CODE
}

trap on_exit EXIT`;
}

/**
 * Generates the wait/health-check block for the SLURM script.
 *
 * In pre-cache mode (`preCache === true`), the script exits immediately after
 * vLLM becomes healthy. Otherwise, it waits for the vLLM process to exit
 * and writes failure diagnostics to the lockfile.
 * @param preCache - Whether to exit after vLLM health check passes
 * @returns Wait/health-check bash code string
 * @see renderHealthCheckAndWait
 */
function renderWaitBlock(preCache: boolean) {
  return !preCache
    ? `
# Keep SLURM job alive while vLLM runs
wait $VLLM_PID
EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
  jq --arg error "vLLM exited with code $EXIT_CODE" \\
  '.status = "failed" | .error = $error' \\
  "$JOB_DETAILS" > "$JOB_DETAILS.tmp" && mv "$JOB_DETAILS.tmp" "$JOB_DETAILS"
fi
`
    : `
# Exit immediately as unmonitored.
echo "vLLM cache compilation completed and cache saved. Closing down."
`;
}

/**
 * Generates the health check and wait logic for the SLURM script.
 *
 * Polls the `/health` endpoint at 15-second intervals, writes `running` to the
 * lockfile, archives JIT compilation caches to shared storage (tar.gz with
 * group permissions), and calls `{@link renderWaitBlock}` to complete the lifecycle.
 * @param ss - Session state containing `{@link Paths.remoteJobCacheFile}`,\n *        `{@link Paths.remoteProjectHfDir}`, and `{@link InferenceJobOptions.serverPort}`
 * @returns Health check and wait bash code string
 * @see SessionState
 * @see renderWaitBlock
 */
function renderHealthCheckAndWait(ss: SessionState): string {
  const serverPort = ss.startArgs.serverPort;

  return `
# Poll /health until vLLM is ready
echo "Waiting for vLLM to become healthy on port ${serverPort}..."
while true; do
  if curl -sf http://localhost:${serverPort}/health > /dev/null 2>&1; then
    break
  fi
  if ! kill -0 $VLLM_PID 2>/dev/null; then
    echo "vLLM process died during startup"
    jq '.status = "failed" | .error = "vLLM process died during startup"' \\
      "$JOB_DETAILS" > "$JOB_DETAILS.tmp" && mv "$JOB_DETAILS.tmp" "$JOB_DETAILS"
    finalize_and_exit 1 "startup failure"
  fi
  sleep 15
done

echo "vLLM is ready"
jq '.status = "running"' "$JOB_DETAILS" > "$JOB_DETAILS.tmp" && mv "$JOB_DETAILS.tmp" "$JOB_DETAILS"

# ═══ JIT CACHE SAVE (background) ═══
# Health check passed → all JIT compilations complete.
# Archive the tmpfs $HOME to shared storage for next cold start.
CACHE_FILE="${ss.paths.remoteProjectJobCacheFile}"
mkdir -p "${ss.paths.remoteProjectJobCacheDir}"
chmod g+w "${ss.paths.remoteProjectJobCacheDir}" 2>/dev/null || true

(
  # Stop the RAM monitor first — avoids noisy output during tar
  if [ -n "\${MONITOR_PID:-}" ] && kill -0 "$MONITOR_PID" 2>/dev/null; then
    kill "$MONITOR_PID" 2>/dev/null; wait "$MONITOR_PID" 2>/dev/null
  fi

  echo "[cache-save] Archiving JIT cache to shared storage..."

  # Kill any lingering compilation processes so the tar is clean
  sleep 2

  # 1. Force permissions inside the directory to be group-accessible before archiving
  chmod -R g+rwX "$NODE_LOCALDIR/user_home" 2>/dev/null || true

  # 2. Tar the cache while wiping out individual user metadata
  tar czf "\${CACHE_FILE}.tmp" \\
    --owner=0 --group=0 \\
    --mode='g+rwX,o-rwx' \\
  -C "$NODE_LOCALDIR" user_home 2>/dev/null && \
  mv "\${CACHE_FILE}.tmp" "$CACHE_FILE" && \\
  chmod 664 "$CACHE_FILE" && \\
  echo "[cache-save] Done: $(du -sh "$CACHE_FILE" | cut -f1)" || \\
  echo "[cache-save] Failed"
) &
disown

${renderWaitBlock(ss.startArgs.preCache)}

finalize_and_exit $EXIT_CODE "vLLM exit"`;
}

/**
 * Generates a complete SLURM batch script for single-node vLLM deployment.
 *
 * Dispatches between interactive (`isInteractive === true`) and batch modes.
 * In interactive mode, pipes output via `tee` and invokes the payload directly.
 * In batch mode, emits `#SBATCH` directives with GPU count, memory (115G/GPU
 * or exclusive for full nodes), and time limit.
 *
 * | Mode | GPU Limit | Memory |
 * |------|-----------|--------|
 * | Full node (4 GPUs) | Exclusive | 0 (unlimited) |
 * | Fractional | ≤3 GPUs | 115G per GPU |
 * @param ss - Session state containing `{@link InferenceJobOptions}` and `{@link Paths}`
 * @returns Complete SLURM bash script string
 * @see SessionState
 * @see renderSingleNodePayload
 */
function renderSingleNodeScript(ss: SessionState): string {
  const opts = ss.startArgs!;
  const paths = ss.paths;

  const runtimePayload = renderSingleNodePayload(ss);

  // Calculate resources using our fractional node logic
  const isFullNode = opts.gpuCount === 4;
  const memValue = isFullNode ? '0' : `${opts.gpuCount * 115}G`;
  // const cpusPerTask = isFullNode ? '256' : `${opts.gpuCount * 64}`;
  const exclusiveFlag = isFullNode ? '#SBATCH --exclusive\n' : '';

  if (opts.isInteractive) {
    // DIRECT INTERACTIVE ACCESS (Via SSH execution wrapper)
    // The script payload itself runs raw because your local orchestrator
    // will invoke this string via an active 'srun' command over SSH.
    return `#!/bin/bash
# Redirect stdout and stderr to both the console and the log file simultaneously
exec > >(tee "${paths.remoteJobLogFile}") 2>&1
echo "=== IVLLM version ${__VERSION__} ==="
${runtimePayload}
echo "Submitted interactive job $VLLM_PID"
    `;
  } else {
    //#SBATCH --partition=interactive
    //#SBATCH --reservation=interactive

    return `#!/bin/bash
#SBATCH --job-name=${opts.jobName}
#SBATCH --nodes=1
#SBATCH --gpus=${opts.gpuCount}
#SBATCH --mem=${memValue}
#SBATCH --cpus-per-gpu=64
#SBATCH --ntasks-per-node=1
#SBATCH --time=${opts.timeLimit}
#SBATCH --output=${paths.remoteJobLogFile}
${exclusiveFlag}

echo "=== IVLLM version ${__VERSION__} ==="
${runtimePayload}
`;
  }
}

/**
 * Generates the payload portion of a single-node SLURM script.
 *
 * Assembles the complete runtime environment: NVHPC preamble,
 * `{@link renderWorkDirSetup}` workdir setup, venv activation, HF cache config,
 * exit diagnostics, `{@link renderMonitor}` watchdog, custom env injection,
 * log env vars, vLLM launch via `srun`, and `{@link renderHealthCheckAndWait}`.
 *
 * | Component | Function |
 * |-----------|--------|
 * | NCCL/GPU setup | `{@link renderNVHPCPreamble}` |
 * | Workdir + caches | `{@link renderWorkDirSetup}` |
 * | Monitoring | `{@link renderMonitor}` |
 * | Env injection | `{@link renderCustomEnv}` |
 * | Env logging | `{@link renderLogEnvVars}` |
 * | Health check | `{@link renderHealthCheckAndWait}` |
 * @param ss - Session state containing `{@link InferenceJobOptions}`, `{@link Paths}`,\n *        and `{@link ServeOptions.model}` for the launch command
 * @returns Single-node payload bash code string
 * @see renderSingleNodeScript
 * @see renderMultiNodePayload
 */
function renderSingleNodePayload(ss: SessionState): string {
  const opts = ss.startArgs;
  const paths = ss.paths;
  const model = opts.configYaml.model;

  const lcaseModel = model.split('/').pop()!.toLowerCase();
  const maxFlash = Math.min(opts.gpuCount * 2, 4); // Safe, scalable compiler throttle
  const maxJobs = Math.min(maxFlash, 16); // Safe, scalable compiler throttle
  const maxTorch = Math.min(maxJobs * 2, 32); // Safe, scalable compiler throttle

  return `
umask 0002
${renderNVHPCPreamble(ss)}

export JOB_DETAILS="${paths.remoteJobLockFile}"
export VLLM_CONFIG="${paths.remoteJobVllmConfigFile}"
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export COMPUTE_HOSTNAME=$(hostname)

export MAX_JOBS=${maxJobs}
export TORCHINDUCTOR_PARALLEL_COMPILE_THREADS=${maxTorch}
export FLASHINFER_NVCC_THREADS=${maxFlash}
export VLLM_ENGINE_ITERATION_TIMEOUT_S=300

# Write initialising status — LOCAL already created the file with "pending";
# we overwrite with full details now that we have the SLURM context.
jq -n \\
  --arg status "initialising" \\
  --arg job_name "${opts.jobName}" \\
  --arg slurm_job_id "$SLURM_JOB_ID" \\
  --arg compute_hostname "$COMPUTE_HOSTNAME" \\
  --arg model "${model}" \\
  --argjson server_port ${opts.serverPort} \\
  '{status: $status, job_name: $job_name, slurm_job_id: $slurm_job_id,
    compute_hostname: $compute_hostname, model: $model, server_port: $server_port}' \\
  > "$JOB_DETAILS"

${renderWorkDirSetup(paths)}

source "${paths.remoteProjectVllmVenvActivate}"
export HF_HOME="${paths.remoteProjectHfDir}"
export HF_HUB_OFFLINE=1
${renderExitDiagnostics()}
${renderExitTrap()}

cd "$WORK_DIR"
${renderMonitor()}

${renderCustomEnv(ss)}

${renderLogEnvVars()}

# vLLM is launched in the background — model, tensor-parallel-size, and all tuning
# options come from the config file; host and port are infrastructure overrides.

srun --export=ALL ${ss.startArgs.isInteractive ? '--overlap ' : ''}\\
  --cpu-bind=cores \\
  vllm serve \\
  --config "$VLLM_CONFIG" \\
  --host 0.0.0.0 \\
  --port ${opts.serverPort} \\
  --served-model-name "${model}" "${lcaseModel}" "default" "${opts.jobName}" \\
  &
VLLM_PID=$!

echo "APP_PID_MATCH:$VLLM_PID"

${renderHealthCheckAndWait(ss)}`;
}

/**
 * Generates a complete SLURM batch script for multi-node vLLM deployment via Ray.
 *
 * Dispatches between interactive and batch modes. Computes `{@link nodeCount}`
 * and `{@link gpusPerNode}` from the total GPU count (max 4 GPUs per node).
 * Always uses `#SBATCH --exclusive` in batch mode.
 *
 * The payload delegates to `{@link renderInteractiveMultiNodePayload}` or
 * `{@link renderMultiNodePayload}` depending on the mode.
 *
 * | Parameter | Formula |
 * |-----------|--------|
 * | `nodeCount` | `Math.ceil(opts.gpuCount / 4)` |
 * | `gpusPerNode` | `Math.ceil(opts.gpuCount / nodeCount)` |
 * @param ss - Session state containing `{@link InferenceJobOptions}` and `{@link Paths}`
 * @returns Complete multi-node SLURM bash script string
 * @see renderSingleNodeScript
 * @see renderMultiNodePayload
 */
function renderMultiNodeScript(ss: SessionState): string {
  const opts = ss.startArgs!;
  const paths = ss.paths;
  const nodeCount = Math.ceil(opts.gpuCount / 4);
  const gpusPerNode = Math.ceil(opts.gpuCount / nodeCount);

  if (opts.isInteractive) {
    // Interactive direct access relies on your local JS script executing the parent allocation:
    // e.g. ssh user@host "srun --nodes=X --gpus-per-node=Y --mem=0 --exclusive bash -s < script.sh"
    return `#!/bin/bash
exec > >(tee "${paths.remoteJobLogFile}.$SLURM_NODEID.log") 2>&1
echo "=== IVLLM version ${__VERSION__}: Node $SLURM_NODEID ==="
${renderInteractiveMultiNodePayload(ss)}
echo "Submitted interactive job $VLLM_PID"
`;
  } else {
    // Traditional SBATCH batch script generation
    return `#!/bin/bash
#SBATCH --job-name=${opts.jobName}
#SBATCH --nodes=${nodeCount}
#SBATCH --gpus-per-node=${gpusPerNode}
#SBATCH --mem=0
#SBATCH --cpus-per-gpu=64
#SBATCH --ntasks-per-node=1
#SBATCH --time=${opts.timeLimit}
#SBATCH --exclusive
#SBATCH --output=${paths.remoteJobLogFile}

echo "=== IVLLM version ${__VERSION__} ==="
${renderMultiNodePayload(ss)}
`;
  }
}

/**
 * Generates custom environment variable injection from the YAML config.
 *
 * Produces `export KEY=VALUE` statements for all entries in
 * `{@link ServeOptions.env}`. These variables are passed to the vLLM process
 * and are specific to each job configuration.
 * @param ss - Session state containing `{@link InferenceJobOptions}` and
 *        `{@link ServeOptions.env}` for custom env var injection
 * @returns Environment variable export statements
 * @see ServeOptions
 * @see renderSingleNodePayload
 */
function renderCustomEnv(ss: SessionState) {
  const envVars = ss.startArgs.configYaml.env;
  return envVars.length > 0
    ? envVars.map((e) => `export ${e.key}="${e.value}"`).join('\n')
    : '';
}

/**
 * Generates environment variable logging for the SLURM script.
 *
 * Produces an `echo` block that outputs all `VLLM_`, `RAY_`, `NCCL_`, and
 * `FI_` prefixed environment variables to the job log file. Used for
 * diagnostics and debugging vLLM deployment configuration.
 * @returns Environment logging bash code string
 * @see renderSingleNodePayload
 */
function renderLogEnvVars() {
  return `
echo "=== Python version ==="
python -c "
import torch, deep_gemm, os
print('PyTorch CUDA:', torch.version.cuda)
print('CUDA_HOME:', os.environ.get('CUDA_HOME'))
print('DeepGEMM Version:', deep_gemm.__version__)
print('Device Compute Capability:', torch.cuda.get_device_capability())
"
echo "=== Final Environment Variables for vLLM ==="
env | grep -E "VLLM_|RAY_|NCCL_|FI_|NVHPC|CUDA_|LD_CONFIG|CPATH|PATH|SLURM_|TRITON"
echo "============================================"
`;
}

/**
 *
 * @param opts
 * @param cmd
 */
// function renderMultiNodePayload(ss: SessionState): string {
//   const opts = ss.startArgs;
//   const paths = ss.paths;
//   const nodeCount = Math.ceil(opts.gpuCount / 4);
//   const gpusPerNode = Math.ceil(opts.gpuCount / nodeCount);
//
//   const rayObjectStoreMemoryGiB = 4;
//
//   // const cpusPerTask = '256';
//   const model = opts.configYaml.model;
//   const lcaseModel = model.split('/').pop()!.toLowerCase();
//
//   // Multi node compilation caching:
//   // E.g. Node 1 and Node 2 write to independent, clean compilation caches
//   // This is achieved by rewriting $HOME for each worker to $NODE_LOCALDIR which is node local
//   // Each node restores caches from $PROJECTDIR/ivllm/caches
//
//   return `
// umask 0002
// ${renderNVHPCPreamble(ss)}
//
// export JOB_DETAILS="${paths.remoteJobLockFile}"
// export VLLM_CONFIG="${paths.remoteJobVllmConfigFile}"
// export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
// export GPUS_PER_NODE=${gpusPerNode}
// export HEAD_NODE=$(scontrol show hostnames $SLURM_NODELIST | head -n1)
// export COMPUTE_HOSTNAME=$HEAD_NODE
// export HEAD_NODE_IP=$(getent hosts "$HEAD_NODE" | awk '{ print $1 }')
// export RAY_PORT=6378
//
// # Write initialising status — LOCAL already created the file with "pending";
// # we overwrite with full details now that we have the SLURM context.
// jq -n \\
//   --arg status "initialising" \\
//   --arg job_name "${opts.jobName}" \\
//   --arg slurm_job_id "$SLURM_JOB_ID" \\
//   --arg compute_hostname "$COMPUTE_HOSTNAME" \\
//   --arg model "${model}" \\
//   --argjson server_port ${opts.serverPort} \\
//   '{status: $status, job_name: $job_name, slurm_job_id: $slurm_job_id,
//     compute_hostname: $compute_hostname, model: $model, server_port: $server_port}' \\
//   > "$JOB_DETAILS"
//
// ${renderWorkDirSetup(paths)}
//
// source "${paths.remoteProjectVllmVenvActivate}"
// export HF_HOME=${paths.remoteProjectHfDir}
// export HF_HUB_OFFLINE=1
//
// ${renderExitDiagnostics()}
// ${renderExitTrap()}
//
// export RAY_OBJECT_STORE_MEMORY=${rayObjectStoreMemoryGiB * 1024 * 1024 * 1024}
// export RAY_LOG_TO_DRIVER=1    # Forces worker logs to stream to the head node stdout
// export RAY_RUNTIME_ENV_LOG_TO_DRIVER=1
// # Change the default temp storage location to a persistent cluster directory
// export RAY_TMPDIR="${ss.paths.remoteJobDir}/ray-logs"
//
// cd "$WORK_DIR"
// ${renderMonitor()}
//
// ${renderCustomEnv(ss)}
//
// ${renderLogEnvVars()}
//
// srun --export=ALL ${ss.startArgs.isInteractive ? '--overlap ' : ''}\\
//   --output=vllm_node_%N.log \\
//   --error=vllm_node_%N.err \\
//   --cpu-bind=cores \\
//   ray symmetric-run \\
//   --address "$HEAD_NODE_IP:$RAY_PORT" \\
//   --min-nodes $SLURM_NNODES \\
//   --num-gpus $GPUS_PER_NODE \
//   -- \\
//   vllm serve \\
//   --config $VLLM_CONFIG \\
//   --distributed-executor-backend ray \\
//   --host 0.0.0.0 \\
//   --port ${opts.serverPort} \\
//   --served-model-name \"${model}\" \"${lcaseModel}\" \"default\" \"${opts.jobName}\" \\
// &
// VLLM_PID=$!
//
// echo "APP_PID_MATCH:$VLLM_PID"
//
// ${renderHealthCheckAndWait(ss)}`;
// }

function escapeQuote(cmd: string): string {
  return cmd.replaceAll('"', '\\"');
}

/**
 * Generates the interactive multi-node payload for SLURM execution.
 *
 * Similar to `{@link renderMultiNodePayload}` but designed for interactive
 * `srun` allocations: pipes vLLM output via `tee`, runs vLLM in the foreground,
 * and delegates to `{@link renderHealthCheckAndWait}` for health tracking.
 *
 * Unlike batch mode, no `#SBATCH` directives are emitted — the caller
 * (`{@link renderMultiNodeScript}`) handles those.
 * @param ss - Session state containing `{@link InferenceJobOptions}`, `{@link Paths}`,\n *        and `{@link ServeOptions.model}` for Ray + vLLM interactive launch
 * @returns Interactive multi-node payload bash code string
 * @see renderMultiNodeScript
 * @see renderMultiNodePayload
 */
function renderInteractiveMultiNodePayload(ss: SessionState): string {
  const opts = ss.startArgs;
  const paths = ss.paths;
  const nodeCount = Math.ceil(opts.gpuCount / 4);
  const gpusPerNode = 4;

  const rayObjectStoreMemoryGiB = 4;

  // const cpusPerTask = '256';
  const model = opts.configYaml.model;
  const lcaseModel = model.split('/').pop()!.toLowerCase();
  const maxFlash = 2; // Safe, scalable compiler throttle
  const maxJobs = 8; // Safe, scalable compiler throttle
  const maxTorch = 16; // Safe, scalable compiler throttle

  // Slightly confusingly this will be executed for each node in parallel
  return `
umask 0002
${renderNVHPCPreamble(ss)}

export JOB_DETAILS="${paths.remoteJobLockFile}"
export VLLM_CONFIG="${paths.remoteJobVllmConfigFile}"
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export GPUS_PER_NODE=${gpusPerNode}
export HEAD_NODE=$(scontrol show hostnames $SLURM_NODELIST | head -n1)
export COMPUTE_HOSTNAME=$HEAD_NODE

export HEAD_NODE_IP=$(dig +short $HEAD_NODE)
export RAY_PORT=6378

${renderWorkDirSetup(paths)}
${renderCustomEnv(ss)}

export MAX_JOBS=${maxJobs}
export TORCHINDUCTOR_PARALLEL_COMPILE_THREADS=${maxTorch}
export FLASHINFER_NVCC_THREADS=${maxFlash}
export VLLM_ENGINE_ITERATION_TIMEOUT_S=300

# Export the workspace directory so all spawned Python/Ray processes load sitecustomize.py
export PYTHONPATH="${paths.remoteProjectVllmPluginsDir}:$PYTHONPATH"

source "${paths.remoteProjectVllmVenvActivate}"
export HF_HOME=${paths.remoteProjectHfDir}
export HF_HUB_OFFLINE=1

export RAY_OBJECT_STORE_MEMORY=${rayObjectStoreMemoryGiB * 1024 * 1024 * 1024}
export RAY_LOG_TO_DRIVER=1    # Forces worker logs to stream to the head node stdout
cd "$WORK_DIR"

${renderMonitor()}
${renderExitDiagnostics()}
${renderExitTrap()}


if [ "$SLURM_NODEID" -eq 0 ]; then

  echo "This is the master node."
  echo "=== Node environment: $SLURM_NODEID ===="
  ${renderLogEnvVars()}

  # Write initialising status — LOCAL already created the file with "pending";
  # we overwrite with full details now that we have the SLURM context.
  jq -n \\
  --arg status "initialising" \\
  --arg job_name "${opts.jobName}" \\
  --arg slurm_job_id "$SLURM_JOB_ID" \\
  --arg compute_hostname "$COMPUTE_HOSTNAME" \\
  --arg model "${model}" \\
  --argjson server_port ${opts.serverPort} \\
  '{status: $status, job_name: $job_name, slurm_job_id: $slurm_job_id,
    compute_hostname: $compute_hostname, model: $model, server_port: $server_port}' \\
    > "$JOB_DETAILS"

  cd "${paths.remoteJobDir}"

  VLLM_HOST_IP=$HEAD_NODE_IP vllm serve \\
  --nnodes ${nodeCount} \\
  --node-rank $SLURM_NODEID \\
  --master-addr $HEAD_NODE_IP \\
  --config $VLLM_CONFIG \\
  --port ${opts.serverPort} \\
  --served-model-name "${model}" "${lcaseModel}" "default" "${opts.jobName}" \\
  &
  VLLM_PID=$!

  ${renderHealthCheckAndWait(ss)}

else

  WORKER=$(scontrol show hostnames "$SLURM_NODELIST" | sed -n "$((SLURM_NODEID + 1))p")
  WORKER_IP=$(dig +short $WORKER)
  echo "Starting worker node: $WORKER ($WORKER_IP) attached to $HEAD_NODE_IP"

  VLLM_HOST_IP=$WORKER_IP vllm serve \\
  --nnodes ${nodeCount} \\
  --node-rank $SLURM_NODEID \\
  --headless \\
  --master-addr $HEAD_NODE_IP \\
  --config $VLLM_CONFIG \\
  --served-model-name "${model}" "${lcaseModel}" "default" "${opts.jobName}" \\
  &
  VLLM_PID=$!

  # Keep worker alive while vLLM runs
  wait $VLLM_PID
  EXIT_CODE=$?

fi
`;
}

/**
 * Generates the standard multi-node payload for batch SLURM execution.
 *
 * Coordinates Ray cluster startup across multiple compute nodes: launches the
 * Ray head node on the first node, worker nodes on the remainder (via `for` loop
 * over `scontrol show hostnames`), verifies cluster readiness, then starts vLLM
 * with `--distributed-executor-backend ray`.
 *
 * | Phase | Nodes | Command |
 * |-------|-------|--------|
 * | Head node | Node 0 (1 node) | `ray start --head` via `srun` |
 * | Workers | Nodes 1–N | `ray start --address=$HEAD_NODE_IP:$RAY_PORT` via `srun` |
 * | vLLM serve | Head node | `vllm serve --distributed-executor-backend ray` |
 *
 * Uses `{@link escapeQuote}` to safely escape custom env vars inside `bash -c` strings.
 * @param ss - Session state containing `{@link InferenceJobOptions}`, `{@link Paths}`,\n *        and `{@link ServeOptions.model}` for Ray + vLLM launch
 * @returns Multi-node payload bash code string
 * @see renderMultiNodeScript
 * @see escapeQuote
 */
function renderMultiNodePayload(ss: SessionState): string {
  const opts = ss.startArgs;
  const paths = ss.paths;
  const nodeCount = Math.ceil(opts.gpuCount / 4);
  const gpusPerNode = Math.ceil(opts.gpuCount / nodeCount);

  const rayObjectStoreMemoryGiB = 4;

  // const cpusPerTask = '256';
  const model = opts.configYaml.model;
  const lcaseModel = model.split('/').pop()!.toLowerCase();
  const maxFlash = 4; // Safe, scalable compiler throttle
  const maxJobs = 16; // Safe, scalable compiler throttle
  const maxTorch = 32; // Safe, scalable compiler throttle

  // TODO: Fix multinode $LOCALDIR collisions
  // Multi node compilation caching:
  // E.g. Node 1 and Node 2 write to independent, clean compilation caches
  // This is achieved by rewriting $HOME for each worker to $LOCALDIR which is node local
  // Each node restores caches from $PROJECTDIR/ivllm/caches

  return `
umask 0002
${renderNVHPCPreamble(ss)}


export MAX_JOBS=${maxJobs}
export TORCHINDUCTOR_PARALLEL_COMPILE_THREADS=${maxTorch}
export FLASHINFER_NVCC_THREADS=${maxFlash}
export VLLM_ENGINE_ITERATION_TIMEOUT_S=300

export JOB_DETAILS="${paths.remoteJobLockFile}"
export VLLM_CONFIG="${paths.remoteJobVllmConfigFile}"
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export GPUS_PER_NODE=${gpusPerNode}
export HEAD_NODE=$(scontrol show hostnames $SLURM_NODELIST | head -n1)
export COMPUTE_HOSTNAME=$HEAD_NODE
export HEAD_NODE_IP=$(getent hosts "$HEAD_NODE" | awk '{ print $1 }')
export RAY_PORT=6378

# Write initialising status — LOCAL already created the file with "pending";
# we overwrite with full details now that we have the SLURM context.
jq -n \\
--arg status "initialising" \\
--arg job_name "${opts.jobName}" \\
--arg slurm_job_id "$SLURM_JOB_ID" \\
--arg compute_hostname "$COMPUTE_HOSTNAME" \\
--arg model "${model}" \\
--argjson server_port ${opts.serverPort} \\
'{status: $status, job_name: $job_name, slurm_job_id: $slurm_job_id,
compute_hostname: $compute_hostname, model: $model, server_port: $server_port}' \\
> "$JOB_DETAILS"

${renderWorkDirSetup(paths)}

source "${paths.remoteProjectVllmVenvActivate}"
export HF_HOME=${paths.remoteProjectHfDir}
export HF_HUB_OFFLINE=1

${renderExitDiagnostics()}
${renderExitTrap()}

export RAY_OBJECT_STORE_MEMORY=${rayObjectStoreMemoryGiB * 1024 * 1024 * 1024}
export RAY_LOG_TO_DRIVER=1    # Forces worker logs to stream to the head node stdout
export RAY_RUNTIME_ENV_LOG_TO_DRIVER=1
# Change the default temp storage location to a persistent cluster directory
export RAY_TMPDIR="${ss.paths.remoteJobDir}/ray-logs"

cd "$WORK_DIR"
${renderMonitor()}

${renderCustomEnv(ss)}

${renderLogEnvVars()}

# Start Ray head node
# bash -c is used to guarantee venv PATH is active on the compute node,
# and to avoid any .local/bin/env shadowing /usr/bin/env on login nodes.
echo "Starting Ray head node ($HEAD_NODE)..."
srun --overlap \\
  --nodelist "$HEAD_NODE" \\
  --nodes=1 \\
  --gpus=$GPUS_PER_NODE \\
  --mem=0 \\
  --cpus-per-gpu=64 \\
  --ntasks-per-node=1 \\
  bash -c "
    source ${paths.remoteProjectVllmVenvActivate}
    ${escapeQuote(renderCustomEnv(ss))} \\
    VLLM_HOST_IP=$HEAD_NODE_IP ray start \\
      --block \\
      --head \\
      --node-ip-address=$HEAD_NODE_IP \\
      --port=$RAY_PORT \\
      --object-store-memory=$RAY_OBJECT_STORE_MEMORY
    " &
sleep 20

# Start Ray worker nodes
WORKER_NODES=$(scontrol show hostnames $SLURM_NODELIST | tail -n+2)
for WORKER in $WORKER_NODES; do
  WORKER_IP=$(dig +short $WORKER)
  echo "Starting Ray worker node: $WORKER ($WORKER_IP)"
  srun --overlap \\
    --nodelist "$WORKER" \\
    --nodes=1 \\
    --gpus=$GPUS_PER_NODE \\
    --mem=0 \\
    --cpus-per-gpu=64 \\
    --ntasks-per-node=1 \\
    bash -c "
      source ${paths.remoteProjectVllmVenvActivate}
      ${escapeQuote(renderCustomEnv(ss))}
      ${escapeQuote(renderWorkDirSetup(paths))}
      VLLM_HOST_IP=$WORKER_IP ray start \\
        --block \\
        --address=$HEAD_NODE_IP:$RAY_PORT \\
        --node-ip-address=$WORKER_IP \\
        --object-store-memory=$RAY_OBJECT_STORE_MEMORY
      " &
done
sleep 20

# Verify Ray cluster is ready
srun --overlap \\
  --nodelist "$HEAD_NODE" \\
  --nodes=1 \\
  --gpus=$GPUS_PER_NODE \\
  --mem=0 \\
  --ntasks-per-node=1 \\
  bash -c "source ${paths.remoteProjectVllmVenvActivate} && ray status"

# Start vLLM on the head node via srun --overlap (runs within existing job allocation)
srun --overlap \\
  --nodelist "$HEAD_NODE" \\
  --nodes=1 \\
  --gpus=$GPUS_PER_NODE \\
  --mem=0 \\
  --ntasks-per-node=1 \\
  bash -c "
    cd ${paths.remoteJobDir}

    source ${paths.remoteProjectVllmVenvActivate}
    ${escapeQuote(renderCustomEnv(ss))}
    VLLM_HOST_IP=$HEAD_NODE_IP vllm serve \\
      --config $VLLM_CONFIG \\
      --distributed-executor-backend ray \\
      --host 0.0.0.0 \\
      --port ${opts.serverPort} \\
      --served-model-name \\"${model}\\" \\"${lcaseModel}\\" \\"default\\" \\"${opts.jobName}\\"
  " \\
  &
VLLM_PID=$!

echo "APP_PID_MATCH:$VLLM_PID"

${renderHealthCheckAndWait(ss)}`;
}

// function renderMultiNodeExitDiagnostics(): string {
//   return `${renderExitDiagnostics()}
//   RAY_LOG_ARCHIVE_DIR="$WORK_DIR/ray-logs"
//
//   persist_ray_logs() {
//     mkdir -p "$RAY_LOG_ARCHIVE_DIR"
//     for NODE_NAME in $(scontrol show hostnames $SLURM_NODELIST); do
//       RAY_DESTINATION="$RAY_LOG_ARCHIVE_DIR/$NODE_NAME"
//       ARCHIVE_STATUS_FILE="$RAY_DESTINATION/archive-status.txt"
//       mkdir -p "$RAY_DESTINATION"
//       printf "%s\\n" "Starting Ray log archival for $NODE_NAME" > "$ARCHIVE_STATUS_FILE"
//       if srun --overlap \\
//         --nodelist "$NODE_NAME" \\
//         --nodes=1 \\
//         --ntasks-per-node 1 \\
//         --cpus-per-task 1 \\
//         bash -c '
//     RAY_SESSION_DIR=$(readlink -f /local/user/$UID/ray/session_latest 2>/dev/null || true)
//     RAY_DEST_LITERAL="'"\$RAY_DESTINATION"'"
//     mkdir -p "\$RAY_DEST_LITERAL"
//     if [ -n "\$RAY_SESSION_DIR" ] && [ -d "\$RAY_SESSION_DIR/logs" ]; then
//       cp -a "\$RAY_SESSION_DIR/logs/." "\$RAY_DEST_LITERAL/" 2>/dev/null || true
//       printf "%s\\n" "\$RAY_SESSION_DIR" > "\$RAY_DEST_LITERAL/session_dir.txt"
//       else
//         printf "%s\\n" "No Ray logs found at /local/user/$UID/ray/session_latest" > "\$RAY_DEST_LITERAL/missing.txt"
//         fi
//         '; then
//         printf "%s\\n" "Finished Ray log archival for $NODE_NAME" >> "$ARCHIVE_STATUS_FILE"
//         else
//           printf "%s\\n" "Ray log archival srun failed for $NODE_NAME" >> "$ARCHIVE_STATUS_FILE"
//           fi
//           done
//   }`;
// }

// # $SLURM_JOB_UID is native to Slurm and guaranteed to exist on the compute node
// LOCAL_RAY_DIR="/local/user/${SLURM_JOB_UID}/ray/session_latest"
//
// RAY_SESSION_DIR=$(readlink -f "$LOCAL_RAY_DIR" 2>/dev/null || true)
// mkdir -p "$REMOTE_RAY_DEST"
//
// if [ -n "$RAY_SESSION_DIR" ] && [ -d "$RAY_SESSION_DIR/logs" ]; then
//   cp -a "$RAY_SESSION_DIR/logs/." "$REMOTE_RAY_DEST/" 2>/dev/null || true
//   printf "%s\n" "$RAY_SESSION_DIR" > "$REMOTE_RAY_DEST/session_dir.txt"
//   else
//     printf "%s\n" "No Ray logs found at $LOCAL_RAY_DIR" > "$REMOTE_RAY_DEST/missing.txt"
//     fi

/**
 * Top-level entry point — generates a complete SLURM script for vLLM deployment.
 *
 * Dispatches based on GPU count: `{@link renderSingleNodeScript}` for ≤4 GPUs
 * (single-node), `{@link renderMultiNodeScript}` for >4 GPUs (multi-node via Ray).
 *
 * | GPU Count | Mode | Backend |
 * |-----------|------|--------|
 * | ≤4 | Single-node | `srun vllm serve` |
 * | >4 | Multi-node | `ray symmetric-run vllm serve` |
 * @param opts - Session state containing `{@link InferenceJobOptions}` and `{@link Paths}`
 * @returns Complete SLURM bash script string
 * @see renderSingleNodeScript
 * @see renderMultiNodeScript
 */
export function renderInferenceScript(opts: SessionState): string {
  if (opts.startArgs === undefined)
    throw new Error('SessionState not correctly set up');
  return (
    opts.startArgs.gpuCount > 4
      ? renderMultiNodeScript(opts)
      : renderSingleNodeScript(opts)
  ).trimStart();
}
