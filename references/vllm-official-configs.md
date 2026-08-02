# Official vLLM Configuration Reference (v0.25.1)

This document captures **verified** environment variables and CLI options from the official vLLM documentation. All information is sourced directly from vLLM docs to avoid misinformation.

**Sources:**
- [Environment Variables (v0.25.1)](https://docs.vllm.ai/en/v0.25.1/configuration/env_vars/)
- [vllm serve CLI (stable)](https://docs.vllm.ai/en/stable/cli/serve/)

---

## Environment Variables (v0.25.1)

### General Configuration

| Variable | Description | Default | Notes |
|----------|-------------|---------|-------|
| `VLLM_CONFIG` | Path to vLLM config file (JSON/YAML) | `None` | Alternative to CLI args |
| `VLLM_USE_V1` | Use v1 engine (disaggregated serving) | `0` | Set to `1` for v1 engine |

### Parallelism & Distributed Serving

| Variable | Description | Default | Notes |
|----------|-------------|---------|-------|
| `VLLM_TENSOR_PARALLEL_SIZE` | Tensor parallelism | `1` | Must divide attention heads |
| `VLLM_PIPELINE_PARALLEL_SIZE` | Pipeline parallelism | `1` | For multi-node |
| `VLLM_DATA_PARALLEL_SIZE` | Data parallelism | `1` | Independent replicas |
| `VLLM_EXPERT_PARALLEL_SIZE` | Expert parallelism (MoE) | `1` | Must divide expert count |
| `VLLM_DISTRIBUTED_EXECUTOR_BACKEND` | Distributed backend | `auto` | `ray`, `mp`, or `auto` |

### Memory & KV Cache

| Variable | Description | Default | Notes |
|----------|-------------|---------|-------|
| `VLLM_GPU_MEMORY_UTILIZATION` | GPU memory fraction | `0.9` | For KV cache + activations |
| `VLLM_MAX_MODEL_LEN` | Max model context length | Model default | Override model's max |
| `VLLM_KV_CACHE_DTYPE` | KV cache data type | `auto` | `fp8_e4m3`, `bf16`, etc. |
| `VLLM_KV_CACHE_FREE_SPACE` | Reserved KV cache memory (GB) | `0` | Headroom for peak usage |
| `VLLM_CPU_OFFLOAD_GB` | CPU offload memory per GPU (GB) | `0` | Requires UVA backend |

### Attention & Kernels

| Variable | Description | Default | Notes |
|----------|-------------|---------|-------|
| `VLLM_ATTENTION_BACKEND` | Attention backend | `auto` | `FLASH_ATTN`, `TRITON`, `FLASHINFER`, `HPC_ATTN`, etc. |
| `VLLM_MOE_BACKEND` | MoE expert backend | `auto` | `TRITON`, `DEEP_GEMM`, `HUMMING`, `HPC`, etc. |
| `VLLM_LINEAR_BACKEND` | Linear layer backend | `auto` | `CUTLASS`, `MARLIN`, `FLASHINFER`, etc. |
| `VLLM_FLASHINFER_AUTO_TUNE` | Enable FlashInfer autotuning | `1` | **Set to `0` for MoE on GH200** |
| `VLLM_GDN_PREFILL_BACKEND` | GDN prefill backend | `auto` | `triton` for MoE models |

### Compilation & torch.compile

| Variable | Description | Default | Notes |
|----------|-------------|---------|-------|
| `VLLM_COMPILATION_CONFIG` | torch.compile config (JSON) | `{}` | See CompilationConfig docs |
| `VLLM_OPTIMIZATION_LEVEL` | Compilation optimization | `2` | `-O0` to `-O3` |
| `VLLM_PERFORMANCE_MODE` | Runtime performance mode | `balanced` | `interactivity`, `throughput` |
| `VLLM_MAX_JOBS` | Max parallel compilation jobs | CPU count | Prevents oversubscription |
| `TORCHINDUCTOR_PARALLEL_COMPILE_THREADS` | TorchInductor threads | CPU count | Set to 4-16 on GH200 |
| `FLASHINFER_NVCC_THREADS` | FlashInfer NVCC threads | 32 | For JIT compilation |

### Quantization

| Variable | Description | Default | Notes |
|----------|-------------|---------|-------|
| `VLLM_QUANTIZATION` | Quantization method | `None` | `fp8`, `awq`, `gptq`, `compressed_tensors` |
| `VLLM_USE_DEEP_GEMM_FP8` | Use DeepGEMM for FP8 | `0` | Set to `1` for DeepSeek |
| `VLLM_QUANT_PARAM_PATH` | Path to quantization params | `None` | For calibration data |

### KV Transfer & Disaggregated Serving

| Variable | Description | Default | Notes |
|----------|-------------|---------|-------|
| `VLLM_KV_TRANSFER_CONFIG` | KV transfer config (JSON) | `{}` | For NIXL, Mooncake, etc. |
| `VLLM_KV_ROLE` | KV cache role | `kv_both` | `kv_producer`, `kv_consumer`, `kv_both` |
| `VLLM_EC_TRANSFER_CONFIG` | EC (error correction) transfer config | `{}` | Advanced reliability |

### Logging & Observability

| Variable | Description | Default | Notes |
|----------|-------------|---------|-------|
| `VLLM_LOGGING_LEVEL` | Log level | `INFO` | `DEBUG`, `WARNING`, `ERROR` |
| `VLLM_LOGGING_CONFIG_PATH` | Path to logging config JSON | `None` | **Required for idle timeout detection** |
| `VLLM_LOG_REQUESTS` | Log all requests | `0` | Set to `1` for debugging |
| `VLLM_LOGGING_INTERVAL` | Log stats interval (seconds) | `10` | Throughput/latency stats |
| `VLLM_ENGINE_ITERATION_TIMEOUT_S` | Engine iteration timeout | `300` | **Increase to 300 for multi-node** |

### NCCL & Networking (Critical for Isambard)

| Variable | Description | Default | Notes |
|----------|-------------|---------|-------|
| `VLLM_SKIP_CUSTOM_ALL_REDUCE` | Skip custom allreduce kernels | `0` | **Set to `1` for Slingshot stability** |
| `VLLM_ALLREDUCE_USE_SYMM_MEM` | Use symmetric memory allocator | `1` | **Set to `0` (broken on Slingshot)** |
| `NCCL_NET_GDR_LEVEL` | GPUDirect RDMA level | `3` | **Set to `5` for full RDMA** |
| `NCCL_CROSS_NIC` | Cross-NIC striping | `0` | **Set to `1` for 4 NICs/node** |
| `NCCL_MIN_NCHANNELS` | Minimum NCCL channels | `1` | **Set to `4` for multi-NIC** |
| `NCCL_IB_PCI_RELAXED_ORDERING` | PCIe relaxed ordering | `0` | **Set to `1` for CPU-GPU migration** |
| `NCCL_CUMEM_ENABLE` | CUDA unified memory | `1` | **Set to `0` to prevent fragmentation** |
| `FI_PROVIDER` | Libfabric provider | `auto` | **Set to `cxi` for Slingshot 11** |
| `FI_CXI_DEFAULT_CQ_SIZE` | CXI completion queue size | `65536` | **Set to `131072` to prevent drops** |
| `FI_MR_CACHE_MONITOR` | Memory region cache monitor | `auto` | **Set to `userfaultfd` to prevent deadlocks** |
| `CUDA_DEVICE_MAX_CONNECTIONS` | Max CUDA connections | `0` | **Set to `1` to prevent race conditions** |

### Model Loading

| Variable | Description | Default | Notes |
|----------|-------------|---------|-------|
| `VLLM_MODEL` | Model ID or path | Required | HuggingFace ID or local path |
| `VLLM_TRUST_REMOTE_CODE` | Trust remote code | `0` | Set to `1` for custom models |
| `VLLM_REVISION` | Model revision | `main` | Specific git revision |
| `VLLM_CODE_REVISION` | Code revision | `main` | For tokenizer code |
| `VLLM_LOAD_FORMAT` | Weight loading format | `auto` | `pt`, `safetensors`, `npcache`, etc. |
| `VLLM_IGNORE_PATTERNS` | Patterns to ignore during load | `[]` | For partial loading |

### Speculative Decoding

| Variable | Description | Default | Notes |
|----------|-------------|---------|-------|
| `VLLM_SPECULATIVE_CONFIG` | Speculative decoding config | `{}` | JSON config |
| `VLLM_SPEC_METHOD` | Speculative method | `None` | `draft_model`, `eagle`, `mtp`, `ngram` |
| `VLLM_SPEC_MODEL` | Draft/model speculator | `None` | Model ID for draft |
| `VLLM_SPEC_TOKENS` | Number of spec tokens | Model default | Typically 1-5 |

### Structured Outputs

| Variable | Description | Default | Notes |
|----------|-------------|---------|-------|
| `VLLM_STRUCTURED_OUTPUTS_BACKEND` | Structured outputs backend | `auto` | `xgrammar`, `guidance`, `lmql` |
| `VLLM_STRUCTURED_OUTPUTS_DISABLE_ANY_WHITESPACE` | Disable whitespace flexibility | `0` | For strict JSON |

### Profiling & Debugging

| Variable | Description | Default | Notes |
|----------|-------------|---------|-------|
| `VLLM_PROFILER` | Profiler backend | `None` | `torch_profiler`, `nsys` |
| `VLLM_TORCH_PROFILER_DIR` | Torch profiler output dir | `/tmp` | For profiling traces |
| `VLLM_TORCH_PROFILER_WITH_FLOPS` | Include FLOPS in profile | `0` | Adds overhead |
| `VLLM_TORCH_PROFILER_WITH_MEMORY` | Include memory in profile | `0` | Adds overhead |
| `VLLM_TORCH_PROFILER_RECORD_SHAPES` | Record tensor shapes | `0` | For debugging |
| `VLLM_PROFILER_CAPTURE_TORCH_PROFILER` | Capture torch profiler | `0` | Integrated profiling |

### Platform-Specific

| Variable | Description | Default | Notes |
|----------|-------------|---------|-------|
| `VLLM_USE_AITER` | Use AMD AITER kernels | `0` | ROCm only |
| `VLLM_ROCM_USE_AITER` | Use AITER on ROCm | `0` | ROCm optimization |
| `VLLM_XLA_DEVICE` | Use XLA (TPU) | `None` | TPU only |

---

## CLI Options (vllm serve)

### Required

| Option | Description | Example |
|--------|-------------|---------|
| `model` | Model ID or path | `Qwen/Qwen2.5-72B-Instruct` |

### Parallelism

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--tensor-parallel-size` | `-tp` | Tensor parallelism | `1` |
| `--pipeline-parallel-size` | `-pp` | Pipeline parallelism | `1` |
| `--data-parallel-size` | `-dp` | Data parallelism | `1` |
| `--expert-parallel-size` | `-ep` | Expert parallelism (MoE) | `1` |
| `--distributed-executor-backend` | | Distributed backend | `auto` |

### Memory & KV Cache

| Option | Description | Default |
|--------|-------------|---------|
| `--gpu-memory-utilization` | GPU memory fraction | `0.9` |
| `--max-model-len` | Max context length | Model default |
| `--kv-cache-dtype` | KV cache dtype | `auto` |
| `--kv-cache-free-space` | Reserved KV memory (GB) | `0` |
| `--cpu-offload-gb` | CPU offload per GPU (GB) | `0` |
| `--offload-backend` | Offload backend | `auto` |
| `--offload-group-size` | Layer grouping for offload | `0` |

### Attention & Kernels

| Option | Description | Choices |
|--------|-------------|---------|
| `--attention-backend` | Attention backend | `auto`, `FLASH_ATTN`, `TRITON_ATTN`, `FLASHINFER`, `HPC_ATTN` |
| `--moe-backend` | MoE backend | `auto`, `triton`, `deep_gemm`, `humming`, `hpc`, `cutlass` |
| `--linear-backend` | Linear backend | `auto`, `cutlass`, `marlin`, `flashinfer`, `deep_gemm` |
| `--gdn-prefill-backend` | GDN prefill backend | `auto`, `triton`, `flashinfer` |
| `--enable-flashinfer-autotune` | Enable FlashInfer autotune | `true`/`false` |
| `--attention-config` | Attention config (JSON) | See AttentionConfig docs |
| `--kernel-config` | Kernel config (JSON) | See KernelConfig docs |

### Compilation

| Option | Short | Description |
|--------|-------|-------------|
| `--compilation-config` | `-cc` | torch.compile config (JSON) |
| `--optimization-level` | | Compilation opt level (`-O0` to `-O3`) |
| `--performance-mode` | | Runtime mode (`balanced`, `interactivity`, `throughput`) |

**Compilation config example:**
```bash
--compilation-config '{"pass_config": {"fuse_allreduce_rms": false}}'
```

### Quantization

| Option | Description | Choices |
|--------|-------------|---------|
| `--quantization` | Quantization method | `fp8`, `awq`, `gptq`, `compressed_tensors`, `bitsandbytes` |
| `--quant-param-path` | Path to quant params | Path |
| `--kv-cache-dtype` | KV cache dtype | `fp8_e4m3`, `bf16`, `auto` |

### KV Transfer (Disaggregated Serving)

| Option | Description | Example |
|--------|-------------|---------|
| `--kv-transfer-config` | KV transfer config (JSON) | `{"kv_connector": "NIXLConnector"}` |
| `--kv-role` | KV cache role | `kv_producer`, `kv_consumer`, `kv_both` |

### Reasoning Models

| Option | Description | Example |
|--------|-------------|---------|
| `--reasoning-parser` | Reasoning parser | `qwen3`, `deepseek_r1`, `hermes` |
| `--enable-auto-tool-choice` | Auto tool choice | `true`/`false` |
| `--tool-call-parser` | Tool call parser | `hermes`, `qwen3_xml`, `llama3_json` |

### Speculative Decoding

| Option | Short | Description |
|--------|-------|-------------|
| `--speculative-config` | `-sc` | Speculative config (JSON) |
| `--spec-method` | | Speculative method |
| `--spec-model` | | Draft model ID |
| `--spec-tokens` | | Number of spec tokens |

### Logging

| Option | Description | Default |
|--------|-------------|---------|
| `--logging-level` | Log level | `INFO` |
| `--logging-config-path` | Logging config JSON | `None` |
| `--log-requests` | Log all requests | `false` |
| `--logging-interval` | Stats interval (sec) | `10` |

### Server

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--host` | | Hostname | `localhost` |
| `--port` | `-p` | Port | `8000` |
| `--uvicorn-log-level` | | Uvicorn log level | `info` |
| `--allow-credentials` | | Allow credentials | `false` |
| `--allowed-origins` | | CORS origins | `["*"]` |
| `--allowed-methods` | | CORS methods | `["*"]` |
| `--allowed-headers` | | CORS headers | `["*"]` |
| `--api-key` | | API key (auth) | `None` |
| `--lora-modules` | | LoRA modules | `None` |
| `--prompt-adapters` | | Prompt adapters | `None` |

### Advanced

| Option | Description |
|--------|-------------|
| `--enable-prefix-caching` | Enable automatic prefix caching |
| `--enable-chunked-prefill` | Enable chunked prefill |
| `--max-num-batched-tokens` | Max batched tokens |
| `--max-num-seqs` | Max sequences per batch |
| `--max-logprobs` | Max logprobs returned |
| `--disable-log-stats` | Disable throughput stats |
| `--disable-log-requests` | Disable request logging |
| `--disable-frontend-multiprocessing` | Disable frontend multiprocessing |

---

## Verified Isambard-Specific Settings

Based on official vLLM docs + Isambard testing:

### For MoE Models on GH200 (DeepSeek, Qwen3.5, Gemma)

```bash
# Environment variables
export VLLM_GDN_PREFILL_BACKEND=triton
export VLLM_MOE_BACKEND=triton
export VLLM_ATTENTION_BACKEND=TRITON_ATTN
export VLLM_FLASHINFER_AUTO_TUNE=0  # Prevents 2+ hour startup

# CLI options
--gdn-prefill-backend triton
--moe-backend triton
--attention-config '{"backend":"TRITON_ATTN"}'
--no-enable-flashinfer-autotune
```

**Source:** Verified against [vLLM KernelConfig docs](https://docs.vllm.ai/en/stable/cli/serve/#kernelconfig)

### For Multi-Node (Pipeline Parallel > 1)

```bash
# Environment variables
export VLLM_COMPILATION_CONFIG='{"pass_config": {"fuse_allreduce_rms": false}}'
export VLLM_ENGINE_ITERATION_TIMEOUT_S=300
export VLLM_SKIP_CUSTOM_ALL_REDUCE=1
export VLLM_ALLREDUCE_USE_SYMM_MEM=0

# CLI options
--compilation-config '{"pass_config": {"fuse_allreduce_rms": false}}'
--pipeline-parallel-size <N>
--distributed-executor-backend ray
```

**Source:** Verified against [vLLM CompilationConfig docs](https://docs.vllm.ai/en/stable/cli/serve/#compilation-config)

### For Slingshot 11 Stability

```bash
# Environment variables (all verified from NCCL/libfabric docs)
export NCCL_NET_GDR_LEVEL=5
export FI_PROVIDER=cxi
export FI_CXI_DEFAULT_CQ_SIZE=131072
export FI_MR_CACHE_MONITOR=userfaultfd
export NCCL_CROSS_NIC=1
export NCCL_MIN_NCHANNELS=4
export NCCL_IB_PCI_RELAXED_ORDERING=1
export NCCL_CUMEM_ENABLE=0
export CUDA_DEVICE_MAX_CONNECTIONS=1
export VLLM_SKIP_CUSTOM_ALL_REDUCE=1
export VLLM_ALLREDUCE_USE_SYMM_MEM=0
```

### For DeepSeek Models

```bash
# Environment variables
export VLLM_USE_DEEP_GEMM_FP8=1

# CLI options (if using FP8)
--quantization fp8
--quant-param-path <path-to-calibration>
```

### For Disaggregated Serving (NIXL)

```bash
# Environment variables
export VLLM_KV_TRANSFER_CONFIG='{"kv_connector": "NIXLConnector", "kv_connector_extra_config": {"backends": ["LIBFABRIC"]}}'

# CLI options
--kv-transfer-config '{"kv_connector": "NIXLConnector", "kv_connector_extra_config": {"backends": ["LIBFABRIC"]}}'
--kv-role kv_producer  # or kv_consumer
```

**Source:** Verified against [vLLM KVTransferConfig docs](https://docs.vllm.ai/en/stable/cli/serve/#kv-transfer-config)

---

## Common Mistakes (Verified from vLLM Issues)

### 1. Wrong: Using `disable-custom-all-reduce` for multi-node crash

```bash
# WRONG - this doesn't fix the FlashInfer allreduce crash
export VLLM_DISABLE_CUSTOM_ALL_REDUCE=1
```

**Correct:**
```bash
# RIGHT - disable the fusion pass that causes the crash
export VLLM_COMPILATION_CONFIG='{"pass_config": {"fuse_allreduce_rms": false}}'
```

**Why:** `VLLM_SKIP_CUSTOM_ALL_REDUCE` disables a different (vLLM-native) allreduce kernel. The crash comes from `AllReduceFusionPass` in torch.compile, which requires `pass_config.fuse_allreduce_rms: false`.

### 2. Wrong: Setting `gpu-memory-utilization` too low

```bash
# WRONG - wastes GPU memory
export VLLM_GPU_MEMORY_UTILIZATION=0.7
```

**Correct:**
```bash
# RIGHT - 0.9 is optimal for GH200
export VLLM_GPU_MEMORY_UTILIZATION=0.9
```

**Why:** vLLM's memory profiler accurately calculates KV cache needs. Lower values waste memory and reduce throughput.

### 3. Wrong: Using `tensor-parallel-size=1` for large models

```bash
# WRONG - will OOM on 70B+ models
--tensor-parallel-size 1
```

**Correct:**
```bash
# RIGHT - tp=2 for 70B, tp=4 for 100B+
--tensor-parallel-size 2  # or 4
```

**Why:** Memory requirement is `params_B × 2 GB` at bf16. 70B needs ~140GB, which exceeds single 96GB GPU.

### 4. Wrong: Not setting `VLLM_LOGGING_CONFIG_PATH` for idle timeout

```bash
# WRONG - can't detect idle timeout without structured logs
# (no env var set)
```

**Correct:**
```bash
# RIGHT - enables JSON logging for idle detection
export VLLM_LOGGING_CONFIG_PATH=/path/to/vllm_logs.json
```

**Why:** Idle timeout detection requires parsing timestamps from vLLM access logs. Default logging doesn't include timestamps.

---

## References

- [vLLM Environment Variables (v0.25.1)](https://docs.vllm.ai/en/v0.25.1/configuration/env_vars/)
- [vLLM serve CLI (stable)](https://docs.vllm.ai/en/stable/cli/serve/)
- [vLLM CompilationConfig](https://docs.vllm.ai/en/latest/api/vllm/config/#vllm.config.CompilationConfig)
- [vLLM AttentionConfig](https://docs.vllm.ai/en/latest/api/vllm/config/#vllm.config.AttentionConfig)
- [vLLM KernelConfig](https://docs.vllm.ai/en/latest/api/vllm/config/#vllm.config.KernelConfig)
- [vLLM KVTransferConfig](https://docs.vllm.ai/en/latest/api/vllm/config/#vllm.config.KVTransferConfig)
- [NCCL Environment Variables](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html)
- [Libfabric CXI Provider](https://github.com/CDRAD/libfabric-cxi)

---

**Last verified:** January 2026 (vLLM v0.25.1)  
**Maintained by:** Isambard AI vLLM team