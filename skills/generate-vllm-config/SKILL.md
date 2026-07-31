---
name: generate-vllm-config
description: 'Use when asked to generate, create, or write a vLLM config, vllm.yaml, or configuration file for a HuggingFace model to run on Isambard AI HPC. Triggers on phrases like "generate a config for <model>", "create a vllm.yaml for <model>", "what config do I need for <model>", "set up config for llama/qwen/mistral/deepseek". Do NOT use for general vLLM questions unrelated to config file generation.'
license: MIT
metadata:
  author: Rob Challen
  version: 0.1
---

# Generate vLLM Config for Isambard AI

Generates a ready-to-use `vllm.yaml` config file for running a HuggingFace model on Isambard AI HPC using `ivllm`. It fetches the model card to determine architecture and parameter count, calculates memory requirements against Isambard AI's GH200 120GB nodes (~96 GiB usable HBM3e per GPU), selects appropriate parallelism, and writes the YAML file.

## ⚡ Quick Reference — Key Decisions

> Read this first. These are the decisions that cause the most problems when done wrong.

### Quantization

FP8 is King on Isambard-AI: The absolute fastest way to run quantized models on a GH200 is FP8 W8A8 (8-bit weights, 8-bit activations). This precision format triggers the GH200's native Hopper Transformer Engine tensor cores, achieving roughly double the tokens-per-second compared to running smaller quantisations.

If you are picking out checkpoints to run on Isambard-AI, look for models compressed using LLM Compressor into FP8 format (often labeled as neuralmagic or vllm-compatible-fp8 on Hugging Face) for maximum scaling performance.

NVFP4 is not supported on the GH200. The NVIDIA GH200 Grace Hopper Superchip is based on the Hopper architecture (Compute Capability 9.0). Hopper natively supports standard FP16, BF16, and FP8 (E4M3/E5M2) precisions, but it lacks the specialized execution units required for 4-bit floating-point data.

For Maximum VRAM Savings (4-Bit): GPTQ INT4 or AWQ INT4. If you must shrink a model down to 4-bit to fit into memory, use standard integer-based quantization formats like AWQ or GPTQ. vLLM handles these formats smoothly on Hopper by mapping them to optimized Machete or Marlin GPU inference kernels. This is only worth doing where a model has about 400B parameters and would be too large for one node unquantised.


```
                     Is your target model > 70B parameters?
                                       │
                      ┌────────────────┴────────────────┐
                      ▼ YES                             ▼ NO
        Can it fit into 1 Node at FP8?         Run on a SINGLE NODE
          (Available VRAM ~380GB total)         with TP=4 or lower.
                      │                        Maximum performance.
             ┌────────┴────────┐
             ▼ YES             ▼ NO
     Use FP8 Quantization.     Can it fit into 1 Node at AWQ-INT4?
     Keep it on ONE node.      (e.g., MiniMax-M3 ~258GB Model Footprint)
     Set TP=4. Max speed.                      │
                                      ┌────────┴────────┐
                                      ▼ YES             ▼ NO
                              Use AWQ-INT4 Quantization. Must scale to MULTI-NODE.
                              Keep it on ONE node.       (e.g., GLM-5.2 FP8)
                              Set TP=4. Avoids network   Set TP=4 (within node)
                              packet latency penalty.    & PP across nodes via
                                                         optimized Slingshot fabric.
```

### Parallelism (the #1 most common mistake)

| GPUs needed | Default `tensor-parallel-size` | Notes |
|-------------|-------------------------------|-------|
| 1 GPU | `tensor-parallel-size: 1` | Only if model fits and user wants minimal resources |
| 1–2 GPUs | **`tensor-parallel-size: 2`** (default) | Good balance of queue time and capacity |
| 3–4 GPUs | **`tensor-parallel-size: 4`** (full node) | Required to fit, or user explicitly wants max throughput |
| >4 GPUs | **`tensor-parallel-size: 4`** + `pipeline-parallel-size: N` | Multi-node — warn the user first |

**Rule of thumb**: `needed_gpus = ceil(params_B × 2 / 86.4)`. Then apply the table above.

### Critical gotchas (config errors that cause real failures)

| Issue | Impact | Fix |
|-------|--------|-----|
| **MoE + flashinfer** | 2+ hour JIT compilation hang | Must set `gdn-prefill-backend`, `moe-backend`, `attention-config`, `enable-flashinfer-autotune: false` (see Step 7) |
| **`-cc.dotted.shorthand`** | vLLM startup crash | Expand to long form: `-cc.key=val` → `compilation-config: '{"key": val}'` (see Step 8) |
| **`tensor-parallel-size` does not divide attention heads** | vLLM crash on weight loading | tp must be 1, 2, or 4 and must evenly divide `num_attention_heads` |
| **`min-vllm-version` too low** | Unknown key / missing flag error | Set to earliest version that supports the model + features |

### Defaults (don't change these unless you have a reason)

| Setting | Default | When to change |
|---------|---------|----------------|
| `max-model-len` | Full native context length | Only if explicit KV cache OOM calculation shows it won't fit |
| `gpu-memory-utilization` | `0.90` | Rarely |
| `dtype` | `bfloat16` | Use `fp8` only if memory-constrained and FP8 variant exists |
| `enable-auto-tool-choice` | `true` | Only for base/pretrain checkpoints or pure reasoning models |

### CPU offload threshold

The OffloadingConnector extends the prefix cache by offloading completed KV blocks to slower but larger tiers (CPU host memory, plus optional secondary tiers) as they are produced. Hits in the offload tiers are promoted back to GPU on demand. Transfers between GPU and CPU use DMA (cudaMemcpyAsync) and run asynchronously alongside model computation, so offloading adds minimal CPU- and GPU-core overhead.

```
vllm serve <model> \
  --kv-transfer-config '{
    "kv_connector": "OffloadingConnector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {
      "block_size": 64,
      "cpu_bytes_to_use": 1000000000
    }
  }'
```

cpu_bytes_to_use: a bigger CPU tier means fewer trips to slower secondary tiers and a higher hit rate. The value is total across all workers, not per-worker. Leave headroom for the rest of the host workload. For single-tier (CPU-only) setups, set cpu_bytes_to_use larger than the aggregate GPU KV cache. Because offloading is immediate, a smaller CPU tier just mirrors what the GPU already holds and adds no hit rate.

---

## When to Use This Skill

- User asks to generate or create a vLLM config or `vllm.yaml` for a named model
- User asks "what config do I need to run \<model\> on Isambard?"
- User is about to run `ivllm` and needs a config file
- User provides a HuggingFace model ID and wants to know how many GPUs are needed

Do NOT use for general vLLM help, `ivllm` troubleshooting, or non-config questions.

## Prerequisites

- **Model ID**: A HuggingFace model ID (e.g. `Qwen/Qwen2.5-72B-Instruct`). Ask the user if not provided.
- **Output filename**: Ask the user — default is `vllm.yaml` in the current directory.
- **Web access**: Required to fetch the model card from `https://huggingface.co/<model-id>`.
- **Write permission**: To save the YAML file.

## Steps

### 1. Confirm inputs

If the model ID was not provided as an argument, ask the user:
- Which HuggingFace model they want to use
- What filename to write to (default: `vllm.yaml`)

### 2. Fetch the model card

Fetch `https://huggingface.co/<model-id>` and extract:

| Field | Where to find it | Example |
|-------|-----------------|---------|
| Total parameter count | Model Overview table or description | "72.7B parameters" |
| Architecture type | Description — is it dense or MoE? | "Mixture of Experts" |
| Active parameters (MoE only) | Model Overview table | "3.3B activated" |
| Native context length | Model card — "Context Length" | "128,000 tokens" |
| Recommended dtype | Model card or vLLM quickstart snippet | "bfloat16" |
| Reasoning/thinking support | Any mention of `--reasoning-parser`, `enable_thinking`, or chain-of-thought | Yes/No |

If the model card is not accessible or the parameter count cannot be determined, ask the user to provide it.

### 3. Look up the official vLLM recipe (if available)

Go to [recipes.vllm.ai](https://recipes.vllm.ai) — this is the authoritative source for vLLM deployment recipes. It supersedes checking HuggingFace model cards for configuration advice, but both sources may be required.

The site has a four-level structure:

1. **Index** — `https://recipes.vllm.ai/models.json` lists every model with a recipe. Use this to check whether a recipe exists for your model. Look up the model's `"json"` path (e.g. `"/Qwen/Qwen3.5-397B-A17B.json"`) — this is the URL path segment to append to `https://recipes.vllm.ai`.
2. **Top-level recipe** — `https://recipes.vllm.ai/<path>.json` contains the full recipe with model metadata, features, hardware overrides, and variants.
3. **Hardware-specific overrides** — `https://recipes.vllm.ai/<path>/hw/h200.json` provides a flat, fully resolved deployment snapshot with all overrides already applied.
4. **Strategy-specific** — `https://recipes.vllm.ai/<path>/strategies/<strategy>.json` provides alternative deployment strategies (e.g. multi-node, different parallelism approaches).

#### Three JSON types and jq paths

The top-level recipe and the hardware/strategy-specific files have **completely different structures**. The top-level recipe is nested and contains raw, un-resolved data; the hardware/strategy files are flat snapshots with overrides already baked in.

**Top-level recipe JSON** (`/<org>/<model>.json`):

```json
.model.model_id                      # HuggingFace model ID
.model.min_vllm_version              # Minimum vLLM version required
.model.architecture                  # "moe" or "dense"
.model.parameter_count               # e.g. "1600B"
.model.active_parameters             # e.g. "49B" (MoE only)
.model.context_length                # Native context length (integer)
.model.base_args[]                   # Always-applied CLI args
.recommended_command.env             # Top-level env vars
.recommended_command.argv[]          # CLI args (default strategy)
.recommended_command.hardware_profile.gpu_count   # GPUs per node
.hardware_overrides.hopper.extra_args[]           # Extra CLI args for Hopper
.hardware_overrides.hopper.extra_env              # Extra env vars for Hopper
.features.tool_calling.args[]        # Tool-calling CLI args
.features.reasoning.args[]           # Reasoning CLI args
.variants.default.precision          # "fp8", "bf16", etc.
.variants.default.vram_minimum_gb    # Min VRAM for this variant
```

**Hardware-specific JSON** (`/<org>/<model>/hw/h200.json`):

This is a **resolved snapshot** — all hardware overrides are already applied. No `model`, `features`, or `hardware_overrides` keys exist here.

```json
.env                                 # Env vars (overrides already included)
.argv[]                              # Resolved CLI args (all overrides baked in)
.node_count                          # Nodes required
.strategy                            # Strategy name
.strategy_spec.name                  # Strategy display name
.strategy_spec.description           # What the strategy does
.strategy_spec.vllm_args[]           # vLLM-specific flags only (e.g. --enable-expert-parallel)
.hardware_profile.gpu_count          # GPUs per node
.hardware_profile.vram_gb            # Total VRAM across all GPUs
.hardware_profile.multi_node         # true/false
.alternatives                        # Links to other strategy JSON files
```

**Strategy-specific JSON** (`/<org>/<model>/strategies/<strategy>.json`):

Same base as hardware-specific, with **head/worker split for multi-node**:

```json
.env                                 # Env vars (includes GLOO/NCCL_SOCKET_IFNAME for multi-node)
.head_argv[]                         # CLI args for head node
.worker_argv[]                       # CLI args for worker node(s)
.node_count                          # Total nodes
.strategy_spec.name                  # Strategy name
.strategy_spec.vllm_args[]           # vLLM-specific flags
.hardware_profile.gpu_count          # GPUs per node
.hardware_profile.multi_node         # true/false
```

#### Which JSON to use

- **Model metadata** (parameter count, architecture, context length): **top-level recipe** — the hardware-specific file doesn't include these.
- **Env vars and CLI args for your config**: **hardware-specific** (`/hw/h100.json` or `/hw/h200.json`) — overrides are already resolved, so you don't need to manually merge `hardware_overrides.hopper.extra_env` with the base `env`.
- **Multi-node strategies**: **strategy-specific** files — they include `head_argv`/`worker_argv` and multi-node env vars.

**Adapting to Isambard AI:** Isambard AI's GH200 (Grace CPU + H100 GPU) is not listed as a separate hardware target. It falls between H100 and H200 in specs (closer to H100), so use the H100 recipe as a starting point and adapt, but vllm recipes usually assume 8 independent H100 GPUs whereas Isambard has trays of 4 GH200 closely wired together. Their tensor parallelisation choices may not be correct for Isammbard.

Each node has 4 NVIDIA GH200 Grace Hopper Superchips with NVLink-C2C interconnect. Each node has 460 GB of usable CPU memory (115 GB per CPU) and 384 GB of GPU memory (96 GiB usable per GPU). In total, there is 844 GB of CPU + GPU memory per node.

The recipes are written for various hardware; translate GPU counts to Isambard AI node counts using: `nodes = ceil(recipe_gpus / 4)`.

### 4. Establish vLLM minimum version

- **min-vllm-version**: Set this to the minimum vLLM version required to run the model. `ivllm start` checks this against the installed vLLM version and fails early if not satisfied. This key is **stripped before the config is passed to `vllm serve`** (vLLM errors on unknown keys). Use the earliest vLLM version in which the model and any required parsers/features were added. If unsure, check the vLLM changelog or recipes page, or set to `"0.19.1"` as a conservative baseline (the current default install version).

vLLM CLI options and config keys vary between releases. Always consult the documentation for the specific version being deployed:

- **All versions index**: https://app.readthedocs.org/projects/vllm/
- **Versioned serve CLI docs**: `https://docs.vllm.ai/en/v<version>/cli/serve/`
  e.g. https://docs.vllm.ai/en/v0.19.1/cli/serve/

When checking whether a feature, parser, or config key exists, verify it in the docs for the version specified by `min-vllm-version` in the config (or the user's configured `vllm-version`). Avoid referencing `/en/latest/` when the deployed version is fixed.

### 5. Calculate memory and parallelism

Use the rules in [references/vllm-config-guide.md](references/vllm-config-guide.md):

```
weights_GB ≈ total_params_B × 2          (bfloat16, both dense and MoE)
usable_per_gpu = 86.4 GB                 (96 GB × 0.90 utilization; GH200 reports ~95.6 GiB usable)
needed_gpus = ceil(weights_GB / usable_per_gpu)
```

**GH200 unified memory + CPU offload (`--cpu-offload-gb`):**

The GH200's NVLink-C2C (900 GB/s coherent interconnect) creates a unified memory address space between CPU (460 GB) and GPU (96 GB). The `--cpu-offload-gb` flag reserves CPU memory space per GPU for model weights, accessed via UVA zero-copy during forward passes. This is effectively a virtual way to increase GPU memory — e.g., 24 GB GPU + 10 GB offload = 34 GB virtual capacity.

Use `--cpu-offload-gb` if:
- The model's weight memory exceeds what fits on available GPUs even at fp8
- The user explicitly requests it or asks about memory optimization for very large models
- It requires vLLM ≥ 0.6.0 (feature availability)

**Example:**
```yaml
model: meta-llama/Llama-3.1-70B
tensor-parallel-size: 1               # 70B @ bf16 ≈ 140 GB; exceeds single 96GB GPU
# On single GPU with offload:
cpu-offload-gb: 50                  # reserves 50 GB CPU memory (passed as CLI arg)
gpu-memory-utilization: 0.90        # uses ~86 GB GPU (140-50 = 90 GB, close to limit)
```

When reporting to the user, include:
- "CPU offload: cpu-offload-gb <N>"
- "Requires fast CPU-GPU interconnect (NVLink-C2C on GH200)"
- "Minimum vLLM version: 0.6.0+"

**Single node** (needed_gpus ≤ 4):

⚠️ **Parallelism is the most common config error.** Using `tensor-parallel-size: 1` when the model needs 2 GPUs will cause an OOM crash that wastes the user's time debugging. Always apply these rules:

- **Default to `tensor-parallel-size: 2`** for models needing 1–2 GPUs. This gives a good balance of capacity and good citizenship.
- Use `tensor-parallel-size: 4` (full node) if:
  - The model needs 3–4 GPUs to fit, **or**
  - The model needs 1–2 GPUs but the user explicitly wants maximum throughput or longest possible context, **or**
  - The model's number of attention heads is not divisible by 2 but is divisible by 4 (rare; check the model card)
- `pipeline-parallel-size` = 1 (omit from config)

**Multi-node** (needed_gpus > 4):
- `tensor-parallel-size` = 4
- `pipeline-parallel-size` = ceil(needed_gpus / 4)
- ⚠️ **Warn the user**: multi-node jobs require more resources and inter-node communication adds latency. `ivllm start` supports multi-node via Ray automatically, but confirm the user needs multi-node before proceeding.

### 6. Choose max-model-len

- **Default to the model's full native context length.** KV cache headroom depends on `tensor-parallel-size` and whether weights fit with room to spare. Do not reduce the context pre-emptively.
- Only reduce `max-model-len` if an explicit OOM analysis shows the KV cache at native context would exhaust available memory after weights are loaded. Calculate:

```
kv_per_token ≈ num_kv_heads × head_dim × 2 bytes × num_layers  (standard dense attention)
total_kv_GB  = kv_per_token × max_tokens / 1e9
available_for_kv = (usable_per_gpu × tensor_parallel_size) − weights_GB  # scales with tp  # scales with tp
```

- If the native context does exceed available KV budget, reduce to the largest power-of-two that fits, and note the native context in a YAML comment.
- Exception: hybrid architectures (e.g. Qwen3.5-35B-A3B with Gated DeltaNet layers) have a tiny KV footprint — keep the full context.

### 7. Check for special options

- **Reasoning models** (Qwen3, DeepSeek-R1, QwQ, etc.): add the `reasoning-parser`. Use `qwen3` for Qwen3 series; `deepseek_r1` for DeepSeek-R1/V3 and most others. Check the model card vLLM quickstart snippet for the exact name.

🚨 **MoE models — flashinfer JIT hang (this causes 2+ hour startup delays)**:

If the model is MoE **and** `tensor-parallel-size >= 2`, **you must** set these four keys to force Triton kernels and disable the autotune profiling loop:

```yaml
gdn-prefill-backend: "triton"
moe-backend: triton
attention-config: '{"backend":"TRITON_ATTN"}'
enable-flashinfer-autotune: false
```

This applies to DeepSeek, Qwen3.5, Gemma, and all MoE models on Isambard AI. The autotune loop takes >2 hours on these models; the Triton kernels are pre-compiled and work immediately.

If you do end up having to use them they will be cached and re-used in theory but if the user is complaining about long start up times this is where to start.

- **FP8 quantization**: GH200/H100 (Hopper) has native FP8 tensor cores. If the model is memory-constrained or throughput is important, suggest `quantization: fp8`. This halves weight memory (`params_B × 1 GB` vs `× 2 GB`). Check if a pre-quantized `-FP8` variant exists on HuggingFace — prefer it over runtime quantization.
- **Tool calling**: Always include `enable-auto-tool-choice: true` and the matching `tool-call-parser` unless the model is known not to support function calling (e.g. base/pretrain checkpoints, pure reasoning models without tool support). The parser is required — without it, tool call responses come back as raw text rather than structured `tool_calls` objects. See the tool-call parser table in `references/vllm-config-guide.md`.
- **Prefix caching**: Recommend `enable-prefix-caching: true` for agent/chatbot use cases with repeated system prompts. Low cost, high benefit.
- **Environment variables (`env:`)**: Some models require environment variables to be set before `vllm serve` starts. These are specified in the `env:` block of the config. The `env:` key is **ivllm-specific** — it is stripped from the YAML before passing to `vllm serve` (vLLM errors on unknown keys) and rendered as `export` lines in the SLURM script. For multi-node jobs, env vars are automatically propagated into every Ray worker `bash -c` invocation.

Check the vLLM recipe for recommended env vars. Common examples:
```yaml
env:
  VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS: "1"
  VLLM_USE_DEEP_GEMM_FP8: "1"
```

  Always check the model's vLLM recipe page for any `extra_env` recommendations. Values should be quoted strings. Only include env vars that are explicitly recommended by the recipe or required for the model to run correctly.

### 8. Convert dotted shorthand options to YAML

**CRITICAL**: vLLM recipes sometimes use dotted shorthand notation for nested config values. THESE WILL NOT WORK IN THE ivllm YAML config. The recipe might show:

`-cc.pass_config.fuse_allreduce_rms=False`

Where `-cc` is shorthand for `--compilation-config` and the dotted suffix sets nested keys. Convert via the verbose long form:

`--compilation-config '{"pass_config": {"fuse_allreduce_rms": false}}'`

Then use as a YAML key:

`compilation-config: '{"pass_config": {"fuse_allreduce_rms": false}}'`

🚨 **Compilation-config shorthand gotcha — this causes vLLM startup crashes**:

Any dotted shorthand (`-cc.key=val`) **must** be expanded to its long-form flag (`--cc '{"key": "val"}'`), then converted to a YAML key (`compilation-config: '{"key": "val"}'`).

**GOTCHA — merging with existing values**: The recipe may already have a `--compilation-config` value that must be merged, not replaced. For example, the DeepSeek-V4-Pro recipe has:

`--compilation-config '{"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY"}'`

When adding the pass config option above, merge the keys into a single JSON object:

`compilation-config: '{"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY", "pass_config": {"fuse_allreduce_rms": false}}'`

**General rule**: So far we have only encountered this with `-cc`/`--compilation-config`, but the same pattern likely applies to any similar dotted shorthand.

### 9. Write the YAML file

Write the config to the requested filename. Include comments to explain the key choices:

```yaml
# vLLM config for <model-id> on Isambard AI
# Generated by ivllm generate-vllm-config
#
# Hardware: 4 × GH200 120GB per node (Isambard AI Phase 1/2) — ~96 GiB usable HBM3e per GPU
# Model: <X>B parameters, ~<Y> GB at bfloat16
# Parallelism: <explanation>

model: <model-id>
tensor-parallel-size: 4
# pipeline-parallel-size: <M>   # for multi-node: each pipeline stage = 1 node (tp=4)
max-model-len: <native_context_length>
gpu-memory-utilization: 0.90
dtype: bfloat16
enable-auto-tool-choice: true
tool-call-parser: <parser>      # see vllm-config-guide.md for parser names
enable-prefix-caching: true
min-vllm-version: "<version>"
# env:                              # uncomment if the vLLM recipe recommends env vars
#   VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS: "1"
# min-vllm-version: minimum vLLM version to run this model; ivllm checks before
# submitting. Stripped before passing to vllm serve (vLLM errors on unknown keys).
```

Only include keys that are non-default or important — keep the config minimal and readable.

### 10. Report to the user

After writing the file, tell the user:
- The filename written
- The model size and memory calculation
- The parallelism chosen and why (tp=4 is default; note if reduced and why)
- How to use it: `ivllm start <job-name> --config <filename>`
- Any warnings (multi-node required, context was reduced from native, etc.)

## Guidance

### Memory calculation principles

The rule of thumb `params_B × 2 GB` for bfloat16 is conservative (it's exact for weights; the actual overhead includes activations and KV cache, handled by `gpu-memory-utilization`). When in doubt, round up.

For MoE models, all expert weights must reside in GPU memory simultaneously even though only a few activate per forward pass — use total parameter count, not active parameter count, for the memory calculation.

### Flashinfer on Isambard

Flash infer compilation seems to have particular problems on isambard with start up times exceeding 2 hours if it is enabled. This is mainly due a mixture of experts models. The following set of options must be applied to all MOE models.

```
# Mixture of experts models need to be forced to use triton
gdn-prefill-backend: "triton"
moe-backend: triton             # Forces vLLM to use pre-compiled Triton MoE kernels
attention-config: '{"backend":"TRITON_ATTN"}'
enable-flashinfer-autotune: false  # Explicitly kills the profiling loop
```

### CPU offload (`--cpu-offload-gb`)

**Key facts from NVIDIA's GH200 unified memory research:**
- NVLink-C2C connects CPU and GPU at 900 GB/s — 7× PCIe Gen 5 bandwidth, memory-coherent
- Creates a single unified memory address space: CPU (120 GB) + GPU (96 GB) share one page table and 4x superchips per node
- `--cpu-offload-gb` uses UVA (Unified Virtual Addressing) for zero-copy access — weights are loaded from CPU memory to GPU memory on-the-fly during each forward pass
- The offload parameter is "virtual GPU memory" — e.g., 24 GB GPU + 10 GB offload = 34 GB virtual capacity
- Requires fast CPU-GPU interconnect (NVLink-C2C on GH200 makes this practical)

**Configuration options (from vLLM OffloadConfig):**

| Option | Description | Default | Notes |
|--------|-------------|---------|-------|
| `--cpu-offload-gb` | CPU memory (GiB) per GPU for weights | 0 | Must use UVA backend |
| `--offload-backend` | "auto", "prefetch", or "uva" | auto | "uva" for cpu-offload-gb; "prefetch" for async |
| `--offload-group-size` | Group N layers, offload last N of each group | 0 | Uses async prefetch, not UVA |
| `--offload-num-in-group` | Layers to offload per group | 1 | Must be ≤ offload-group-size |

**When to suggest CPU offload:**

💡 If `params_B × 2 > 96 GB` and the user wants a **single-node job**, suggest `cpu-offload-gb`.

Example: Llama-3-70B at 140 GB vs 96 GB GPU needs ~44 GB offload.

- Model exceeds single-GPU memory (e.g., Llama-3-70B at 140 GB vs 96 GB GPU)
- Single-node job desired but weights don't fit
- User asks about optimization for large models

**When NOT to suggest:**
- Multi-node is appropriate (better throughput)
- fp8 quantization solves the problem

**Note:** `--cpu-offload-gb` maps to `cpu-offload-gb` in YAML config — all vLLM CLI arguments work as YAML keys (with `--` stripped).

### Parallelism choices

**The default is tp=4.** The allocation on isambard is for a 4 GPU node. Smaller requests are worth considering if the KV cache is under-utilised and we want to be good citizens on Isambard:

4 nodes gives us
- 4× more aggregate KV cache memory, enabling longer contexts and larger batches
- Higher throughput via NVLink-C2C-connected tensor parallelism (near-linear scaling)
- Full use of the allocated node allocation

Common choices:
- **tp=1**: model fits on a single GPU (e.g., < ~40B at bf16, < ~80B at fp8) and the user explicitly wants minimal resource usage.
- **tp=2**: default for models needing 1–2 GPUs (e.g., 72B at bf16). Good balance of queue time and capacity.
- **tp=4**: full node. For models needing 3–4 GPUs, or when the user explicitly wants maximum throughput or very long contexts.

Multi-node (pp>1) adds inter-node communication latency and complexity — only use it when the model's weights genuinely won't fit on a single 4-GPU node.

Valid `tensor-parallel-size` values must divide the model's number of attention heads evenly. 1, 2, and 4 work for virtually all modern models.

### Context length

**Default to the model's full native context length.** With tp=4 across 4 × GH200 120GB (~345 GB usable), there is substantial KV cache headroom. The point of running on Isambard AI is to go big — large contexts, high throughput.

Only reduce `max-model-len` if an explicit calculation shows the KV cache at native context would exhaust remaining memory after weights are loaded. If reduction is needed, note the native context in a comment and reduce to the next power-of-two that fits. Never silently cap at 32K for a model that supports 128K+ unless forced by memory.

### Multi-node parallelism

Default to `tensor-parallel-size: 4` to match Isambard's 4xGH200 superchips per node. Then scale accross nodes using pipeline parallelism, until you have the total memory that the model and caches require.

`ivllm start` supports multi-node SLURM jobs via Ray. When `pipeline-parallel-size` * `tensor-parallel-size` > 4, it automatically sets `--nodes=N` and bootstraps a Ray cluster across those nodes. No manual SLURM setup required.

### Tool parser `tool-call-parser`

Currently supported values must be one of the following:

apertus,cohere_command3,cohere_command4,deepseek_v3,deepseek_v31,deepseek_v32,deepseek_v4,ernie45,functiongemma,gemma4,gigachat3,glm45,glm47,granite,granite-20b-fc,granite4,hermes,hunyuan_a13b,hy_v3,internlm,jamba,kimi_k2,lfm2,llama3_json,llama4_json,llama4_pythonic,longcat,mimo,minimax,minimax_m2,mistral,olmo3,openai,phi4_mini_json,poolside_v1,pythonic,qwen3_coder,qwen3_xml,seed_oss,step3,step3p5,xlam

### Default Environment variables

The scripts that start up vllm will define a set of defaults. These can be overridden by `env:` entries but these are tested on isambard for stability.

```
# Prevent torch from over-subscribing CPU cores across parallel workers.
# GH200 has 72 cores; 16 threads/worker is safe for the 4-GPU-per-node case.
export OMP_NUM_THREADS=16

# Force NCCL to map over the Libfabric Cassini driver (Slingshot 11)
export NCCL_NET_GDR_LEVEL=5          # Enforce full GPUDirect RDMA across nodes
export FI_PROVIDER="cxi"             # Enforce Cray Cassini fabric provider
export FI_CXI_DEFAULT_CQ_SIZE=131072 # Expand Completion Queue size to prevent dropped frames

# Prevent Slingshot Memory Hooks Deadlocks
# HPE Slingshot uses 'memhooks' by default, which clashes with vLLM's memory allocation and hangs.
# Switching to userfaultfd guarantees stable collective communications.
export FI_MR_CACHE_MONITOR=userfaultfd

# Handle multi-NIC striping
# Each Isambard node has 4 separate Cassini NICs (one per GH200) operating at 200Gbps.
# These ensure NCCL spreads parallel communication across all 4 rails.
export NCCL_CROSS_NIC=1
export NCCL_MIN_NCHANNELS=4

# prevents parallel GPU worker processes from overlapping data transfers and causing a race condition or kernel hang during deep pipeline/tensor synchronizations
export CUDA_DEVICE_MAX_CONNECTIONS=1

# Prevents catastrophic virtual memory fragmentation inside the unified space
export NCCL_CUMEM_ENABLE=0

# Relaxed ordering tells the PCIe root complex that memory pages migrating
# between the LPDDR5 CPU memory and the HBM3 GPU memory don't need strict serial locks.
export NCCL_IB_PCI_RELAXED_ORDERING=1

# VLLM networking and compilation overrides:
export VLLM_SKIP_CUSTOM_ALL_REDUCE=1        # Force standard NCCL for stability across Slingshot 11
export VLLM_ENGINE_ITERATION_TIMEOUT_S=300  # Prevent timeouts during multi-node graph setups
export VLLM_ALLREDUCE_USE_SYMM_MEM=0        # Disable broken experimental symmetric memory allocator

# Prevent torch compile from starting up all 256 cores of a node at once
export MAX_JOBS=${maxJobs}                                  # usually 4
export TORCHINDUCTOR_PARALLEL_COMPILE_THREADS=${maxTorch}   # usually 4-16
export FLASHINFER_NVCC_THREADS=${maxFlash}                  # usually 32
export VLLM_ENGINE_ITERATION_TIMEOUT_S=300
```

## Validation

Before writing the file, verify every item:

### Parallelism

- [ ] `tensor-parallel-size × pipeline-parallel-size` = total GPUs needed
- [ ] `tensor-parallel-size` is 1, 2, or 4
- [ ] `tensor-parallel-size` evenly divides `num_attention_heads` (vLLM will crash on weight loading otherwise)
- [ ] Parallelism follows the default rules: **tp=2** for 1–2 GPUs, **tp=4** for 3–4 GPUs
- [ ] If multi-node: user has been warned about resource requirements (multi-node uses `#SBATCH --nodes=N` with `vllm serve --distributed-executor-backend ray`)

### Memory & context length

- [ ] `max-model-len` is a positive integer ≤ native context length
- [ ] If native context is very large (e.g. 131072), confirm the user wants the full context or if a smaller value is acceptable
- [ ] If model exceeds single-GPU memory and user wants single-node: `cpu-offload-gb` has been suggested
- [ ] If model is memory-constrained and an FP8 variant exists: `quantization: fp8` has been considered

### Special options

- [ ] **MoE model + tp ≥ 2**: flashinfer settings present (`gdn-prefill-backend`, `moe-backend`, `attention-config`, `enable-flashinfer-autotune: false`)
- [ ] **Reasoning model**: correct `reasoning-parser` name from the recipe
- [ ] **Tool calling**: `enable-auto-tool-choice: true` + matching `tool-call-parser` (or explicitly omitted for base models)
- [ ] **Prefix caching**: `enable-prefix-caching: true` recommended for agent/chatbot use cases

### Config integrity

- [ ] `model` field exactly matches the HuggingFace model ID (case-sensitive)
- [ ] `min-vllm-version` is set to the earliest vLLM version that supports this model + features
- [ ] `env:` only contains variables explicitly recommended by the vLLM recipe or required for the model
- [ ] All parameter options use the long name (two hyphens).
- [ ] `-cc`/`--compilation-config` shorthand expanded to long form and merged if already present

## Troubleshooting

| Problem | Solution |
|---------|---------|
| Model card not accessible | Ask user for parameter count and architecture manually |
| Parameter count ambiguous | Use total parameters (not non-embedding, not active) for memory calculation |
| Model fits on paper but OOM in practice | Reduce `max-model-len` by half; or suggest `quantization: fp8` |
| Context reduced from native | Always note native context in a comment; reduce to next power-of-two that fits, not to an arbitrary 32K |
| Reasoning parser name unknown | Check the model card vLLM quickstart snippet — it usually names the parser explicitly; or check the recipes page |
| MoE model shows poor throughput | Add `enable-expert-parallel: true` with `tensor-parallel-size >= 2` |
| Memory borderline | Try `quantization: fp8` — halves weight memory with minimal accuracy loss on Hopper (GH200/H100) |
| `ivllm start` fails with "version too low" | The installed vLLM (`ivllm config --vllm-version`) is below `min-vllm-version`; run `ivllm setup` with a newer version or update the config |

## References

- [Isambard AI hardware specs](references/isambard-specs.md)
- [vLLM config options and memory guide](references/vllm-config-guide.md)
- [vLLM serve cli options](references/vllm-serve-cli-0.23.0.md)
- [vLLM config environment variables](references/vllm-env-vars-0.23.0.md)
- [vLLM model-specific recipes](https://docs.vllm.ai/projects/recipes/en/latest/)
- [Isambard AI specs online](https://docs.isambard.ac.uk/specs/#system-specifications-isambard-ai-phase-2)

