## vllm serve

## JSON CLI Arguments

When passing JSON CLI arguments, the following sets of arguments are equivalent:

- `--json-arg '{"key1": "value1", "key2": {"key3": "value2"}}'`
- `--json-arg.key1 value1 --json-arg.key2.key3 value2`

Additionally, list elements can be passed individually using `+`:

- `--json-arg '{"key4": ["value3", "value4", "value5"]}'`
- `--json-arg.key4+ value3 --json-arg.key4+='value4,value5'`

## Arguments

#### \--headless

Run in headless mode. See multi-node data parallel documentation for more details.

Default: `False`

#### \--api-server-count, -asc

How many API server processes to run. Defaults to data\_parallel\_size if not specified.

#### \--config

Read CLI options from a config file. Must be a YAML with the following options: [configuration/serve\_args](https://docs.vllm.ai/en/stable/configuration/serve_args/)

#### \--grpc

Launch a gRPC server instead of the HTTP OpenAI-compatible server. Requires: pip install vllm\[grpc\].

Default: `False`

#### \--disable-log-stats

Disable logging statistics.

Default: `False`

#### \--aggregate-engine-logging

Log aggregate rather than per-engine statistics when using data parallelism.

Default: `False`

#### \--fail-on-environ-validation, --no-fail-on-environ-validation

If set, the engine will raise an error if environment validation fails.

Default: `False`

#### \--shutdown-timeout

Shutdown timeout in seconds. 0 = abort, >0 = wait.

Default: `0`

#### \--gdn-prefill-backend

Possible choices: `flashinfer`, `triton`, `cutedsl`

Select GDN prefill backend.

#### \--kda-prefill-backend

Possible choices: `auto`, `triton`, `flashkda`

Select KDA prefill backend.

#### \--enable-log-requests, --no-enable-log-requests

Enable logging request information, dependent on log level: - INFO: Request ID, parameters and LoRA request. - DEBUG: Prompt inputs (e.g: text, token IDs). You can set the minimum log level via `VLLM_LOGGING_LEVEL`.

Default: `False`

### Frontend

Arguments for the OpenAI-compatible frontend server.

#### \--lora-modules

#### \--chat-template

#### \--chat-template-content-format

Possible choices: `auto`, `openai`, `string`

Default: `auto`

#### \--trust-request-chat-template, --no-trust-request-chat-template

Default: `False`

#### \--default-chat-template-kwargs

: Should either be a valid JSON string or JSON keys passed individually.

#### \--response-role

Default: `assistant`

#### \--return-tokens-as-token-ids, --no-return-tokens-as-token-ids

Default: `False`

#### \--enable-auto-tool-choice, --no-enable-auto-tool-choice

Default: `False`

#### \--exclude-tools-when-tool-choice-none, --no-exclude-tools-when-tool-choice-none

Default: `False`

#### \--tool-call-parser

#### \--tool-parser-plugin

Default: `""`

#### \--tool-server

#### \--log-config-file

#### \--max-log-len

#### \--enable-prompt-tokens-details, --no-enable-prompt-tokens-details

Default: `False`

#### \--enable-per-request-metrics, --no-enable-per-request-metrics

Default: `False`

#### \--enable-server-load-tracking, --no-enable-server-load-tracking

Default: `False`

#### \--enable-force-include-usage, --no-enable-force-include-usage

Default: `False`

#### \--enable-tokenizer-info-endpoint, --no-enable-tokenizer-info-endpoint

Default: `False`

#### \--enable-log-outputs, --no-enable-log-outputs

Default: `False`

#### \--enable-log-deltas, --no-enable-log-deltas

Default: `True`

#### \--cohere-is-reasoning-model, --no-cohere-is-reasoning-model

Default: `True`

#### \--cohere-format

Default: `cmd4`

#### \--log-error-stack, --no-log-error-stack

Default: `False`

#### \--tokens-only, --no-tokens-only

Default: `False`

#### \--fingerprint-mode

Possible choices: `custom`, `full`, `hash`, `none`

Default: `full`

#### \--fingerprint-value

#### \--host

Host name.

#### \--port

Port number.

Default: `8000`

#### \--data-parallel-supervisor-port

HTTP port for aggregated health endpoints in multi-port external LB mode.

Default: `9256`

#### \--dp-supervisor-probe-interval-s

Seconds between aggregated health probes in multi-port external LB mode.

Default: `5.0`

#### \--dp-supervisor-probe-timeout-s

Seconds to wait between retries when a child health probe fails with a connection error in multi-port external LB mode.

Default: `5.0`

#### \--dp-supervisor-probe-failure-threshold

Number of consecutive connection-error retries before a child health probe is declared failed in multi-port external LB mode.

Default: `3`

#### \--uds

Unix domain socket path. If set, host and port arguments are ignored.

#### \--uvicorn-log-level

Possible choices: `critical`, `debug`, `error`, `info`, `trace`, `warning`

Log level for uvicorn.

Default: `info`

#### \--disable-uvicorn-access-log, --no-disable-uvicorn-access-log

Disable uvicorn access log.

Default: `False`

#### \--disable-access-log-for-endpoints

Comma-separated list of endpoint paths to exclude from uvicorn access logs. This is useful to reduce log noise from high-frequency endpoints like health checks. Example: "/health,/metrics,/ping". When set, access logs for requests to these paths will be suppressed while keeping logs for other endpoints.

#### \--allow-credentials, --no-allow-credentials

Allow credentials.

Default: `False`

#### \--allowed-origins

Allowed origins.

Default: `['*']`

#### \--allowed-methods

Allowed methods.

Default: `['*']`

#### \--allowed-headers

Allowed headers.

Default: `['*']`

#### \--api-key

If provided, the server will require one of these keys to be presented in the header.

#### \--ssl-keyfile

The file path to the SSL key file.

#### \--ssl-certfile

The file path to the SSL cert file.

#### \--ssl-ca-certs

The CA certificates file.

#### \--enable-ssl-refresh, --no-enable-ssl-refresh

Refresh SSL Context when SSL certificate files change

Default: `False`

#### \--ssl-cert-reqs

Whether client certificate is required (see stdlib ssl module's).

Default: `0`

#### \--ssl-ciphers

SSL cipher suites for HTTPS (TLS 1.2 and below only). Example: 'ECDHE-RSA-AES256-GCM-SHA384:ECDHE-RSA-CHACHA20-POLY1305'

#### \--root-path

FastAPI root\_path when app is behind a path based routing proxy.

#### \--middleware

Additional ASGI middleware to apply to the app. We accept multiple --middleware arguments. The value should be an import path. If a function is provided, vLLM will add it to the server using `@app.middleware('http')`. If a class is provided, vLLM will add it to the server using `app.add_middleware()`.

Default: `[]`

#### \--enable-request-id-headers, --no-enable-request-id-headers

If specified, API server will add X-Request-Id header to responses.

Default: `False`

#### \--disable-fastapi-docs, --no-disable-fastapi-docs

Disable FastAPI's OpenAPI schema, Swagger UI, and ReDoc endpoint.

Default: `False`

#### \--h11-max-incomplete-event-size

Maximum size (bytes) of an incomplete HTTP event (header or body) for h11 parser. Helps mitigate header abuse. Default: 4194304 (4 MB).

Default: `4194304`

#### \--h11-max-header-count

Maximum number of HTTP headers allowed in a request for h11 parser. Helps mitigate header abuse. Default: 256.

Default: `256`

#### \--enable-offline-docs, --no-enable-offline-docs

Enable offline FastAPI documentation for air-gapped environments. Uses vendored static assets bundled with vLLM.

Default: `False`

#### \--enable-flash-late-interaction, --no-enable-flash-late-interaction

If set, run pooling score MaxSim on GPU in the API server process. Can significantly improve late-interaction scoring performance.

Default: `True`

### ModelConfig

Configuration for the model.

#### \--model

Name or path of the Hugging Face model to use. It is also used as the content for `model_name` tag in metrics output when `served_model_name` is not specified.

Default: `Qwen/Qwen3-0.6B`

#### \--runner

Possible choices: `auto`, `draft`, `generate`, `pooling`

The type of model runner to use. Each vLLM instance only supports one model runner, even if the same model can be used for multiple types.

Default: `auto`

#### \--convert

Possible choices: `auto`, `classify`, `embed`, `none`

Convert the model using adapters defined in [vllm.model\_executor.models.adapters](https://docs.vllm.ai/en/stable/api/vllm/model_executor/models/adapters/#vllm.model_executor.models.adapters "            vllm.model_executor.models.adapters"). The most common use case is to adapt a text generation model to be used for pooling tasks.

Default: `auto`

#### \--tokenizer

Name or path of the Hugging Face tokenizer to use. If unspecified, model name or path will be used.

#### \--tokenizer-mode

Possible choices: `auto`, `cohere`, `deepseek_v32`, `deepseek_v4`, `hf`, `inkling`, `kimi_k3`, `mistral`, `slow`

Tokenizer mode:

- "auto" will use the tokenizer from `mistral_common` for Mistral models if available, otherwise it will use the "hf" tokenizer.
- "hf" will use the fast tokenizer if available.
- "slow" will always use the slow tokenizer.
- "mistral" will always use the tokenizer from `mistral_common`.
- "deepseek\_v32" will always use the tokenizer from `deepseek_v32`.
- "deepseek\_v4" will always use the tokenizer from `deepseek_v4`.
- "kimi\_k3" will always use the "hf" tokenizer but render chat prompts with Kimi K3's Python XTML encoding instead of a Jinja template.
- "cohere" uses the standard HF tokenizer but renders the chat template via the `cohere_melody` library (cmd3 / cmd4 templates) instead of Jinja, and surfaces grounded-citation metadata on responses.
- Other custom values can be supported via plugins.

To swap the Rust BPE backend that powers HF fast tokenizers for the [fastokens](https://github.com/crusoecloud/fastokens) implementation, set `VLLM_USE_FASTOKENS=1` instead — that override applies to any mode that loads an HF fast tokenizer (`hf`, `deepseek_v32`, `deepseek_v4`, …).

Default: `auto`

#### \--trust-remote-code, --no-trust-remote-code

Trust remote code (e.g., from HuggingFace) when downloading the model and tokenizer.

Default: `False`

#### \--dtype

Possible choices: `auto`, `bfloat16`, `float`, `float16`, `float32`, `half`

Data type for model weights and activations:

- "auto" will use FP16 precision for FP32 and FP16 models, and BF16 precision for BF16 models.
- "half" for FP16. Recommended for AWQ quantization.
- "float16" is the same as "half".
- "bfloat16" for a balance between precision and range.
- "float" is shorthand for FP32 precision.
- "float32" for FP32 precision.

Default: `auto`

#### \--seed

Random seed for reproducibility.

We must set the global seed because otherwise, different tensor parallel workers would sample different tokens, leading to inconsistent results.

Default: `0`

#### \--hf-config-path

Name or path of the Hugging Face config to use. If unspecified, model name or path will be used.

#### \--allowed-local-media-path

Allowing API requests to read local images or videos from directories specified by the server file system. This is a security risk. Should only be enabled in trusted environments.

Default: `""`

#### \--allowed-media-domains

If set, only media URLs that belong to this domain can be used for multi-modal inputs.

#### \--revision

The specific model version to use. It can be a branch name, a tag name, or a commit id. If unspecified, will use the default version.

#### \--code-revision

The specific revision to use for the model code on the Hugging Face Hub. It can be a branch name, a tag name, or a commit id. If unspecified, will use the default version.

#### \--tokenizer-revision

The specific revision to use for the tokenizer on the Hugging Face Hub. It can be a branch name, a tag name, or a commit id. If unspecified, will use the default version.

#### \--max-model-len

Model context length (prompt and output). If unspecified, will be automatically derived from the model config.

When passing via `--max-model-len`, supports k/m/g/K/M/G in human-readable format. Examples:

- 1k -> 1000
- 1K -> 1024
- 25.6k -> 25,600
- \-1 or 'auto' -> Automatically choose the maximum model length that fits in GPU memory. This will use the model's maximum context length if it fits, otherwise it will find the largest length that can be accommodated.

Parse human-readable integers like '1k', '2M', etc. Including decimal values with decimal multipliers. Also accepts -1 or 'auto' as a special value for auto-detection.

```js
Examples:
- '1k' -> 1,000
- '1K' -> 1,024
- '25.6k' -> 25,600
- '-1' or 'auto' -> -1 (special value for auto-detection)
```

#### \--quantization, -q

Method used to quantize the weights. If `None`, we first check the `quantization_config` attribute in the model config file. If that is `None`, we assume the model weights are not quantized and use `dtype` to determine the data type of the weights.

#### \--quantization-config

User-facing quantization configuration. Carries per-layer-kind specs (linear, moe) and ignore patterns; see:class:[`QuantizationConfigArgs`](https://docs.vllm.ai/en/stable/api/vllm/config/quantization/#vllm.config.quantization.QuantizationConfigArgs "            QuantizationConfigArgs"). Auto-populated from the matching online shorthand when `quantization` is one of the values in `ONLINE_QUANT_SHORTHAND_NAMES`.

Should either be a valid JSON string or JSON keys passed individually.

#### \--allow-deprecated-quantization, --no-allow-deprecated-quantization

Whether to allow deprecated quantization methods.

Default: `False`

#### \--enforce-eager, --no-enforce-eager

Whether to always use eager-mode PyTorch. If True, we will disable CUDA graph and always execute the model in eager mode. If False, we will use CUDA graph and eager execution in hybrid for maximal performance and flexibility.

Default: `False`

#### \--enable-return-routed-experts, --no-enable-return-routed-experts

Whether to return routed experts.

Default: `False`

#### \--max-logprobs

Maximum number of log probabilities to return when `logprobs` is specified in [`SamplingParams`](https://docs.vllm.ai/en/stable/api/vllm/sampling_params/#vllm.sampling_params.SamplingParams "            SamplingParams"). The default value comes the default for the OpenAI Chat Completions API. -1 means no cap, i.e. all (output\_length \* vocab\_size) logprobs are allowed to be returned and it may cause OOM.

Default: `20`

#### \--logprobs-mode

Possible choices: `processed_logits`, `processed_logprobs`, `raw_logits`, `raw_logprobs`

Indicates the content returned in the logprobs and prompt\_logprobs. Supported mode: 1) raw\_logprobs, 2) processed\_logprobs, 3) raw\_logits, 4) processed\_logits. Raw means the values before applying any logit processors, like bad words. Processed means the values after applying all processors, including temperature and top\_k/top\_p. Note: for prompt\_logprobs, processed\_ *and raw\_* yield identical results because prompt tokens do not go through sampling processors.

Default: `raw_logprobs`

#### \--use-fp64-gumbel, --no-use-fp64-gumbel

Whether to use FP64 (instead of FP32) random noise for Gumbel-max and equivalent exponential-race sampling. FP64 preserves lower-tail sampling events that fp32 uniform/exponential draws can truncate, at the cost of significantly lower throughput on most GPUs.

Default: `False`

#### \--disable-sliding-window, --no-disable-sliding-window

Whether to disable sliding window. If True, we will disable the sliding window functionality of the model, capping to sliding window size. If the model does not support sliding window, this argument is ignored.

Default: `False`

#### \--disable-cascade-attn, --no-disable-cascade-attn

Disable cascade attention for V1. While cascade attention does not change the mathematical correctness, disabling it could be useful for preventing potential numerical issues. This defaults to True, so users must opt in to cascade attention by setting this to False. Even when this is set to False, cascade attention will only be used when the heuristic tells that it's beneficial.

Default: `True`

#### \--skip-tokenizer-init, --no-skip-tokenizer-init

Skip initialization of tokenizer and detokenizer. Expects valid `prompt_token_ids` and `None` for prompt from the input. The generated output will contain token ids.

Default: `False`

#### \--enable-prompt-embeds, --no-enable-prompt-embeds

If `True`, enables passing text embeddings as inputs via the `prompt_embeds` key.

WARNING: The vLLM engine may crash if incorrect shape of embeddings is passed. Only enable this flag for trusted users!

Default: `False`

#### \--served-model-name

The model name(s) used in the API. If multiple names are provided, the server will respond to any of the provided names. The model name in the model field of a response will be the first name in this list. If not specified, the model name will be the same as the `--model` argument. Noted that this name(s) will also be used in `model_name` tag content of prometheus metrics, if multiple names provided, metrics tag will take the first one.

#### \--config-format

Possible choices: `auto`, `hf`, `mistral`

The format of the model config to load:

- "auto" will try to load the config in hf format if available after trying to load in mistral format.
- "hf" will load the config in hf format.
- "mistral" will load the config in mistral format.

Default: `auto`

#### \--hf-token

The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated when running `hf auth login` (stored in `~/.cache/huggingface/token`).

#### \--hf-overrides

If a dictionary, contains arguments to be forwarded to the Hugging Face config. If a callable, it is called to update the HuggingFace config.

Default: `{}`

#### \--model-class-overrides

Override the model class used for one or more architectures, mapping the architecture name to a `"module:class"` target (the same format accepted by `ModelRegistry.register_model`). This registers the target class at runtime, e.g. `{"GlmMoeDsaForCausalLM": "vllm.models.deepseek_v32.nvidia.model:DeepseekV32ForCausalLM"}`. This argument is for development and debugging purposes only.

Should either be a valid JSON string or JSON keys passed individually.

Default: `{}`

#### \--pooler-config

Pooler config which controls the behaviour of output pooling in pooling models.

API docs: [`vllm.config.PoolerConfig`](https://docs.vllm.ai/en/stable/api/vllm/config/#vllm.config.PoolerConfig "            PoolerConfig")

Should either be a valid JSON string or JSON keys passed individually.

#### \--generation-config

The folder path to the generation config. Defaults to `"auto"`, the generation config will be loaded from model path. If set to `"vllm"`, no generation config is loaded, vLLM defaults will be used. If set to a folder path, the generation config will be loaded from the specified folder path. If `max_new_tokens` is specified in generation config, then it sets a server-wide limit on the number of output tokens for all requests.

Default: `auto`

#### \--override-generation-config

Overrides or sets generation config. e.g. `{"temperature": 0.5}`. If used with `--generation-config auto`, the override parameters will be merged with the default config from the model. If used with `--generation-config vllm`, only the override parameters are used.

Should either be a valid JSON string or JSON keys passed individually.

Default: `{}`

#### \--enable-sleep-mode, --no-enable-sleep-mode

Enable sleep mode for the engine (only cuda and hip platforms are supported).

Default: `False`

#### \--enable-cumem-allocator, --no-enable-cumem-allocator

Enable the custom cumem allocator to leverage advanced GPU memory allocation features such as multi-node NVLink support.

Sleep mode automatically enables this allocator. Only cuda and hip platforms are supported.

Default: `False`

#### \--model-impl

Possible choices: `auto`, `terratorch`, `transformers`, `vllm`

Which implementation of the model to use:

- "auto" will try to use the vLLM implementation, if it exists, and fall back to the Transformers implementation if no vLLM implementation is available.
- "vllm" will use the vLLM model implementation.
- "transformers" will use the Transformers model implementation.
- "terratorch" will use the TerraTorch model implementation.

Default: `auto`

#### \--override-attention-dtype

Override dtype for attention

#### \--logits-processors

One or more logits processors' fully-qualified class names or class definitions

#### \--io-processor-plugin

IOProcessor plugin name to load at model startup

#### \--renderer-num-workers

Number of worker threads in the renderer thread pool. The pool is consumed by the async renderer path (e.g. the OpenAI-compatible API server started by `vllm serve`) to parallelize tokenization, chat template rendering, and multimodal preprocessing across concurrent requests.

The offline [`LLM`](https://docs.vllm.ai/en/stable/api/vllm/entrypoints/llm/#vllm.entrypoints.llm.LLM "            LLM") entrypoint uses the synchronous renderer path and processes prompts (including multimodal preprocessing) serially, so this setting has no effect there.

Default: `1`

### LoadConfig

Configuration for loading the model weights.

#### \--load-format

The format of the model weights to load.

- "auto" will try to load the weights in the safetensors format and fall back to the pytorch bin format if safetensors format is not available.
- "pt" will load the weights in the pytorch bin format.
- "safetensors" will load the weights in the safetensors format.
- "instanttensor" will load the Safetensors weights on CUDA devices using InstantTensor, which enables distributed loading with pipelined prefetching and fast direct I/O.
- "npcache" will load the weights in pytorch format and store a numpy cache to speed up the loading.
- "dummy" will initialize the weights with random values, which is mainly for profiling.
- "tensorizer" will use CoreWeave's tensorizer library for fast weight loading. See the Tensorize vLLM Model script in the Examples section for more information.
- "runai\_streamer" will load the Safetensors weights using Run:ai Model Streamer.
- "runai\_streamer\_sharded" will load weights from pre-sharded checkpoint files using Run:ai Model Streamer.
- "bitsandbytes" will load the weights using bitsandbytes quantization.
- "sharded\_state" will load weights from pre-sharded checkpoint files, supporting efficient loading of tensor-parallel models.
- "mistral" will load weights from consolidated safetensors files used by Mistral models.
- "modelexpress" will load weights using ModelExpress.
- Other custom values can be supported via plugins.

Default: `auto`

#### \--download-dir

Directory to download and load the weights, default to the default cache directory of Hugging Face.

#### \--safetensors-load-strategy

Possible choices: `eager`, [`lazy`](https://docs.vllm.ai/en/stable/api/vllm/logging_utils/lazy/#vllm.logging_utils.lazy.lazy "            lazy"), `prefetch`, `torchao`, `None`

Specifies the loading strategy for safetensors weights.

- None (default): Uses memory-mapped (lazy) loading. When an NFS filesystem is detected and the total checkpoint size fits within 90%%%% of available RAM, prefetching is enabled automatically.
- "lazy": Weights are memory-mapped from the file. This enables on-demand loading and is highly efficient for models on local storage. Unlike the default (None), auto-prefetch on NFS is not performed.
- "eager": The entire file is read into CPU memory upfront before loading. This is recommended for models on network filesystems (e.g., Lustre, NFS) as it avoids inefficient random reads, significantly speeding up model initialization. However, it uses more CPU RAM.
- "prefetch": Checkpoint files are read into the OS page cache before workers load them, speeding up the model loading phase. Useful on network or high-latency storage.
- "torchao": Weights are loaded in upfront and then reconstructed into torchao tensor subclasses. This is used when the checkpoint was quantized using torchao and saved using safetensors. Needs `torchao >= 0.14.0`.

#### \--safetensors-prefetch-num-threads

Number of worker threads used to prefetch safetensors checkpoint files into the OS page cache when safetensors prefetching is enabled.

Default: `8`

#### \--safetensors-prefetch-block-size

Read size in bytes for each safetensors checkpoint file prefetch.

Parse human-readable integers like '1k', '2M', etc. Including decimal values with decimal multipliers.

```js
Examples:
- '1k' -> 1,000
- '1K' -> 1,024
- '25.6k' -> 25,600
```

Default: `16777216`

#### \--model-loader-extra-config

Extra config for model loader. This will be passed to the model loader corresponding to the chosen load\_format.

Default: `{}`

#### \--ignore-patterns

The list of patterns to ignore when loading the model. Default to "original/\* */* " to avoid repeated loading of llama's checkpoints.

Default: `['original/**/*']`

#### \--use-tqdm-on-load, --no-use-tqdm-on-load

Whether to enable tqdm for showing progress bar when loading model weights.

Default: `True`

#### \--pt-load-map-location

The map location for loading pytorch checkpoint, to support loading checkpoints can only be loaded on certain devices like "cuda", this is equivalent to `{"": "cuda"}`. Another supported format is mapping from different devices like from GPU 1 to GPU 0: `{"cuda:1": "cuda:0"}`. Note that when passed from command line, the strings in dictionary need to be double quoted for json parsing. For more details, see the original doc for `map_location` parameter in [`torch.load`](https://pytorch.org/docs/stable/generated/torch.load.html#torch.load) parameter.

Default: `cpu`

### AttentionConfig

Configuration for attention mechanisms in vLLM.

#### \--attention-backend

Attention backend to use. Use "auto" or None for automatic selection.

### MambaConfig

Configuration for Mamba SSM backends.

#### \--mamba-backend

Mamba SSU backend to use.

Default: `MambaBackendEnum.TRITON`

#### \--mamba-ssu-algorithm

Possible choices: `auto`, `horizontal`, `simple`, `vertical`, `None`

Selective state update algorithm to use with the FlashInfer backend. None defaults to FlashInfer's "auto" algorithm. Forced algorithms must be supported by FlashInfer for the active GPU, state dtype, and decoding mode.

#### \--enable-mamba-cache-stochastic-rounding, --no-enable-mamba-cache-stochastic-rounding

Enable stochastic rounding when writing SSM state to fp16 cache. Uses random bits to unbias the rounding error, which can improve numerical stability for long sequences.

Default: `False`

#### \--mamba-cache-philox-rounds

Number of Philox PRNG rounds for stochastic rounding random number generation. 0 uses the Triton default. Higher values improve randomness quality at the cost of compute.

Default: `0`

### StructuredOutputsConfig

Dataclass which contains structured outputs config for the engine.

#### \--reasoning-parser

Select the reasoning parser depending on the model that you're using. This is used to parse the reasoning content into OpenAI API format.

Default: `""`

#### \--reasoning-parser-plugin

Path to a dynamically reasoning parser plugin that can be dynamically loaded and registered.

Default: `""`

### ParallelConfig

Configuration for the distributed execution.

#### \--distributed-executor-backend

Possible choices: `external_launcher`, `mp`, `ray`, `uni`

Backend to use for distributed model workers, either "ray" or "mp" (multiprocessing). If the product of pipeline\_parallel\_size and tensor\_parallel\_size is less than or equal to the number of GPUs available, "mp" will be used to keep processing on a single host. Otherwise, an error will be raised. To use "mp" you must also set nnodes, and to use "ray" you must manually set distributed\_executor\_backend to "ray".

Note: [TPU](https://docs.vllm.ai/projects/tpu/en/latest/) platform only supports Ray for distributed inference.

#### \--pipeline-parallel-size, -pp

Number of pipeline parallel groups.

Default: `1`

#### \--master-addr

distributed master address for multi-node distributed inference when distributed\_executor\_backend is mp.

Default: `127.0.0.1`

#### \--master-port

distributed master port for multi-node distributed inference when distributed\_executor\_backend is mp.

Default: `29501`

#### \--nnodes, -n

num of nodes for multi-node distributed inference when distributed\_executor\_backend is mp.

Default: `1`

#### \--node-rank, -r

distributed node rank for multi-node distributed inference when distributed\_executor\_backend is mp.

Default: `0`

#### \--distributed-timeout-seconds

Timeout in seconds for distributed operations (e.g., init\_process\_group). If set, this value is passed to torch.distributed.init\_process\_group as the timeout parameter. If None, PyTorch's default timeout is used (600s for NCCL). Increase this for multi-node setups where model downloads may be slow.

#### \--cpu-distributed-timeout-seconds

Timeout (in seconds) for cpu communication groups. If None, PyTorch's default timeout is used (1800s for gloo).

#### \--numa-bind, --no-numa-bind

Enable NUMA binding for GPU worker subprocesses.

By default, workers are pinned to their GPU's NUMA-local CPUs and memory; on PCT-capable Xeons they also auto-bind to the SKU's PCT priority cores.

Default: `False`

#### \--numa-bind-nodes

NUMA node to bind each GPU worker to.

Specify one NUMA node per visible GPU, for example `[0, 0, 1, 1]` for a 4-GPU system with GPUs 0-1 on NUMA node 0 and GPUs 2-3 on NUMA node 1. If unset and `numa_bind=True`, vLLM auto-detects the GPU-to-NUMA topology. The values are passed to `numactl --membind` and `--cpunodebind`, so they must be valid `numactl` NUMA node indices.

#### \--numa-bind-cpus

Optional CPU lists to bind each GPU worker to.

Specify one CPU list per visible GPU, for example `["0-3", "4-7", "8-11", "12-15"]`. When set, vLLM uses `numactl --physcpubind` instead of `--cpunodebind`. This is useful for custom policies such as binding to PCT or other high-frequency cores. Each entry must use `numactl --physcpubind` CPU-list syntax, for example `"0-3"` or `"0,2,4-7"`.

#### \--device-ids

Comma-separated physical GPU device IDs or UUIDs to use (e.g. --device-ids "2,3,5,7"). Avoids setting CUDA\_VISIBLE\_DEVICES, preserving full GPU topology visibility for GPU-NIC affinity and DeepGEMM. Note: has no effect with Ray executors; use Ray placement groups for GPU selection instead.

#### \--tensor-parallel-size, -tp

Number of tensor parallel groups.

Default: `1`

#### \--decode-context-parallel-size, -dcp

Number of ranks that shard the decode KV cache. DCP does not expand the process world size. Without PCP, DCP reuses TP ranks. With PCP, DCP either spans the PCP axis or the full TP x PCP block.

Default: `1`

#### \--dcp-comm-backend

Possible choices: `a2a`, `ag_rs`

Communication backend for Decode Context Parallel (DCP). - "ag\_rs": AllGather + ReduceScatter (default, existing behavior) - "a2a": All-to-All exchange of partial outputs + LSE, then combine with Triton kernel. Reduces NCCL calls from 3 to 2 per layer for MLA models.

Default: `ag_rs`

#### \--dcp-kv-cache-interleave-size

Interleave size of kv\_cache storage while using DCP. dcp\_kv\_cache\_interleave\_size has been replaced by cp\_kv\_cache\_interleave\_size, and will be deprecated when PCP is fully supported.

Default: `1`

#### \--cp-kv-cache-interleave-size

Interleave size of kv\_cache storage while using DCP. Store interleave\_size tokens on dcp\_rank i, then store next interleave\_size tokens on dcp\_rank i+1. Interleave\_size=1: token-level alignment, where token `i` is stored on dcp\_rank `i %% dcp_world_size`. Interleave\_size=block\_size: block-level alignment, where tokens are first populated to the preceding ranks. Tokens are then stored in (rank i+1, block j) only after (rank i, block j) is fully occupied. Block\_size should be greater than or equal to cp\_kv\_cache\_interleave\_size. Block\_size should be divisible by cp\_kv\_cache\_interleave\_size.

Default: `1`

#### \--prefill-context-parallel-size, -pcp

Number of ranks that split prefill sequence computation. PCP expands the process world size but does not increase the KV-cache shard count.

Default: `1`

#### \--data-parallel-size, -dp

Number of data parallel groups. MoE layers will be sharded according to the product of the tensor, prefill-context, and data parallel sizes.

Default: `1`

#### \--data-parallel-rank, -dpn

Data parallel rank of this instance. When set, enables external load balancer mode for MoE data-parallel deployments. Unsupported for non-MoE models; launch independent vLLM instances instead.

#### \--data-parallel-start-rank, -dpr

Starting data parallel rank for secondary nodes.

#### \--data-parallel-size-local, -dpl

Number of data parallel replicas to run on this node.

#### \--data-parallel-address, -dpa

Address of data parallel cluster head-node.

#### \--data-parallel-rpc-port, -dpp

Port for data parallel RPC communication.

#### \--data-parallel-backend, -dpb

Backend for data parallel, either "mp" or "ray".

Default: `mp`

#### \--data-parallel-hybrid-lb, --no-data-parallel-hybrid-lb, -dph

Whether to use "hybrid" DP LB mode. Applies only to online serving and when data\_parallel\_size > 0. Enables running an AsyncLLM and API server on a "per-node" basis where vLLM load balances between local data parallel ranks, but an external LB balances between vLLM nodes/replicas. Set explicitly in conjunction with --data-parallel-start-rank.

Default: `False`

#### \--data-parallel-external-lb, --no-data-parallel-external-lb, -dpe

Whether to use "external" DP LB mode. Applies only to online serving and when data\_parallel\_size > 0. This is useful for a "one-pod-per-rank" wide-EP setup in Kubernetes. Supported only for MoE deployments; non-MoE models should use independent vLLM instances without --data-parallel-\* arguments. Set implicitly when --data-parallel-rank is provided explicitly to vllm serve.

Default: `False`

#### \--data-parallel-multi-port-external-lb, -dpm

Run a node-local supervisor that launches one external-LB API server per local data parallel rank and exposes aggregated health on a supervisor port.

Default: `False`

#### \--enable-expert-parallel, --no-enable-expert-parallel, -ep

Use expert parallelism instead of tensor parallelism for MoE layers.

Default: `False`

#### \--enable-ep-weight-filter, --no-enable-ep-weight-filter

Skip non-local expert weights during model loading when expert parallelism is active. Each rank only reads its own expert shard from disk, which can drastically reduce storage I/O for MoE models with per-expert weight tensors (e.g. DeepSeek, Mixtral, Kimi-K2.5). Has no effect on 3D fused-expert checkpoints (e.g. GPT-OSS) or non-MoE models.

Default: `False`

#### \--all2all-backend

Possible choices: `allgather_reducescatter`, `deepep_high_throughput`, `deepep_low_latency`, `deepep_v2`, `flashinfer_all2allv`, `flashinfer_nvlink_one_sided`, `flashinfer_nvlink_two_sided`, `mori_high_throughput`, `mori_low_latency`, `naive`, `nixl_ep`, `pplx`

All2All backend for MoE expert parallel communication. Available options:

- "allgather\_reducescatter": All2all based on allgather and reducescatter
- "deepep\_high\_throughput": Use deepep high-throughput kernels
- "deepep\_low\_latency": Use deepep low-latency kernels
- "mori\_high\_throughput": MoRI EP with InterNodeV1 for multi-node
- "mori\_low\_latency": MoRI EP with InterNodeV1LL for multi-node
- "nixl\_ep": Use nixl-ep kernels
- "flashinfer\_nvlink\_two\_sided": Use flashinfer two-sided kernels for mnnvl
- "flashinfer\_nvlink\_one\_sided": Use flashinfer high-throughput a2a kernels

Default: `allgather_reducescatter`

#### \--enable-dbo, --no-enable-dbo

Enable dual batch overlap for the model executor.

Default: `False`

#### \--ubatch-size

Number of ubatch size.

Default: `0`

#### \--enable-elastic-ep, --no-enable-elastic-ep

Enable elastic expert parallelism with stateless NCCL groups for DP/EP.

Default: `False`

#### \--dbo-decode-token-threshold

The threshold for dual batch overlap for batches only containing decodes. If the number of tokens in the request is greater than this threshold, microbatching will be used. Otherwise, the request will be processed in a single batch.

Default: `32`

#### \--dbo-prefill-token-threshold

The threshold for dual batch overlap for batches that contain one or more prefills. If the number of tokens in the request is greater than this threshold, microbatching will be used. Otherwise, the request will be processed in a single batch.

Default: `512`

#### \--disable-nccl-for-dp-synchronization, --no-disable-nccl-for-dp-synchronization

Forces the dp synchronization logic in vllm/v1/worker/dp\_utils.py to use Gloo instead of NCCL for its all reduce.

Defaults to True when async scheduling is enabled, False otherwise.

#### \--enable-eplb, --no-enable-eplb

Enable expert parallelism load balancing for MoE layers.

Default: `False`

#### \--eplb-config

Expert parallelism configuration.

API docs: [`vllm.config.EPLBConfig`](https://docs.vllm.ai/en/stable/api/vllm/config/#vllm.config.EPLBConfig "            EPLBConfig")

Should either be a valid JSON string or JSON keys passed individually.

Default: `EPLBConfig(window_size=1000, step_interval=3000, num_redundant_experts=0, log_balancedness=False, log_balancedness_interval=1, use_async=True, policy='default', communicator=None)`

#### \--expert-placement-strategy

Possible choices: `linear`, `round_robin`

The expert placement strategy for MoE layers:

- "linear": Experts are placed in a contiguous manner. For example, with 4 experts and 2 ranks, rank 0 will have experts \[0, 1\] and rank 1 will have experts \[2, 3\].
- "round\_robin": Experts are placed in a round-robin manner. For example, with 4 experts and 2 ranks, rank 0 will have experts \[0, 2\] and rank 1 will have experts \[1, 3\]. This strategy can help improve load balancing for grouped expert models with no redundant experts.

Default: `linear`

#### \--max-parallel-loading-workers

Maximum number of parallel loading workers when loading model sequentially in multiple batches. To avoid RAM OOM when using tensor parallel and large models.

#### \--ray-workers-use-nsight, --no-ray-workers-use-nsight

Whether to profile Ray workers with nsight, see https://docs.ray.io/en/latest/ray-observability/user-guides/profiling.html#profiling-nsight-profiler.

Default: `False`

#### \--disable-custom-all-reduce, --no-disable-custom-all-reduce

Disable the custom all-reduce kernel and fall back to NCCL.

Default: `False`

#### \--worker-cls

The full name of the worker class to use. If "auto", the worker class will be determined based on the platform.

Default: `auto`

#### \--worker-extension-cls

The full name of the worker extension class to use. The worker extension class is dynamically inherited by the worker class. This is used to inject new attributes and methods to the worker class for use in collective\_rpc calls.

Default: `""`

#### \--enable-fault-tolerance, --no-enable-fault-tolerance

Enable fault tolerance for detailed error recovery, such as scaling down fault DPEngineCore.

Default: `False`

#### \--fault-tolerance-config

The configurations for fault tolerance.

API docs: [`vllm.config.FaultToleranceConfig`](https://docs.vllm.ai/en/stable/api/vllm/config/#vllm.config.FaultToleranceConfig "            FaultToleranceConfig")

Should either be a valid JSON string or JSON keys passed individually.

Default: `FaultToleranceConfig(engine_recovery_timeout_sec=120)`

### CacheConfig

Configuration for the KV cache.

#### \--block-size

Size of a contiguous cache block in number of tokens. Accepts None (meaning "use default"). After construction, always int.

#### \--gpu-memory-utilization

The fraction of GPU memory to be used for the model executor, which can range from 0 to 1. For example, a value of 0.5 would imply 50%% GPU memory utilization. If unspecified, will use the default value of 0.92. This is a per-instance limit, and only applies to the current vLLM instance. It does not matter if you have another vLLM instance running on the same GPU. For example, if you have two vLLM instances running on the same GPU, you can set the GPU memory utilization to 0.5 for each instance.

Default: `0.92`

#### \--kv-cache-memory-bytes

Size of KV Cache per GPU in bytes. By default, this is set to None and vllm can automatically infer the kv cache size based on gpu\_memory\_utilization. However, users may want to manually specify the kv cache memory size. kv\_cache\_memory\_bytes allows more fine-grain control of how much memory gets used when compared with using gpu\_memory\_utilization. Note that kv\_cache\_memory\_bytes (when not-None) ignores gpu\_memory\_utilization

Parse human-readable integers like '1k', '2M', etc. Including decimal values with decimal multipliers.

```js
Examples:
- '1k' -> 1,000
- '1K' -> 1,024
- '25.6k' -> 25,600
```

#### \--kv-cache-dtype

Possible choices: `auto`, `bfloat16`, `float16`, `fp8`, `fp8_ds_mla`, `fp8_e4m3`, `fp8_e5m2`, `fp8_inc`, `fp8_per_token_head`, `int4_per_token_head`, `int8_per_token_head`, `nvfp4`, `turboquant_3bit_nc`, `turboquant_4bit_nc`, `turboquant_k3v4_nc`, `turboquant_k8v4`

Data type for kv cache storage. If "auto", will use model data type. CUDA 11.8+ supports fp8 (=fp8\_e4m3) and fp8\_e5m2. ROCm (AMD GPU) supports fp8 (=fp8\_e4m3). Intel Gaudi (HPU) supports fp8 (using fp8\_inc). Some models (namely DeepSeekV3.2) default to fp8, set to bfloat16 to use bfloat16 instead, this is an invalid option for models that do not default to fp8.

Default: `auto`

#### \--num-gpu-blocks-override

Number of GPU blocks to use. This overrides the profiled `num_gpu_blocks` if specified. Does nothing if `None`. Used for testing preemption.

#### \--enable-prefix-caching, --no-enable-prefix-caching

Whether to enable prefix caching.

#### \--prefix-caching-hash-algo

Possible choices: `sha256`, `sha256_cbor`, `xxhash`, `xxhash_cbor`

Set the hash algorithm for prefix caching:

- "sha256" uses Pickle for object serialization before hashing. This is the current default, as SHA256 is the most secure choice to avoid potential hash collisions.
- "sha256\_cbor" provides a reproducible, cross-language compatible hash. It serializes objects using canonical CBOR and hashes them with SHA-256.
- "xxhash" uses Pickle serialization with xxHash (128-bit) for faster, non-cryptographic hashing. Requires the optional `xxhash` package. IMPORTANT: Use of a hashing algorithm that is not considered cryptographically secure theoretically increases the risk of hash collisions, which can cause undefined behavior or even leak private information in multi-tenant environments. Even if collisions are still very unlikely, it is important to consider your security risk tolerance against the performance benefits before turning this on.
- "xxhash\_cbor" combines canonical CBOR serialization with xxHash for reproducible hashing. Requires the optional `xxhash` package.

Default: `sha256`

#### \--calculate-kv-scales, --no-calculate-kv-scales

Deprecated: This option is deprecated and will be removed in v0.19. It enables dynamic calculation of `k_scale` and `v_scale` when kv\_cache\_dtype is fp8. If `False`, the scales will be loaded from the model checkpoint if available. Otherwise, the scales will default to 1.0.

Default: `False`

#### \--kv-cache-dtype-skip-layers

Layer patterns to skip KV cache quantization. Accepts layer indices (e.g., '0', '2', '4') or attention type names (e.g., 'sliding\_window').

Default: `[]`

#### \--kv-sharing-fast-prefill, --no-kv-sharing-fast-prefill

In some KV sharing setups, e.g. YOCO (https://arxiv.org/abs/2405.05254), some layers can skip tokens corresponding to prefill. This flag enables attention metadata for eligible layers to be overridden with metadata necessary for implementing this optimization in some models (e.g. Gemma3n) NOTE: KV cache sharing is not supported for MRv2 (v2 model runner).

Default: `False`

#### \--mamba-cache-dtype

Possible choices: `auto`, `bfloat16`, `float16`, `float32`

The data type to use for the Mamba cache (both the conv as well as the ssm state). If set to 'auto', the data type will be inferred from the model config.

Default: `auto`

#### \--mamba-ssm-cache-dtype

Possible choices: `auto`, `bfloat16`, `float16`, `float32`

The data type to use for the Mamba cache (ssm state only, conv state will still be controlled by mamba\_cache\_dtype). If set to 'auto', the data type for the ssm state will be determined by mamba\_cache\_dtype.

Default: `auto`

#### \--mamba-block-size

Size of a contiguous cache block in number of tokens for mamba cache. Can be set only when prefix caching is enabled. Value must be a multiple of 8 to align with causal\_conv1d kernel.

#### \--prefix-match-unit

The finest token boundary (in tokens) a prefix-cache hit can land on.

Prefix-cache keys are computed every `prefix_match_unit` tokens. It can be set finer than the physical KV cache block sizes (e.g. 32 vs a 1024-token hybrid-model block) as long as every KV cache group's `block_size` is divisible by it, enabling cache hits at boundaries inside a physical block. It controls matching granularity only, not how often states are stored.

This equals to the `hash_block_size` used throughout the KV cache code.

#### \--mamba-cache-mode

Possible choices: `align`, `all`, `none`

The cache strategy for Mamba layers:

- "none": set when prefix caching is disabled.
- "all": cache the mamba state of all tokens at position i \* block\_size. This is the default behavior (for models that support it) when prefix caching is enabled.
- "align": only cache the mamba state of the last token of each scheduler step and when the token is at position i \* block\_size.

Default: `none`

#### \--replayssm-buffer-len

ReplaySSM history buffer length B: with use\_replayssm, standard decode caches recent SSM inputs in a size-B ring buffer and flushes the checkpoint state to HBM every B steps. Default 16.

Default: `16`

#### \--use-replayssm, --no-use-replayssm

Use the ReplaySSM Mamba2 decode kernel: cache recent SSM inputs and skip the per-step full-state store, writing the checkpoint back only on flush. Requires mamba\_cache\_mode 'none' or 'align' (prefix caching) and the Triton mamba backend; standard (non-speculative) decode only. In align mode flushes are most efficient when mamba\_block\_size is a multiple of replayssm\_buffer\_len, but this is not required.

Default: `False`

#### \--kv-offloading-size

Size of the KV cache offloading buffer in GiB. When TP > 1, this is the total buffer size summed across all TP ranks. By default, this is set to None, which means no KV offloading is enabled. When set, vLLM will enable KV cache offloading to CPU using the kv\_offloading\_backend.

#### \--kv-offloading-backend

Possible choices: `lmcache`, `native`

The backend to use for KV cache offloading. Supported backends include 'native' (vLLM native CPU offloading), 'lmcache'. KV offloading is only activated when kv\_offloading\_size is set.

Default: `native`

### OffloadConfig

Configuration for model weight offloading to reduce GPU memory usage.

#### \--offload-backend

Possible choices: `auto`, `prefetch`, `uva`

The backend for weight offloading. Options: - "auto": Selects based on which sub-config has non-default values (prefetch if offload\_group\_size > 0, uva if cpu\_offload\_gb > 0). - "uva": UVA (Unified Virtual Addressing) zero-copy offloading. - "prefetch": Async prefetch with group-based layer offloading.

Default: `auto`

#### \--cpu-offload-gb

The space in GiB to offload to CPU, per GPU. Default is 0, which means no offloading. Intuitively, this argument can be seen as a virtual way to increase the GPU memory size. For example, if you have one 24 GB GPU and set this to 10, virtually you can think of it as a 34 GB GPU. Then you can load a 13B model with BF16 weight, which requires at least 26GB GPU memory. Note that this requires fast CPU-GPU interconnect, as part of the model is loaded from CPU memory to GPU memory on the fly in each model forward pass. This uses UVA (Unified Virtual Addressing) for zero-copy access.

Default: `0`

#### \--cpu-offload-params

The set of parameter name segments to target for CPU offloading. Unmatched parameters are not offloaded. If this set is empty, parameters are offloaded non-selectively until the memory limit defined by `cpu_offload_gb` is reached. Examples: - For parameter name "mlp.experts.w2\_weight": - "experts" or "experts.w2\_weight" will match. - "expert" or "w2" will NOT match (must be exact segments). This allows distinguishing parameters like "w2\_weight" and "w2\_weight\_scale".

Default: `set()`

#### \--offload-group-size

Group every N layers together. Offload last `offload_num_in_group` layers of each group. Default is 0 (disabled). Example: group\_size=8, num\_in\_group=2 offloads layers 6,7,14,15,22,23,... Unlike cpu\_offload\_gb, this uses explicit async prefetching to hide transfer latency.

Default: `0`

#### \--offload-num-in-group

Number of layers to offload per group. Must be <= offload\_group\_size. Default is 1.

Default: `1`

#### \--offload-prefetch-step

Number of layers to prefetch ahead. Higher values hide more latency but use more GPU memory. Default is 1.

Default: `1`

#### \--offload-params

The set of parameter name segments to target for prefetch offloading. Unmatched parameters are not offloaded. If this set is empty, ALL parameters of each offloaded layer are offloaded. Uses segment matching: "w13\_weight" matches "mlp.experts.w13\_weight" but not "mlp.experts.w13\_weight\_scale".

Default: `set()`

### MultiModalConfig

Controls the behavior of multimodal models.

#### \--language-model-only, --no-language-model-only

If True, disables all multimodal inputs by setting all modality limits to 0. Equivalent to setting `--limit-mm-per-prompt` to 0 for every modality.

Default: `False`

#### \--limit-mm-per-prompt

The maximum number of input items and options allowed per prompt for each modality.

Defaults to 999 for each modality.

Legacy format (count only):

Configurable format (with options): {"video": {"count": 1, "num\_frames": 32, "width": 512, "height": 512}, "image": {"count": 5, "width": 512, "height": 512}}

Mixed format (combining both): {"image": 16, "video": {"count": 1, "num\_frames": 32, "width": 512, "height": 512}}

Should either be a valid JSON string or JSON keys passed individually.

Default: `{}`

#### \--enable-mm-embeds, --no-enable-mm-embeds

If `True`, enables passing multimodal embeddings: for [`LLM`](https://docs.vllm.ai/en/stable/api/vllm/entrypoints/llm/#vllm.entrypoints.llm.LLM "            LLM") class, this refers to tensor inputs under `multi_modal_data`; for the OpenAI-compatible server, this refers to chat messages with content `"type": "*_embeds"`.

When enabled with `--limit-mm-per-prompt` set to 0 for a modality, precomputed embeddings skip count validation for that modality, saving memory by not loading encoder modules while still enabling embeddings as an input. Limits greater than 0 still apply to embeddings.

WARNING: The vLLM engine may crash if incorrect shape of embeddings is passed. Only enable this flag for trusted users!

Default: `False`

#### \--media-io-kwargs

Additional args passed to process media inputs, keyed by modalities. For example, to set num\_frames for video, set `--media-io-kwargs '{"video": {"num_frames": 40} }'`

Should either be a valid JSON string or JSON keys passed individually.

Default: `{}`

#### \--mm-processor-kwargs

Arguments to be forwarded to the model's processor for multi-modal data, e.g., image processor. Overrides for the multi-modal processor obtained from `transformers.AutoProcessor.from_pretrained`.

The available overrides depend on the model that is being run.

For example, for Phi-3-Vision: `{"num_crops": 4}`.

Should either be a valid JSON string or JSON keys passed individually.

#### \--mm-processor-cache-gb

The size (in GiB) of the multi-modal processor cache, which is used to avoid re-processing past multi-modal inputs.

This cache is duplicated for each API process and engine core process, resulting in a total memory usage of `mm_processor_cache_gb * (api_server_count + data_parallel_size)`.

Set to `0` to disable this cache completely (not recommended).

Default: `4`

#### \--mm-processor-cache-type

Possible choices: `lru`, `shm`

Type of cache to use for the multi-modal preprocessor/mapper. If `shm`, use shared memory FIFO cache. If `lru`, use mirrored LRU cache.

Default: `lru`

#### \--mm-hasher-algorithm

Possible choices: `blake3`, `sha256`, `sha512`

Hash algorithm to use for multi-modal input caching. Use `"sha256"` or `"sha512"` for FIPS-compliant deployments.

Default: `blake3`

#### \--mm-shm-cache-max-object-size-mb

Size limit (in MiB) for each object stored in the multi-modal processor shared memory cache. Only effective when `mm_processor_cache_type` is `"shm"`.

Default: `128`

#### \--mm-encoder-only, --no-mm-encoder-only

When enabled, skips the language component of the model.

This is usually only valid in disaggregated Encoder process.

Default: `False`

#### \--mm-encoder-tp-mode

Possible choices: `data`, `weights`

Indicates how to optimize multi-modal encoder inference using tensor parallelism (TP).

- `"weights"`: Within the same vLLM engine, split the weights of each layer across TP ranks. (default TP behavior)
- `"data"`: Within the same vLLM engine, split the batched input data across TP ranks to process the data in parallel, while hosting the full weights on each TP rank. This batch-level DP is not to be confused with API request-level DP (which is controlled by `--data-parallel-size`). This is only supported on a per-model basis and falls back to `"weights"` if the encoder does not support DP.

Default: `weights`

#### \--mm-encoder-attn-backend

Optional override for the multi-modal encoder attention backend when using vision transformers. Accepts any value from `vllm.v1.attention.backends.registry.AttentionBackendEnum` (e.g. `FLASH_ATTN`).

#### \--mm-encoder-attn-dtype

Possible choices: `fp8`, `None`

Optional dtype override for ViT encoder attention. Set to `"fp8"` to enable FP8 quantization via the FlashInfer cuDNN backend. When set to `"fp8"` without a scale file, dynamic scaling is used automatically. See docs/features/quantization/fp8\_vit\_attn.md for details.

#### \--mm-encoder-fp8-scale-path

Path to a JSON file containing per-layer FP8 Q/K/V scales for ViT encoder attention. When provided (with `mm_encoder_attn_dtype="fp8"`), static scaling is used. When omitted, dynamic scaling is used.

#### \--mm-encoder-fp8-scale-save-path

When set with dynamic FP8 scaling (`mm_encoder_attn_dtype="fp8"` and no `mm_encoder_fp8_scale_path`), saves the calibrated scales to this file after the amax history buffer is full. The saved file can then be used as `mm_encoder_fp8_scale_path` in subsequent runs.

#### \--mm-encoder-fp8-scale-save-margin

Safety margin multiplied onto scales when auto-saving. A value > 1 leaves headroom so that inputs with larger activations than the calibration set do not overflow FP8 range. Default 1.5.

Default: `1.5`

#### \--interleave-mm-strings, --no-interleave-mm-strings

Enable fully interleaved support for multimodal prompts, while using --chat-template-content-format=string.

Default: `False`

#### \--skip-mm-profiling, --no-skip-mm-profiling

When enabled, skips multimodal memory profiling and only profiles with language backbone model during engine initialization.

This reduces engine startup time but shifts the responsibility to users for estimating the peak memory usage of the activation of multimodal encoder and embedding cache.

Default: `False`

#### \--video-pruning-rate

Fraction of video tokens to prune from each video. Value sits in range \[0;1); pruning is enabled when it is greater than 0. The pruning algorithm is selected by `video_pruning_method`.

#### \--video-pruning-method

Possible choices: `evs`, `vidcom2`

Video token pruning algorithm applied when `video_pruning_rate` > 0: - "evs": Efficient Video Sampling. - "vidcom2": Video Compression Commander.

Default: `evs`

#### \--mm-tensor-ipc

Possible choices: `direct_rpc`, `torch_shm`

IPC (inter-process communication) method for multimodal tensors. - "direct\_rpc": Use msgspec serialization via RPC - "torch\_shm": Use torch.multiprocessing shared memory for zero-copy IPC Defaults to "direct\_rpc".

Default: `direct_rpc`

#### \--mm-ipc-gpu-memory-gb

Amount of GPU memory (in GiB) sequestered on the engine's device for GPU-side multimodal work in the API-server (frontend) process, such as hardware video decoding.

This budget is carved out of the engine's KV-cache memory so the headroom physically exists, and frontend GPU decode paths acquire from a blocking byte-counting semaphore of this size before allocating on the device.

Set to `0` (default) to disable frontend GPU multimodal memory gating.

Default: `0`

### LoRAConfig

Configuration for LoRA.

#### \--enable-lora, --no-enable-lora

If True, enable handling of LoRA adapters.

#### \--max-loras

Max number of LoRAs in a single batch.

Default: `1`

#### \--max-lora-rank

Possible choices: `1`, `8`, `16`, `32`, `64`, `128`, `256`, `320`, `512`

Max LoRA rank.

Default: `16`

#### \--lora-dtype

Data type for LoRA. If auto, will default to base model dtype.

Default: `auto`

#### \--enable-tower-connector-lora, --no-enable-tower-connector-lora

If `True`, LoRA support for the tower (vision encoder) and connector of multimodal models will be enabled. This is an experimental feature and currently only supports some MM models such as the Qwen VL series. The default is False.

Default: `False`

#### \--max-cpu-loras

Maximum number of LoRAs to store in CPU memory. Must be >= than `max_loras`.

#### \--fully-sharded-loras, --no-fully-sharded-loras

By default, only half of the LoRA computation is sharded with tensor parallelism. Enabling this will use the fully sharded layers. At high sequence length, max rank or tensor parallel size, this is likely faster.

Default: `False`

#### \--lora-target-modules

Restrict LoRA to specific module suffixes (e.g., \["o\_proj", "qkv\_proj"\]). If None, all supported LoRA modules are used. This allows deployment-time control over which modules have LoRA applied, useful for performance tuning.

#### \--default-mm-loras

Dictionary mapping specific modalities to LoRA model paths; this field is only applicable to multimodal models and should be leveraged when a model always expects a LoRA to be active when a given modality is present. Note that currently, if a request provides multiple additional modalities, each of which have their own LoRA, we do NOT apply default\_mm\_loras because we currently only support one lora adapter per prompt. When run in offline mode, the lora IDs for n modalities will be automatically assigned to 1-n with the names of the modalities in alphabetic order.

Should either be a valid JSON string or JSON keys passed individually.

#### \--specialize-active-lora, --no-specialize-active-lora

Whether to construct lora kernel grid by the number of active LoRA adapters. When set to True, separate cuda graphs will be captured for different counts of active LoRAs (powers of 2 up to max\_loras), which can improve performance for variable LoRA usage patterns at the cost of increased startup time and memory usage. Only takes effect when cudagraph\_specialize\_lora is True.

Default: `False`

#### \--enable-mixed-moe-lora-format, --no-enable-mixed-moe-lora-format

If True, force the engine to use the universal 2D MoE LoRA wrapper (`FusedMoEWithLoRA`) regardless of the model's `is_3d_moe_weight` flag, so that 2D-format and 3D-format MoE LoRA adapters can be served in the same deployment. Only meaningful for MoE models; ignored otherwise. Default False keeps the existing model-driven behavior.

Default: `False`

#### \--enable-moe-shared-loras, --no-enable-moe-shared-loras

If True, load MoE expert adapters in the "shared-outer" layout, where the gate/up (`w1` / `w3`) lora\_A and the down (`w2`) lora\_B are shared across all experts (stored once with expert-dim 1) instead of per-expert. The shared factors are broadcast to the expert count at kernel time. Only meaningful for MoE models whose adapters use this layout; ignored otherwise.

Default: `False`

### ObservabilityConfig

Configuration for observability - metrics and tracing.

#### \--show-hidden-metrics-for-version

Enable deprecated Prometheus metrics that have been hidden since the specified version. For example, if a previously deprecated metric has been hidden since the v0.7.0 release, you use `--show-hidden-metrics-for-version=0.7` as a temporary escape hatch while you migrate to new metrics. The metric is likely to be removed completely in an upcoming release.

#### \--otlp-traces-endpoint

Target URL to which OpenTelemetry traces will be sent.

#### \--collect-detailed-traces

Possible choices: `all`, `model`, `worker`, `None`, `model,worker`, `model,all`, `worker,model`, `worker,all`, `all,model`, `all,worker`

It makes sense to set this only if `--otlp-traces-endpoint` is set. If set, it will collect detailed traces for the specified modules. This involves use of possibly costly and or blocking operations and hence might have a performance impact.

Note that collecting detailed timing information for each request can be expensive.

#### \--kv-cache-metrics, --no-kv-cache-metrics

Enable KV cache residency metrics (lifetime, idle time, reuse gaps). Uses sampling to minimize overhead. Requires log stats to be enabled (i.e., --disable-log-stats not set).

Default: `False`

#### \--kv-cache-metrics-sample

Sampling rate for KV cache metrics (0.0, 1.0\]. Default 0.01 = 1%% of blocks.

Default: `0.01`

#### \--cudagraph-metrics, --no-cudagraph-metrics

Enable CUDA graph metrics (number of padded/unpadded tokens, runtime cudagraph dispatch modes, and their observed frequencies at every logging interval).

Default: `False`

#### \--enable-layerwise-nvtx-tracing, --no-enable-layerwise-nvtx-tracing

Enable layerwise NVTX tracing. This traces the execution of each layer or module in the model and attach information such as input/output shapes to nvtx range markers. Noted that this doesn't work with CUDA graphs enabled.

Default: `False`

#### \--enable-mfu-metrics, --no-enable-mfu-metrics

Enable Model FLOPs Utilization (MFU) metrics.

Default: `False`

#### \--enable-logging-iteration-details, --no-enable-logging-iteration-details

Enable detailed logging of iteration details. If set, vllm EngineCore will log iteration details This includes number of context/generation requests and tokens and the elapsed cpu time for the iteration.

Default: `False`

#### \--jit-monitor-mode

Possible choices: `error`, `warn`

How to handle post-warmup JIT compilation events.

Default: `warn`

#### \--jit-monitor-verbose, --no-jit-monitor-verbose

Log every monitored JIT compile with runtime details. This can emit many logs and add overhead, so it is intended for debugging.

Default: `False`

### SchedulerConfig

Scheduler configuration.

#### \--max-num-batched-tokens

Maximum number of tokens that can be processed in a single iteration.

The default value here is mainly for convenience when testing. In real usage, this should be set in `EngineArgs.create_engine_config`.

Parse human-readable integers like '1k', '2M', etc. Including decimal values with decimal multipliers.

```js
Examples:
- '1k' -> 1,000
- '1K' -> 1,024
- '25.6k' -> 25,600
```

#### \--max-num-scheduled-tokens

Maximum number of tokens that the scheduler may issue in a single iteration.

This is usually equal to max\_num\_batched\_tokens, but can be smaller in cases when the model might append tokens into the batch (such as speculative decoding). Defaults to max\_num\_batched\_tokens.

Parse human-readable integers like '1k', '2M', etc. Including decimal values with decimal multipliers.

```js
Examples:
- '1k' -> 1,000
- '1K' -> 1,024
- '25.6k' -> 25,600
```

#### \--max-num-seqs

Maximum number of sequences to be processed in a single iteration.

The default value here is mainly for convenience when testing. In real usage, this should be set in `EngineArgs.create_engine_config`.

#### \--long-prefill-token-threshold

For chunked prefill, a request is considered long if the prompt is longer than this number of tokens. 0 disables the cap (default).

Default: `0`

#### \--scheduling-policy

Possible choices: `fcfs`, `priority`

The scheduling policy to use:

- "fcfs" means first come first served, i.e. requests are handled in order of arrival.
- "priority" means requests are handled based on given priority (lower value means earlier handling) and time of arrival deciding any ties).

Default: `fcfs`

#### \--enable-chunked-prefill, --no-enable-chunked-prefill

If True, prefill requests can be chunked based on the remaining `max_num_batched_tokens`.

The default value here is mainly for convenience when testing. In real usage, this should be set in `EngineArgs.create_engine_config`.

#### \--disable-chunked-mm-input, --no-disable-chunked-mm-input

If set to true and chunked prefill is enabled, we do not want to partially schedule a multimodal item. Only used in V1 This ensures that if a request has a mixed prompt (like text tokens TTTT followed by image tokens IIIIIIIIII) where only some image tokens can be scheduled (like TTTTIIIII, leaving IIIII), it will be scheduled as TTTT in one step and IIIIIIIIII in the next.

Default: `False`

#### \--scheduler-cls

The scheduler class to use. "vllm.v1.core.sched.scheduler.Scheduler" is the default scheduler. Can be a class directly or the path to a class of form "mod.custom\_class".

#### \--scheduler-reserve-full-isl, --no-scheduler-reserve-full-isl

If True, the scheduler checks whether the full input sequence length fits in the KV cache before admitting a new request, rather than only checking the first chunk. Prevents over-admission and KV cache thrashing with chunked prefill.

Default: `True`

#### \--watermark

Fraction of total KV cache blocks to keep free (the watermark) when admitting waiting or preempted requests into the running queue. This headroom helps avoid frequent KV cache eviction and the resulting repeated preemption of requests when GPU memory is scarce. Must be in the range \[0.0, 1.0); 0.0 (the default) disables the watermark.

Default: `0.0`

#### \--prefill-schedule-interval

For data-parallel deployments, only admit new prefill requests once every N engine steps, aligned across DP ranks, to better balance per-step forward-pass times.

Default: `1`

#### \--disable-hybrid-kv-cache-manager, --no-disable-hybrid-kv-cache-manager

If set to True, KV cache manager will allocate the same size of KV cache for all attention layers even if there are multiple type of attention layers like full attention and sliding window attention. If set to None, the default value will be determined based on the environment and starting configuration.

#### \--async-scheduling, --no-async-scheduling

If set to False, disable async scheduling. Async scheduling helps to avoid gaps in GPU utilization, leading to better latency and throughput.

#### \--stream-interval

The interval (or buffer size) for streaming in terms of token length. A smaller value (1) makes streaming smoother by sending each token immediately, while a larger value (e.g., 10) reduces host overhead and may increase throughput by batching multiple tokens before sending.

Default: `1`

### CompilationConfig

Configuration for compilation.

You must pass CompilationConfig to VLLMConfig constructor. VLLMConfig's post\_init does further initialization. If used outside of the VLLMConfig, some fields will be left in an improper state.

It contains PassConfig, which controls the custom fusion/transformation passes. The rest has three parts:

- Top-level Compilation control:
	- [`mode`](https://docs.vllm.ai/en/stable/api/vllm/config/#vllm.config.CompilationConfig.mode "            mode = None
		class-attribute
		instance-attribute
		")
		- [`debug_dump_path`](https://docs.vllm.ai/en/stable/api/vllm/config/#vllm.config.CompilationConfig.debug_dump_path "            debug_dump_path = None
		class-attribute
		instance-attribute
		")
		- [`cache_dir`](https://docs.vllm.ai/en/stable/api/vllm/config/#vllm.config.CompilationConfig.cache_dir "            cache_dir = ''
		class-attribute
		instance-attribute
		")
		- [`backend`](https://docs.vllm.ai/en/stable/api/vllm/config/#vllm.config.CompilationConfig.backend "            backend = ''
		class-attribute
		instance-attribute
		")
		- [`custom_ops`](https://docs.vllm.ai/en/stable/api/vllm/config/#vllm.config.CompilationConfig.custom_ops "            custom_ops = field(default_factory=list)
		class-attribute
		instance-attribute
		")
		- [`splitting_ops`](https://docs.vllm.ai/en/stable/api/vllm/config/#vllm.config.CompilationConfig.splitting_ops "            splitting_ops = None
		class-attribute
		instance-attribute
		")
		- [`compile_mm_encoder`](https://docs.vllm.ai/en/stable/api/vllm/config/#vllm.config.CompilationConfig.compile_mm_encoder "            compile_mm_encoder = False
		class-attribute
		instance-attribute
		")
- CudaGraph capture:
	- [`cudagraph_mode`](https://docs.vllm.ai/en/stable/api/vllm/config/#vllm.config.CompilationConfig.cudagraph_mode "            cudagraph_mode = None
		class-attribute
		instance-attribute
		")
		- [`cudagraph_capture_sizes`](https://docs.vllm.ai/en/stable/api/vllm/config/#vllm.config.CompilationConfig.cudagraph_capture_sizes "            cudagraph_capture_sizes = None
		class-attribute
		instance-attribute
		")
		- [`max_cudagraph_capture_size`](https://docs.vllm.ai/en/stable/api/vllm/config/#vllm.config.CompilationConfig.max_cudagraph_capture_size "            max_cudagraph_capture_size = None
		class-attribute
		instance-attribute
		")
		- [`cudagraph_num_of_warmups`](https://docs.vllm.ai/en/stable/api/vllm/config/#vllm.config.CompilationConfig.cudagraph_num_of_warmups "            cudagraph_num_of_warmups = 0
		class-attribute
		instance-attribute
		")
		- [`cudagraph_copy_inputs`](https://docs.vllm.ai/en/stable/api/vllm/config/#vllm.config.CompilationConfig.cudagraph_copy_inputs "            cudagraph_copy_inputs = False
		class-attribute
		instance-attribute
		")
- Inductor compilation:
	- [`compile_sizes`](https://docs.vllm.ai/en/stable/api/vllm/config/#vllm.config.CompilationConfig.compile_sizes "            compile_sizes = None
		class-attribute
		instance-attribute
		")
		- \[`compile_ranges_endpoints`\] \[vllm.config.CompilationConfig.compile\_ranges\_endpoints\]
		- [`inductor_compile_config`](https://docs.vllm.ai/en/stable/api/vllm/config/#vllm.config.CompilationConfig.inductor_compile_config "            inductor_compile_config = field(default_factory=dict)
		class-attribute
		instance-attribute
		")
		- [`inductor_passes`](https://docs.vllm.ai/en/stable/api/vllm/config/#vllm.config.CompilationConfig.inductor_passes "            inductor_passes = field(default_factory=dict)
		class-attribute
		instance-attribute
		")
		- custom inductor passes

Why we have different sizes for cudagraph and inductor: - cudagraph: a cudagraph captured for a specific size can only be used for the same size. We need to capture all the sizes we want to use. - inductor: a graph compiled by inductor for a general shape can be used for different sizes. Inductor can also compile for specific sizes, where it can have more information to optimize the graph with fully static shapes. However, we find the general shape compilation is sufficient for most cases. It might be beneficial to compile for certain small batchsizes, where inductor is good at optimizing.

#### \--cudagraph-capture-sizes

Sizes to capture cudagraph. - None (default): capture sizes are inferred from vllm config. - list\[int\]: capture sizes are specified as given.

#### \--max-cudagraph-capture-size

The maximum cudagraph capture size.

If cudagraph\_capture\_sizes is specified, this will be set to the largest size in that list (or checked for consistency if specified). If cudagraph\_capture\_sizes is not specified, the list of sizes is generated automatically following the pattern:

```js
[1, 2, 4] + list(range(8, 256, 8)) + list(
range(256, max_cudagraph_capture_size + 1, 16))
```

If not specified, max\_cudagraph\_capture\_size is set to min(max\_num\_seqs\*2, 512) by default. This voids OOM in tight memory scenarios with small max\_num\_seqs, and prevents capture of many large graphs (>512) that would greatly increase startup time with limited performance benefit.

### KernelConfig

Configuration for kernel selection and warmup behavior.

#### \--ir-op-priority

vLLM IR op priority for dispatching/lowering during the forward pass. Platform defaults appended automatically during VllmConfig.**post\_init**.

Should either be a valid JSON string or JSON keys passed individually.

Default: `IrOpPriorityConfig(rms_norm=[], fused_add_rms_norm=[])`

#### \--enable-flashinfer-autotune, --no-enable-flashinfer-autotune

If True, run FlashInfer autotuning during kernel warmup.

#### \--enable-bf16x3-router-gemm, --no-enable-bf16x3-router-gemm

If True, use the experimental SM100 BF16x3 CuteDSL router GEMM.

Default: `False`

#### \--moe-backend

Possible choices: `aiter`, `auto`, `batched_triton`, `cutlass`, `deep_gemm`, `deep_gemm_mega_moe`, `emulation`, `flashinfer_b12x`, `flashinfer_cutedsl`, `flashinfer_cutlass`, `flashinfer_trtllm`, `flydsl`, `hpc`, `humming`, `marlin`, `triton`, `triton_unfused`

Backend for MoE expert computation kernels. Available options:

- "auto": Automatically select the best backend based on model and hardware
- "triton": Use Triton-based fused MoE kernels
- "batched\_triton": Use batched Triton experts (moe\_mmk) on the batched activation format (\[E\_local, max\_num\_tokens, K\])
- "deep\_gemm": Use DeepGEMM kernels (FP8 block-quantized only)
- "deep\_gemm\_mega\_moe": Use DeepGEMM mega MoE kernels
- "cutlass": Use vLLM CUTLASS kernels
- "flashinfer\_trtllm": Use FlashInfer with TRTLLM-GEN kernels
- "flashinfer\_cutlass": Use FlashInfer with CUTLASS kernels
- "flashinfer\_cutedsl": Use FlashInfer with CuteDSL kernels (FP4 only)
- "flashinfer\_b12x": Use FlashInfer CuteDSL fused MoE for SM12x (RTX Pro 6000 / DGX Spark)
- "marlin": Use Marlin kernels (weight-only quantization)
- "humming": Use Humming Mixed Precision kernels
- "triton\_unfused": Use Triton unfused MoE kernels
- "aiter": Use AMD AITer kernels (ROCm only)
- "flydsl": Use AMD FlyDSL kernels (ROCm only)
- "hpc": Use HPC kernels (FP8 and Hopper only)
- "emulation": use BF16/FP16 GEMM, dequantizing weights and running QDQ on activations.

Default: `auto`

#### \--linear-backend

Possible choices: `aiter`, `auto`, `conch`, `cutlass`, `deep_gemm`, `emulation`, `exllama`, `fbgemm`, `flashinfer_b12x`, `flashinfer_cudnn`, `flashinfer_cutedsl`, `flashinfer_cutlass`, `flashinfer_trtllm`, `humming`, `machete`, `marlin`, `torch`, `triton`, `xpu`, `xpu_woq`

Backend for quantized linear layer GEMM kernels. Available options:

- "auto": Automatically select the best backend based on model and hardware
- "cutlass": Use CUTLASS-based kernels
- "flashinfer\_cutlass": Use FlashInfer with CUTLASS kernels
- "flashinfer\_cutedsl": Use FlashInfer with CuTe-DSL kernels (NVFP4, MXFP8)
- "flashinfer\_trtllm": Use FlashInfer with TensorRT-LLM kernels
- "flashinfer\_cudnn": Use FlashInfer with cuDNN kernels
- "flashinfer\_b12x": Use FlashInfer b12x CuteDSL NVFP4 GEMM (SM120+)
- "marlin": Use Marlin kernels
- "triton": Use Triton-based kernels
- "deep\_gemm": Use DeepGEMM kernels
- "torch": Use PyTorch native scaled\_mm kernels
- "aiter": Use AMD AITer kernels (ROCm only)
- "machete": Use Machete kernels (mixed-precision)
- "fbgemm": Use FBGEMM kernels
- "conch": Use Conch mixed-precision kernels
- "exllama": Use Exllama mixed-precision kernels
- "emulation": Use slow dequant-to-BF16 emulation (for testing only)
- "xpu": Use XPU kernels
- "xpu\_woq": Use XPU kernels for weight-only quantization (e.g. W8A16)

Default: `auto`

### VllmConfig

Dataclass which contains all vllm-related configuration. This simplifies passing around the distinct configurations in the codebase.

#### \--speculative-config, -sc

Speculative decoding configuration.

API docs: [`vllm.config.SpeculativeConfig`](https://docs.vllm.ai/en/stable/api/vllm/config/#vllm.config.SpeculativeConfig "            SpeculativeConfig")

Should either be a valid JSON string or JSON keys passed individually.

#### \--spec-method

Possible choices: `bailing_hybrid_mtp`, `custom_class`, `deepseek_mtp`, `dflash`, `draft_model`, `dspark`, `eagle`, `eagle3`, `ernie_mtp`, `exaone4_5_mtp`, `exaone_moe_mtp`, `extract_hidden_states`, `gemma4_mtp`, `glm4_moe_lite_mtp`, `glm4_moe_mtp`, `glm_ocr_mtp`, `hy_v3_mtp`, `inkling_mtp`, `kimi_k3_mtp`, `longcat_flash_mtp`, `medusa`, `mimo_mtp`, `mimo_v2_mtp`, `minimax_m3_mtp`, `mlp_speculator`, `mtp`, `nemotron_h_mtp`, `ngram`, `ngram_gpu`, `pangu_ultra_moe_mtp`, `qwen3_5_mtp`, `qwen3_next_mtp`, `step3p5_mtp`, `suffix`, `None`

The name of the speculative method to use. If users provide and set the `model` param, the speculative method type will be detected automatically if possible, if `model` param is not provided, the method name must be provided.

If using `ngram` method, the related configuration `prompt_lookup_max` and `prompt_lookup_min` should be considered.

#### \--spec-model

The name of the draft model, eagle head, or additional weights, if provided.

#### \--spec-tokens

The number of speculative tokens, if provided. It will default to the number in the draft model config if present, otherwise, it is required.

#### \--diffusion-config, -dc

Diffusion LLM (dLLM) configuration.

API docs: [`vllm.config.DiffusionConfig`](https://docs.vllm.ai/en/stable/api/vllm/config/#vllm.config.DiffusionConfig "            DiffusionConfig")

Should either be a valid JSON string or JSON keys passed individually.

#### \--kv-transfer-config

The configurations for distributed KV cache transfer.

API docs: [`vllm.config.KVTransferConfig`](https://docs.vllm.ai/en/stable/api/vllm/config/#vllm.config.KVTransferConfig "            KVTransferConfig")

Should either be a valid JSON string or JSON keys passed individually.

#### \--kv-events-config

The configurations for event publishing.

API docs: [`vllm.config.KVEventsConfig`](https://docs.vllm.ai/en/stable/api/vllm/config/#vllm.config.KVEventsConfig "            KVEventsConfig")

Should either be a valid JSON string or JSON keys passed individually.

#### \--ec-transfer-config

The configurations for distributed EC cache transfer.

API docs: [`vllm.config.ECTransferConfig`](https://docs.vllm.ai/en/stable/api/vllm/config/#vllm.config.ECTransferConfig "            ECTransferConfig")

Should either be a valid JSON string or JSON keys passed individually.

#### \--compilation-config, -cc

`torch.compile` and cudagraph capture configuration for the model.

As a shorthand, one can append compilation arguments via -cc.parameter=argument such as `-cc.mode=3` (same as `-cc='{"mode":3}'`).

You can specify the full compilation config like so: `{"mode": 3, "cudagraph_capture_sizes": [1, 2, 4, 8]}`

API docs: [`vllm.config.CompilationConfig`](https://docs.vllm.ai/en/stable/api/vllm/config/#vllm.config.CompilationConfig "            CompilationConfig")

Should either be a valid JSON string or JSON keys passed individually.

Default: `{'mode': None, 'debug_dump_path': None, 'cache_dir': '', 'compile_cache_save_format': 'binary', 'backend': 'inductor', 'custom_ops': [], 'ir_enable_torch_wrap': None, 'splitting_ops': None, 'compile_mm_encoder': False, 'cudagraph_mm_encoder': False, 'encoder_cudagraph_token_budgets': [], 'encoder_cudagraph_max_vision_items_per_batch': 0, 'encoder_cudagraph_max_frames_per_batch': None, 'compile_sizes': None, 'compile_ranges_endpoints': None, 'inductor_compile_config': {'enable_auto_functionalized_v2': False, 'combo_kernels': True, 'benchmark_combo_kernel': True}, 'inductor_passes': {}, 'cudagraph_mode': None, 'cudagraph_num_of_warmups': 0, 'cudagraph_capture_sizes': None, 'cudagraph_copy_inputs': False, 'cudagraph_specialize_lora': True, 'use_inductor_graph_partition': None, 'pass_config': {}, 'max_cudagraph_capture_size': None, 'dynamic_shapes_config': {'type': <DynamicShapesType.BACKED: 'backed'>, 'evaluate_guards': False, 'assume_32_bit_indexing': False}, 'local_cache_dir': None, 'fast_moe_cold_start': None, 'static_all_moe_layers': []}`

#### \--attention-config, -ac

Attention configuration.

API docs: [`vllm.config.AttentionConfig`](https://docs.vllm.ai/en/stable/api/vllm/config/#vllm.config.AttentionConfig "            AttentionConfig")

Should either be a valid JSON string or JSON keys passed individually.

Default: `AttentionConfig(backend=None, minimax_m3_msa_decode_backend='triton', backend_per_kind={}, flash_attn_version=None, use_prefill_decode_attention=False, flash_attn_max_num_splits_for_cuda_graph=32, tq_max_kv_splits_for_cuda_graph=32, use_trtllm_attention=None, disable_flashinfer_q_quantization=False, mla_prefill_backend=None, use_prefill_query_quantization=False, use_fp4_indexer_cache=False, indexer_kv_dtype='bf16', use_non_causal=False, sparse_mla_force_mqa=False, flex_attn_block_m=None, flex_attn_block_n=None, flex_attn_q_block_size=None, flex_attn_kv_block_size=None)`

#### \--reasoning-config

The configurations for reasoning model.

API docs: [`vllm.config.ReasoningConfig`](https://docs.vllm.ai/en/stable/api/vllm/config/#vllm.config.ReasoningConfig "            ReasoningConfig")

Should either be a valid JSON string or JSON keys passed individually.

#### \--kernel-config

Kernel configuration.

API docs: [`vllm.config.KernelConfig`](https://docs.vllm.ai/en/stable/api/vllm/config/#vllm.config.KernelConfig "            KernelConfig")

Should either be a valid JSON string or JSON keys passed individually.

Default: `KernelConfig(ir_op_priority=IrOpPriorityConfig(rms_norm=[], fused_add_rms_norm=[]), enable_flashinfer_autotune=None, enable_cutedsl_warmup=True, enable_jit_warmup=True, enable_bf16x3_router_gemm=False, moe_backend='auto', linear_backend='auto')`

#### \--additional-config

Additional config for specified platform. Different platforms may support different configs. Make sure the configs are valid for the platform you are using. Contents must be hashable.

Default: `{}`

#### \--structured-outputs-config

Structured outputs configuration.

API docs: [`vllm.config.StructuredOutputsConfig`](https://docs.vllm.ai/en/stable/api/vllm/config/#vllm.config.StructuredOutputsConfig "            StructuredOutputsConfig")

Should either be a valid JSON string or JSON keys passed individually.

Default: `StructuredOutputsConfig(backend='auto', disable_any_whitespace=False, disable_additional_properties=False, reasoning_parser='', reasoning_parser_plugin='', enable_in_reasoning=False)`

#### \--profiler-config

Profiling configuration.

API docs: [`vllm.config.ProfilerConfig`](https://docs.vllm.ai/en/stable/api/vllm/config/#vllm.config.ProfilerConfig "            ProfilerConfig")

Should either be a valid JSON string or JSON keys passed individually.

Default: `ProfilerConfig(profiler=None, torch_profiler_dir='', torch_profiler_with_stack=True, torch_profiler_with_flops=False, torch_profiler_use_gzip=True, torch_profiler_dump_cuda_time_total=True, torch_profiler_record_shapes=False, torch_profiler_with_memory=False, capture_torch_profiler=False, detailed_trace_annotation=False, ignore_frontend=False, delay_iterations=0, max_iterations=0, warmup_iterations=0, active_iterations=5, wait_iterations=0)`

#### \--optimization-level

The optimization level. These levels trade startup time cost for performance, with -O0 having the best startup time and -O3 having the best performance. -O2 is used by default. See OptimizationLevel for full description.

Default: `2`

#### \--performance-mode

Possible choices: `balanced`, `interactivity`, `throughput`

Performance mode for runtime behavior, 'balanced' is the default. 'interactivity' favors low end-to-end per-request latency at small batch sizes (fine-grained CUDA graphs, latency-oriented kernels). 'throughput' favors aggregate tokens/sec at high concurrency (larger CUDA graphs, more aggressive batching, throughput-oriented kernels).

Default: `balanced`

#### \--weight-transfer-config

The configurations for weight transfer during RL training.

API docs: [`vllm.config.WeightTransferConfig`](https://docs.vllm.ai/en/stable/api/vllm/config/#vllm.config.WeightTransferConfig "            WeightTransferConfig")

Should either be a valid JSON string or JSON keys passed individually.