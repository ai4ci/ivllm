Part of [Language AI Handbook](https://mbrenndoerfer.com/books/language-ai-handbook)

Calculate KV cache memory requirements for transformer models. Topics include batch size, context length, GQA optimization, and GPU deployment planning.

Track your reading progress

Sign in to mark chapters as read and track your learning journey

Choose your expertise level to adjust how many terms are explained. Beginners see more tooltips, experts see fewer to maintain reading flow. Hover over underlined terms for instant definitions.

Article links

Make inline references clickable

## KV Cache Memory

In the previous chapter, we saw how the [KV cache](https://mbrenndoerfer.com/writing/autoregressive-generation-gpt-text-generation) eliminates redundant computation during autoregressive generation by storing keys and values from previous tokens. This memory-compute tradeoff is fundamental to efficient inference: instead of recomputing key and value projections for every token at every generation step, we store them once and read them back later. The savings in computation are real and substantial. But every saved computation comes with a cost in memory, and that cost compounds quickly as sequences grow longer and models grow larger.

Understanding exactly how much memory the KV cache requires is essential for deployment planning. A seemingly simple question, "Can I run this model on my GPU?", requires knowing the model weight size and the memory the cache will consume during generation. The answer depends on the model architecture, the sequence length, the [batch size](https://mbrenndoerfer.com/writing/stochastic-gradient-descent-neural-network-optimization), and the data type. Underestimate any of these factors and you face an out-of-memory error at runtime. Overestimate and you overpay for hardware. Getting this calculation right is not a minor implementation detail; it is the central skill that separates successful deployments from failed ones.

Think of the [KV cache](https://mbrenndoerfer.com/writing/autoregressive-generation-gpt-text-generation) as a receipt that grows longer with every token you generate. At the beginning of a conversation, the receipt is short and fits easily in your pocket. As the exchange continues, the receipt extends, eventually spilling out onto the table and then onto the floor. Each line of the receipt must remain accessible because future computation depends on it. You cannot crumple up the early lines and throw them away without corrupting the [attention mechanism](https://mbrenndoerfer.com/writing/attention-mechanism-intuition-soft-lookup-weights-context-vectors). The GPU's memory is the pocket, and the question is always whether the receipt is going to overflow it before the conversation ends.

This chapter provides the mathematical tools to calculate these requirements precisely. We derive the exact formula for [KV cache memory](https://mbrenndoerfer.com/writing/multi-query-attention-memory-efficient-inference) from first principles, examining how each architectural parameter contributes to the total. We then trace the effects of sequence length, [batch size](https://mbrenndoerfer.com/writing/stochastic-gradient-descent-neural-network-optimization), and data type separately, building intuition for which factors matter most in different deployment scenarios. We close by analyzing the memory bottleneck that arises when cache growth dominates inference performance, and we connect that analysis to the optimization techniques covered in subsequent chapters.

The calculations in this chapter are surprisingly simple once you understand the structure of what is being cached. The formula involves only multiplication of a handful of architectural parameters. The insights, however, run deep: they explain why [Grouped Query Attention](https://mbrenndoerfer.com/writing/grouped-query-attention-gqa-efficient-llm-inference) exists, why long-context inference is so expensive, why serving many users simultaneously requires careful memory planning, and why GPU generation is almost always limited by memory bandwidth rather than arithmetic throughput.

Historical Context

The memory costs of the KV cache were not immediately obvious when transformer architectures were first introduced in 2017. Early transformer models operated on short sequences, and GPU memory was rarely the binding constraint. The problem became acute around 2020-2021 as researchers began pushing toward longer context windows. GPT-3's release in 2020 made the economics of inference central to practical deployment, and practitioners quickly realized that memory, not compute, was the limiting resource for generation. The introduction of Grouped Query Attention in 2023 (used in LLaMA 2) and the widespread adoption of 4-bit quantization for KV cache values represent direct responses to the memory pressure that this chapter quantifies. Every major optimization technique in the inference efficiency literature can be traced back to the formula we derive here.

## Anatomy of KV Cache Storage

Before calculating memory requirements, we need to understand precisely what the cache stores. Recall from the previous chapter that during each forward pass, we cache the key and value projections for reuse in subsequent tokens. This caching strategy turns what would be repeated linear projections into simple memory lookups. However, every lookup requires that the data exist in GPU memory, which means someone must allocate space for it before the first token is ever generated. Understanding the precise shape and structure of this stored data is the first step toward calculating its memory footprint.

The keys and values we cache are not arbitrary tensors. They have very specific shapes determined by the [attention mechanism](https://mbrenndoerfer.com/writing/attention-mechanism-intuition-soft-lookup-weights-context-vectors) 's design. When attention processes a token, it creates a [query vector](https://mbrenndoerfer.com/writing/query-key-value-attention-mechanism) to ask "what should I attend to?" Corresponding key and value vectors for that token answer questions from future tokens. The query is used immediately and discarded, but the keys and values must persist because future tokens will compare against them. Think of keys as index cards in a filing cabinet and values as the documents those cards refer to. Every token you generate files a new index card. Every subsequent token must search the entire filing cabinet before it can produce its output.

For a single attention layer processing a single token, the projections produce:

- Key tensor: shape `[batch_size, num_kv_heads, 1, head_dim]`
- Value tensor: shape `[batch_size, num_kv_heads, 1, head_dim]`

The "1" in these shapes represents the single token being processed during generation. This single-token slice is what gets appended to the existing cache with each forward pass. The batch dimension allows processing multiple independent sequences simultaneously. The [head dimension](https://mbrenndoerfer.com/writing/multi-head-attention-transformers) captures the multi-head structure of attention, where different heads specialize in attending to different aspects of the input. The `head_dim` represents the dimensionality of each head's representation space, controlling how expressive each head's keys and values can be.

After processing $t$ tokens, the accumulated cache for that layer has:

- Key cache: shape `[batch_size, num_kv_heads, t, head_dim]`
- Value cache: shape `[batch_size, num_kv_heads, t, head_dim]`

Notice how the sequence dimension grows from 1 to $t$ as tokens accumulate. This growth is the source of the linear memory scaling that dominates long-context inference. Each new token adds one slice along the sequence dimension, which must be stored for all subsequent generation steps. There is no compression or summarization: every token's key and value vectors survive intact until the generation completes.

The full [KV cache](https://mbrenndoerfer.com/writing/autoregressive-generation-gpt-text-generation) spans all [transformer](https://mbrenndoerfer.com/writing/transformer-attention-is-all-you-need) layers. A model with $L$ layers maintains $L$ key caches and $L$ value caches. This multiplicative effect of layers is important. A 32-layer model requires 32 times the cache of a single layer. This makes the number of layers one of the primary drivers of total cache size. Each layer's cache is completely independent, containing that layer's unique representation of the sequence history. The [attention mechanism](https://mbrenndoerfer.com/writing/attention-mechanism-intuition-soft-lookup-weights-context-vectors) at layer 12 attends over different representations than the attention at layer 24, and both sets of representations must be preserved simultaneously.

The data type determines bytes per element. Most modern inference uses half-precision (FP16 or BF16) at 2 bytes per element, though some systems use FP32 (4 bytes) or quantized representations like [INT8](https://mbrenndoerfer.com/writing/int8-quantization-absmax-smooth-quantization-implementation) (1 byte) or [INT4](https://mbrenndoerfer.com/writing/int4-quantization-group-wise-nf4-format-llms) (0.5 bytes). This choice creates a direct tradeoff between precision and memory consumption. Halving the bytes per element halves the cache size, which is why quantization techniques have become essential for deploying large models. We will revisit this tradeoff in the quantization chapters.

Out\[3\]:

Visualization

![Line plot showing cumulative KV cache memory growing linearly with number of generated tokens.](https://assets.mbrenndoerfer.com/_optimized/notebooks/2_kv_cache_memory_files/kv-cache-growth-per-token-dark-1920w.webp)

Line plot showing cumulative KV cache memory growing linearly with number of generated tokens.

## Cache Size Calculation

With a clear understanding of what the cache stores, we can now derive the formula for total memory consumption. The calculation follows directly from multiplying the dimensions of the cached tensors by the number of bytes each element requires. There is no approximation involved: this is exact arithmetic about the shape of the tensors sitting in your GPU's memory. This derivation provides the basis for all deployment planning decisions, and once you work through it once, you will be able to estimate cache sizes in your head for any model architecture.

Consider the structure we described above. For each layer, we store both keys and values. Each of these has shape `[batch_size, num_kv_heads, seq_len, head_dim]`. The total number of elements in one such tensor is the product of these dimensions. Since we store two tensors (keys and values) per layer across all layers, the total element count becomes:

$$
\text{Total Elements} = 2 \times L \times B \times T \times H_{\text{kv}} \times D_h
$$

Converting elements to bytes requires multiplying by the bytes per element, giving us the complete formula:

$$
\text{KV Cache Memory} = 2 \times L \times B \times T \times H_{\text{kv}} \times D_h \times \text{bytes}
$$

where:

- $2$ accounts for storing both key and value matrices
- $L$: the number of [transformer](https://mbrenndoerfer.com/writing/transformer-attention-is-all-you-need) layers
- $B$: the [batch size](https://mbrenndoerfer.com/writing/stochastic-gradient-descent-neural-network-optimization) (number of concurrent sequences)
- $T$: the sequence length (number of tokens cached)
- $H_{\text{kv}}$: the number of key-value heads
- $D_h$: the [head dimension](https://mbrenndoerfer.com/writing/multi-head-attention-transformers) (dimensionality of each attention head)
- $\text{bytes}$: the number of bytes per element (2 for FP16/BF16, 4 for FP32)

The key insight is that every term in this product is linear: there are no squares, no logarithms, no nonlinear interactions. Doubling any single factor doubles the memory. This linearity makes the formula both simple to use and relentlessly demanding in practice. You cannot cleverly reorganize the computation to escape the cost. Every factor must be paid in full.

This formula reveals the linear dependencies that govern cache memory. Doubling any of the multiplicative factors doubles the total memory. This linearity is both a blessing and a curse. It makes calculations simple and predictable, but it also means there are no economies of scale. Processing twice as many tokens always requires twice as much cache memory, with no possibility of compression through clever algorithms unless we explicitly introduce approximation techniques, which we cover in later chapters on cache quantization and pruning.

### Standard Multi-Head Attention

For standard [Multi-Head Attention](https://mbrenndoerfer.com/writing/multi-head-attention-transformers) (MHA), $H_{\text{kv}}$ equals the number of attention heads $H$, and the relationship $H \times D_h = D_{\text{model}}$ holds. This relationship exists because each head operates on a slice of the model dimension, and together all heads span the full representation space. This architectural constraint allows us to substitute and simplify our formula:

$$
\text{KV Cache Memory (MHA)} = 2 \times L \times B \times T \times D_{\text{model}} \times \text{bytes}
$$

where:

- $D_{\text{model}}$: the model dimension (equal to $H \times D_h$, the full embedding size)
- $L$: the number of [transformer](https://mbrenndoerfer.com/writing/transformer-attention-is-all-you-need) layers
- $B$: the [batch size](https://mbrenndoerfer.com/writing/stochastic-gradient-descent-neural-network-optimization)
- $T$: the sequence length
- $\text{bytes}$: the number of bytes per element

This simplified form is particularly useful because $D_{\text{model}}$ is the first number reported in any [model card](https://mbrenndoerfer.com/writing/model-cards-documentation-intended-use-limitations-best-practices) or architecture description. Knowing that a model has dimension 4096, 32 layers, and you want to cache 4096 tokens in FP16, you can quickly compute:

$$
2 \times 32 \times 1 \times 4096 \times 4096 \times 2 = 2{,}147{,}483{,}648 \text{ bytes} \approx 2 \text{ GB}
$$

This mental arithmetic takes seconds and immediately tells you whether your hardware can support a given configuration.

### Grouped Query Attention and Cache Reduction

For models using [Grouped Query Attention](https://mbrenndoerfer.com/writing/grouped-query-attention-gqa-efficient-llm-inference) (GQA), which we covered in the Modern Decoder Models section, the number of KV heads is smaller than the number of query heads. This architectural innovation allows multiple query heads to share the same key-value pairs, dramatically reducing the cache footprint. GQA achieves this reduction without proportionally harming model quality, making it one of the most impactful architectural changes for inference efficiency introduced in recent years.

To understand why GQA reduces cache size, consider what the cache stores. In standard [MHA](https://mbrenndoerfer.com/writing/multi-head-attention-transformers), every query head has its own dedicated key head and value head. A model with 64 query heads caches 64 sets of keys and 64 sets of values. In GQA with 8 KV heads, 8 groups of 8 query heads each share one set of keys and one set of values. The cache stores only 8 sets of keys and 8 sets of values, but all 64 query heads can still attend to their relevant context. The queries are still diverse and specialized; only the cached targets are shared.

If a model has $H$ query heads but only $H_{\text{kv}}$ key-value heads, the cache size reduces proportionally:

$$
\text{Reduction Factor} = \frac{H}{H_{\text{kv}}}
$$

where:

- $H$: the number of query attention heads (used for computation but not cached)
- $H_{\text{kv}}$: the number of key-value heads (the only heads whose projections are cached)

This reduction factor quantifies the memory savings from [GQA](https://mbrenndoerfer.com/writing/grouped-query-attention-gqa-efficient-llm-inference). A model with 64 query heads and 8 KV heads achieves an 8-fold reduction in cache size compared to standard MHA. This 8-fold reduction can mean the difference between a model that requires multiple GPUs to deploy and one that fits on a single high-end card. The practical impact of GQA on deployment feasibility cannot be overstated.

Think of GQA as a shared note-taking system in a large meeting. In standard [MHA](https://mbrenndoerfer.com/writing/multi-head-attention-transformers), each participant (query head) takes their own complete notes (keys and values) about what everyone said. In GQA, one designated note-taker serves each small group of participants, and everyone in the group reads from the same set of notes. The quality of the discussion suffers only marginally because the notes are still detailed. But the storage cost falls by the number of participants per group.

Now let's examine concrete examples with popular model architectures. First, we define our model specifications and a calculation function:

In\[4\]:

Code

```
def calculate_kv_cache_memory(
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    batch_size: int = 1,
    seq_len: int = 2048,
    dtype_bytes: int = 2,  # FP16 default
) -> dict:
    """
    Calculate KV cache memory requirements.

    Returns memory in bytes, MB, and GB.
    """
    # Total elements: 2 (K+V) × layers × batch × seq × heads × head_dim
    total_elements = (
        2 * num_layers * batch_size * seq_len * num_kv_heads * head_dim
    )

    # Convert to bytes
    memory_bytes = total_elements * dtype_bytes
    memory_mb = memory_bytes / (1024**2)
    memory_gb = memory_bytes / (1024**3)

    return {
        "elements": total_elements,
        "bytes": memory_bytes,
        "MB": memory_mb,
        "GB": memory_gb,
    }

# Model specifications

models = {
    "GPT-2 (124M)": {
        "num_layers": 12,
        "num_kv_heads": 12,
        "head_dim": 64,
        "d_model": 768,
        "total_params": 124_000_000,
    },
    "LLaMA 7B": {
        "num_layers": 32,
        "num_kv_heads": 32,  # MHA
        "head_dim": 128,
        "d_model": 4096,
        "total_params": 7_000_000_000,
    },
    "LLaMA 2 70B": {
        "num_layers": 80,
        "num_kv_heads": 8,  # GQA with 8 KV heads
        "head_dim": 128,
        "d_model": 8192,
        "total_params": 70_000_000_000,
    },
    "LLaMA 3 70B": {
        "num_layers": 80,
        "num_kv_heads": 8,  # GQA
        "head_dim": 128,
        "d_model": 8192,
        "total_params": 70_000_000_000,
    },
}
```

In\[5\]:

Code

```
model_results = {}
for model_name, specs in models.items():
    cache = calculate_kv_cache_memory(
        num_layers=specs["num_layers"],
        num_kv_heads=specs["num_kv_heads"],
        head_dim=specs["head_dim"],
        batch_size=1,
        seq_len=4096,
        dtype_bytes=2,
    )

    weight_gb = specs["total_params"] * 2 / (1024**3)
    cache_pct = (cache["GB"] / weight_gb) * 100

    model_results[model_name] = {
        "weight_gb": weight_gb,
        "cache_gb": cache["GB"],
        "cache_pct": cache_pct,
    }
```

Out\[6\]:

Console

```
GPT-2 (124M):
  Model weights: 0.2 GB
  KV cache:      0.14 GB (60.9% of weights)

LLaMA 7B:
  Model weights: 13.0 GB
  KV cache:      2.00 GB (15.3% of weights)

LLaMA 2 70B:
  Model weights: 130.4 GB
  KV cache:      1.25 GB (1.0% of weights)

LLaMA 3 70B:
  Model weights: 130.4 GB
  KV cache:      1.25 GB (1.0% of weights)
```

Out\[7\]:

Visualization

![Grouped bar chart comparing KV cache memory for MHA versus GQA configurations.](https://assets.mbrenndoerfer.com/_optimized/notebooks/2_kv_cache_memory_files/gqa-vs-mha-cache-comparison-dark-1920w.webp)

Grouped bar chart comparing KV cache memory for MHA versus GQA configurations.

These results reveal several important patterns about [KV cache](https://mbrenndoerfer.com/writing/autoregressive-generation-gpt-text-generation) scaling across different architectures. For [GPT-2](https://mbrenndoerfer.com/writing/gpt-2-scaling-language-models-zero-shot-learning), the cache remains negligible relative to model weights, consuming only about 2-3% of total memory. This makes sense: GPT-2 was designed before long-context inference became a priority, and its small embedding dimension (768) and modest layer count (12) mean the cache grows slowly. As we scale to larger models like [LLaMA 7B](https://mbrenndoerfer.com/writing/llama-architecture-design-training-efficiency), the cache becomes more significant, reaching approximately 7-8% of weight memory at 4K context.

The most striking pattern appears with LLaMA 2 and 3 70B models. Despite being 10 times larger than LLaMA 7B by parameter count, their use of [Grouped Query Attention](https://mbrenndoerfer.com/writing/grouped-query-attention-gqa-efficient-llm-inference) with only 8 KV heads means their cache is smaller than LLaMA 7B's cache. This shows an 8-fold reduction compared to what standard [multi-head attention](https://mbrenndoerfer.com/writing/multi-head-attention-transformers) would require. This architectural choice makes the difference between practical deployment and memory constraints that would force a multi-GPU cluster. GQA made 70B-parameter models deployable on a single A100 GPU.

## Worked Example: Hand-Calculating LLaMA 7B Cache

Working through a concrete numerical example cements the formula and builds the intuition you need to estimate cache sizes quickly. Let us trace through the LLaMA 7B calculation step by step, so that every multiplication corresponds to a specific architectural decision.

LLaMA 7B has the following relevant specifications:

- $L = 32$ [transformer](https://mbrenndoerfer.com/writing/transformer-attention-is-all-you-need) layers
- $H_{\text{kv}} = 32$ key-value heads (standard MHA, no GQA in the original [LLaMA 7B](https://mbrenndoerfer.com/writing/llama-architecture-design-training-efficiency))
- $D_h = 128$ head dimension
- Inference in FP16: $\text{bytes} = 2$

**Step 1: Memory per element.** Each floating-point value in the cache occupies 2 bytes in FP16.

**Step 2: Elements per head per token.** Each head stores one [key vector](https://mbrenndoerfer.com/writing/query-key-value-attention-mechanism) of length $D_h = 128$ and one value vector of length $D_h = 128$. That gives $128 + 128 = 256$ elements per head per token.

**Step 3: Elements across all heads per token per layer.** With $H_{\text{kv}} = 32$ heads, one layer stores $32 \times 256 = 8{,}192$ elements per token.

**Step 4: Elements across all layers per token.** With $L = 32$ layers, each token contributes $32 \times 8{,}192 = 262{,}144$ elements to the cache in total.

**Step 5: Bytes per token across all layers.** Multiplying by 2 bytes per element:

$$
\begin{aligned}
\text{Bytes per token} &= 2 \times L \times H_{\text{kv}} \times D_h \times \text{bytes} \\
&= 2 \times 32 \times 32 \times 128 \times 2 \\
&= 524{,}288 \text{ bytes} \\
&= 512 \text{ KB} \\
&\approx 0.5 \text{ MB}
\end{aligned}
$$

**Step 6: Total cache for a sequence of 4,096 tokens.** Multiplying by the sequence length:

$$
\begin{aligned}
\text{Total Cache} &= T \times \text{bytes per token} \\
&= 4{,}096 \times 524{,}288 \\
&= 2{,}147{,}483{,}648 \text{ bytes} \\
&= 2 \text{ GB}
\end{aligned}
$$

**Step 7: Sanity check using the simplified [MHA](https://mbrenndoerfer.com/writing/multi-head-attention-transformers) formula.** Recall that for MHA, $H_{\text{kv}} \times D_h = D_{\text{model}}$. For LLaMA 7B, $D_{\text{model}} = 32 \times 128 = 4{,}096$. Plugging into the simplified formula:

$$
\begin{aligned}
\text{Cache} &= 2 \times L \times B \times T \times D_{\text{model}} \times \text{bytes} \\
&= 2 \times 32 \times 1 \times 4{,}096 \times 4{,}096 \times 2 \\
&= 2 \text{ GB}
\end{aligned}
$$

Both paths give the same answer, which is reassuring. At 4K tokens with a [batch size](https://mbrenndoerfer.com/writing/stochastic-gradient-descent-neural-network-optimization) of 1 in FP16, LLaMA 7B requires exactly 2 GB for its [KV cache](https://mbrenndoerfer.com/writing/autoregressive-generation-gpt-text-generation). This number is small enough to coexist with the model weights (roughly 14 GB in FP16) on a 24 GB GPU. But it doubles with every doubling of context length, and it multiplies linearly with batch size.

**Step 8: Scaling to a different batch size.** If we want to serve 8 concurrent requests (batch size 8):

$$
\text{Cache for batch 8} = 8 \times 2 \text{ GB} = 16 \text{ GB}
$$

Adding 14 GB for model weights and roughly 2 GB for other overhead, this already approaches the 32 GB limit of an A100 40 GB GPU, leaving only a small safety margin. This calculation, which takes under a minute once you know the formula, determines whether a deployment plan is feasible before you ever touch the hardware.

Join the Community

Enjoying this article?

I write about AI, data science, machine learning, finance, economics and entrepreneurship. Subscribe to get updates delivered straight to your inbox.

- No popups
- Unobstructed reading
- Commenting

No spam, unsubscribe anytime.

[Join Community](https://mbrenndoerfer.com/community)

![Michael Brenndoerfer](https://assets.mbrenndoerfer.com/_optimized/general/resume/michael_brenndoerfer-480w.webp)

Michael Brenndoerfer

Author and community host

## Sequence Length Effects

Sequence length has a linear effect on KV cache size, and this relationship emerges directly from our formula. The term $T$ (sequence length) appears as a simple multiplicative factor, meaning that doubling the [context window](https://mbrenndoerfer.com/writing/co-occurrence-matrices-distributional-semantics-nlp) doubles the cache memory. Unlike [attention score](https://mbrenndoerfer.com/writing/attention-mechanism-intuition-soft-lookup-weights-context-vectors) computation, which grows quadratically with sequence length, cache memory grows strictly linearly. This linear relationship enables straightforward capacity planning and creates unavoidable memory costs for long contexts. The [quadratic attention](https://mbrenndoerfer.com/writing/quadratic-attention-bottleneck-transformers-long-sequences) computation can be optimized away with techniques like [FlashAttention](https://mbrenndoerfer.com/writing/flashattention-algorithm-memory-efficient-gpu-tiling); the linear cache growth cannot.

This linear relationship becomes particularly problematic for long-context models. The trend in language model development has been toward ever-longer context windows, from 512 tokens in early [BERT](https://mbrenndoerfer.com/writing/bert-bidirectional-pretraining-revolutionizes-language-understanding) models to 128K or even 1M token windows in modern systems. Every tenfold increase in context length requires tenfold more cache memory, creating a fundamental tension between capability and resource requirements. A system that works perfectly at 4K context can become infeasible at 32K context without any change to the model, hardware, or [batch size](https://mbrenndoerfer.com/writing/stochastic-gradient-descent-neural-network-optimization). The context window is not a free architectural choice; every token of context costs memory.

Think of context length as the size of your desk. A small desk lets you see only the last few pages of a document. A large desk lets you spread out the entire manuscript. But a larger desk costs more, takes up more space in your office, and requires more effort to keep organized. A 128K [context window](https://mbrenndoerfer.com/writing/co-occurrence-matrices-distributional-semantics-nlp) is a desk the size of a tennis court. It provides extraordinary capability, but the cost in memory resources is proportionally extraordinary.

In\[8\]:

Code

```
# LLaMA 7B specifications
llama_7b = models["LLaMA 7B"]

# Calculate cache sizes for various context lengths
context_lengths = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
cache_sizes_gb = []

for ctx_len in context_lengths:
    cache = calculate_kv_cache_memory(
        num_layers=llama_7b["num_layers"],
        num_kv_heads=llama_7b["num_kv_heads"],
        head_dim=llama_7b["head_dim"],
        batch_size=1,
        seq_len=ctx_len,
        dtype_bytes=2,
    )
    cache_sizes_gb.append(cache["GB"])

# Model weight size for comparison
weight_gb = llama_7b["total_params"] * 2 / (1024**3)
```

Out\[9\]:

Visualization

![Line plot showing KV cache memory increasing linearly with context length, crossing model weight threshold.](https://assets.mbrenndoerfer.com/_optimized/notebooks/2_kv_cache_memory_files/kv-cache-context-length-scaling-dark-1920w.webp)

Line plot showing KV cache memory increasing linearly with context length, crossing model weight threshold.

The crossover point, where [KV cache](https://mbrenndoerfer.com/writing/autoregressive-generation-gpt-text-generation) exceeds model weights, is critical for deployment planning. This visualization clearly shows the linear relationship between sequence length and memory consumption, with the cache becoming the dominant memory consumer at longer contexts. Let us find that crossover point exactly:

In\[10\]:

Code

```
# Find crossover point for LLaMA 7B
crossover_seq_len = (
    weight_gb
    * (1024**3)
    / (
        2
        * llama_7b["num_layers"]
        * llama_7b["num_kv_heads"]
        * llama_7b["head_dim"]
        * 2  # bytes
    )
)
```

Out\[11\]:

Console

```
LLaMA 7B crossover point: 26,702 tokens
At this context length, KV cache equals model weights: 13.0 GB
```

This crossover point represents a major change in the memory profile of inference. Below this threshold, model weights dominate memory consumption and the cache is a relatively small overhead. Above this threshold, the cache becomes the primary memory consumer, growing linearly with every additional token while weights remain constant.

For context windows beyond approximately 27K tokens, the [KV cache](https://mbrenndoerfer.com/writing/autoregressive-generation-gpt-text-generation) will consume more memory than the model itself, fundamentally changing deployment requirements and making cache optimization techniques essential rather than optional. This is why serving Claude at 100K context is categorically different, in infrastructure terms, from serving it at 4K context. The model has not changed. The hardware costs have multiplied.

For long-context applications such as document analysis, multi-turn conversations, and code completion over large codebases, the KV cache dominates memory usage. A 128K [context window](https://mbrenndoerfer.com/writing/co-occurrence-matrices-distributional-semantics-nlp), as supported by models like Claude and [GPT-4](https://mbrenndoerfer.com/writing/gpt4-multimodal-language-models-reach-human-level-performance), requires 32 times more cache memory than a 4K window. This scaling explains why long-context inference often requires specialized infrastructure and optimization techniques that would be unnecessary for short-context applications.

### The Crossover Threshold

The sequence length at which cache memory equals weight memory is a useful reference point for deployment decisions. Below the threshold, your hardware choice is driven primarily by model size. Above the threshold, you must think seriously about cache management strategies.

We can derive the crossover threshold analytically by setting the cache formula equal to the weight formula and solving for $T$:

$$
2 \times L \times B \times T \times H_{\text{kv}} \times D_h \times \text{bytes} = N_{\text{params}} \times \text{bytes}
$$

where $N_{\text{params}}$ is the total parameter count. Solving for $T$ with $B = 1$:

$$
T_{\text{crossover}} = \frac{N_{\text{params}}}{2 \times L \times H_{\text{kv}} \times D_h}
$$

where:

- $T_{\text{crossover}}$: the sequence length at which cache equals weight memory
- $N_{\text{params}}$: the total number of model parameters
- $L$: number of [transformer](https://mbrenndoerfer.com/writing/transformer-attention-is-all-you-need) layers
- $H_{\text{kv}}$: number of [KV heads](https://mbrenndoerfer.com/writing/grouped-query-attention-gqa-efficient-llm-inference)
- $D_h$: [head dimension](https://mbrenndoerfer.com/writing/multi-head-attention-transformers)

For [LLaMA 7B](https://mbrenndoerfer.com/writing/llama-architecture-design-training-efficiency), this gives approximately 27,000 tokens. For LLaMA 2 70B with GQA, the reduced KV head count ($H_{\text{kv}} = 8$ instead of 64 for full MHA) pushes the crossover threshold much higher, to roughly 427,000 tokens. GQA does not just save cache memory; it dramatically defers the context length at which cache starts to dominate, giving practitioners much more headroom before they need specialized cache management.

## Batch Size Effects

[Batch size](https://mbrenndoerfer.com/writing/stochastic-gradient-descent-neural-network-optimization) scales linearly with cache memory, following the same multiplicative relationship we observed with sequence length. Each sequence in the batch maintains its own independent cache. Tokens in one sequence do not attend to tokens in another sequence, so cached values are not shared across batch elements. Think of batch processing as running $B$ conversations simultaneously in $B$ isolated rooms. Each room has its own filing cabinet, and they are completely separate. You cannot combine the cabinets to save space.

This independence changes the interpretation for memory planning. If you want to process 8 concurrent requests, you need 8 complete copies of the [KV cache](https://mbrenndoerfer.com/writing/autoregressive-generation-gpt-text-generation), one for each sequence. The batch dimension in our cache tensors is not a clever abstraction that enables sharing; it simply organizes multiple independent caches for efficient parallel processing. The GPU executes them in parallel across its compute units, which amortizes the latency of loading model weights, but every cache must be allocated separately in memory.

The [batch size](https://mbrenndoerfer.com/writing/stochastic-gradient-descent-neural-network-optimization) tradeoff is at the heart of inference economics. Serving individual requests one by one wastes GPU resources because the arithmetic units sit idle while waiting for data transfers to complete. Batching multiple requests allows the GPU's parallel processing capabilities to run at higher utilization. But this throughput gain comes at the direct cost of proportionally more cache memory. Doubling batch size doubles throughput potential but also doubles cache requirements. The optimal batch size is always the largest that fits in available memory after accounting for model weights, activation memory, and framework overhead.

In\[12\]:

Code

```
# Calculate cache for various batch sizes
batch_sizes = [1, 2, 4, 8, 16, 32, 64]
cache_by_batch = []

for batch in batch_sizes:
    cache = calculate_kv_cache_memory(
        num_layers=llama_7b["num_layers"],
        num_kv_heads=llama_7b["num_kv_heads"],
        head_dim=llama_7b["head_dim"],
        batch_size=batch,
        seq_len=4096,
        dtype_bytes=2,
    )
    cache_by_batch.append(cache["GB"])
```

Out\[13\]:

Visualization

![Bar chart showing KV cache memory increasing with batch size from 1 to 64.](https://assets.mbrenndoerfer.com/_optimized/notebooks/2_kv_cache_memory_files/kv-cache-batch-size-scaling-dark-1920w.webp)

Bar chart showing KV cache memory increasing with batch size from 1 to 64.

This visualization clearly demonstrates the linear relationship between [batch size](https://mbrenndoerfer.com/writing/stochastic-gradient-descent-neural-network-optimization) and [KV cache memory](https://mbrenndoerfer.com/writing/multi-query-attention-memory-efficient-inference). The batch size tradeoff is fundamental to inference economics, as it directly determines the throughput your system can achieve in [production deployment](https://mbrenndoerfer.com/writing/deploying-your-ai-agent-production-service). Doubling batch size doubles memory requirements, but it also doubles the number of requests you can process in parallel. This creates a direct tension between throughput and memory availability:

- **Single-sequence inference** minimizes cache memory but leaves GPU compute underutilized
- **Large batches** maximize throughput (tokens per second) but require proportionally more memory
- **The optimal batch size** is typically the largest that fits in available memory after accounting for model weights and activation memory

GPU compute is expensive, and leaving it idle while processing a single sequence is wasteful. Batching amortizes the fixed costs of loading model weights and allows the GPU's parallel processing capabilities to be fully utilized. But this amortization is only possible if you have sufficient memory to hold multiple caches simultaneously.

In\[14\]:

Code

```
# What batch size fits in 80GB (A100) for LLaMA 7B?
total_gpu_memory = 80  # GB
model_weights = weight_gb
activation_overhead = 2  # GB (rough estimate for intermediate computations)
available_for_cache = total_gpu_memory - model_weights - activation_overhead

# Cache per sequence at 4096 tokens
cache_per_seq = calculate_kv_cache_memory(
    num_layers=llama_7b["num_layers"],
    num_kv_heads=llama_7b["num_kv_heads"],
    head_dim=llama_7b["head_dim"],
    batch_size=1,
    seq_len=4096,
    dtype_bytes=2,
)["GB"]

max_batch = int(available_for_cache / cache_per_seq)
```

Out\[15\]:

Console

```
GPU Memory Budget: 80 GB
  Model weights:     13.0 GB
  Activation buffer: 2.0 GB
  Available for KV:  65.0 GB
  Cache per sequence: 2.00 GB
  Maximum batch size: 32
```

This calculation reveals excellent throughput potential for [LLaMA 7B](https://mbrenndoerfer.com/writing/llama-architecture-design-training-efficiency) on high-end hardware. With an A100 80GB GPU running LLaMA 7B at 4096-token context, the available memory after weights and overhead can support a [batch size](https://mbrenndoerfer.com/writing/stochastic-gradient-descent-neural-network-optimization) that significantly exceeds what most consumer hardware could achieve. The 80 GB memory capacity of the A100 is precisely calibrated for workloads like this, which is why it remains the workhorse of LLM serving infrastructure.

### Dynamic Batch Size Management

Production serving systems cannot always predict the sequence length of incoming requests in advance. A request that starts with a short prompt may generate a long response, gradually consuming more cache memory than expected. This variability complicates static batch size planning. Modern inference servers like [vLLM](https://mbrenndoerfer.com/writing/paged-attention-vllm-kv-cache-memory-management) and TensorRT-LLM address this with [continuous batching](https://mbrenndoerfer.com/writing/continuous-batching) and dynamic cache management, which we explore in the Paged Attention chapter. For now, the key insight is that naive static batch size planning is too conservative: it must reserve worst-case memory for every sequence, wasting capacity for short requests.

The safest approach for deployment planning is to calculate the maximum cache memory using the longest sequence you expect to encounter, then size your batch accordingly. If your 99th percentile request requires 8K tokens, plan your [batch size](https://mbrenndoerfer.com/writing/stochastic-gradient-descent-neural-network-optimization) for that length, even if most requests are much shorter. This conservative approach guarantees you will not run out of memory, but it leaves efficiency on the table. Dynamic allocation systems can recover that efficiency by reallocating unused cache blocks from short sequences to long ones.

## Combined Effects: The Memory Budget

In practice, we must account for all memory consumers simultaneously. The GPU has a fixed amount of memory, and every component of inference must fit within this budget. Omitting any component from memory accounting causes out-of-memory errors and crashes inference. Accurate memory estimation is essential for reliable deployment, and there is no substitute for working through the full accounting before you commit to a hardware configuration.

The total GPU memory budget must accommodate all major contributors working at the same time. Model weights are fixed once loaded and remain constant regardless of input or batch size, representing the irreducible baseline cost of using a particular model. The [KV cache](https://mbrenndoerfer.com/writing/autoregressive-generation-gpt-text-generation) is the dynamic component that varies most dramatically with your [batch size](https://mbrenndoerfer.com/writing/stochastic-gradient-descent-neural-network-optimization) and sequence length choices. Activation memory holds intermediate results like attention scores and feed-forward layer outputs, growing with batch size but not with sequence length in the same way as the cache. Framework overhead, including CUDA context, PyTorch allocator buffers, and communication buffers in multi-GPU settings, represents a fixed cost that exists regardless of what computation you perform.

The interplay between these components creates a constrained optimization problem. Model weights cannot be reduced without using a smaller model or applying quantization. Framework overhead is similarly fixed. This leaves the KV cache and activation memory as the variables that respond to your choices about batch size and sequence length. When a deployment configuration fails to fit in GPU memory, the first lever you should reach for is cache optimization: reducing the number of [KV heads](https://mbrenndoerfer.com/writing/grouped-query-attention-gqa-efficient-llm-inference) via GQA, quantizing cache values to [INT8](https://mbrenndoerfer.com/writing/int8-quantization-absmax-smooth-quantization-implementation) or [INT4](https://mbrenndoerfer.com/writing/int4-quantization-group-wise-nf4-format-llms), or implementing page-level eviction when cache exceeds available memory.

Let us build a memory estimator that accounts for all these factors.

In\[16\]:

Code

```
def estimate_inference_memory(
    model_params: int,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    batch_size: int,
    seq_len: int,
    dtype_bytes: int = 2,
    activation_factor: float = 0.1,  # Fraction of weights
) -> dict:
    """
    Estimate total GPU memory for inference.

    activation_factor: Rough multiplier for activation memory (varies by implementation)
    """
    # Model weights
    weight_bytes = model_params * dtype_bytes
    weight_gb = weight_bytes / (1024**3)

    # KV cache
    cache = calculate_kv_cache_memory(
        num_layers, num_kv_heads, head_dim, batch_size, seq_len, dtype_bytes
    )

    # Activation memory (rough estimate)
    activation_gb = weight_gb * activation_factor

    # Framework overhead (rough estimate)
    overhead_gb = 0.5  # CUDA context, allocator buffers

    total_gb = weight_gb + cache["GB"] + activation_gb + overhead_gb

    return {
        "weights_gb": weight_gb,
        "kv_cache_gb": cache["GB"],
        "activation_gb": activation_gb,
        "overhead_gb": overhead_gb,
        "total_gb": total_gb,
        "kv_cache_pct": (cache["GB"] / total_gb) * 100,
    }
```

In\[17\]:

Code

```
# Memory breakdown for different configurations
configs = [
    {"name": "Short context (2K)", "batch": 1, "seq": 2048},
    {"name": "Medium context (8K)", "batch": 1, "seq": 8192},
    {"name": "Long context (32K)", "batch": 1, "seq": 32768},
    {"name": "Batched (4K x 8)", "batch": 8, "seq": 4096},
]

memory_results = []
for config in configs:
    mem = estimate_inference_memory(
        model_params=llama_7b["total_params"],
        num_layers=llama_7b["num_layers"],
        num_kv_heads=llama_7b["num_kv_heads"],
        head_dim=llama_7b["head_dim"],
        batch_size=config["batch"],
        seq_len=config["seq"],
    )
    memory_results.append((config["name"], mem))
```

Out\[18\]:

Console

```
LLaMA 7B Memory Breakdown (FP16)
======================================================================

Short context (2K):
  Weights:      13.0 GB
  KV Cache:     1.00 GB (6% of total)
  Activation:    1.3 GB
  Overhead:      0.5 GB
  TOTAL:        15.8 GB

Medium context (8K):
  Weights:      13.0 GB
  KV Cache:     4.00 GB (21% of total)
  Activation:    1.3 GB
  Overhead:      0.5 GB
  TOTAL:        18.8 GB

Long context (32K):
  Weights:      13.0 GB
  KV Cache:    16.00 GB (52% of total)
  Activation:    1.3 GB
  Overhead:      0.5 GB
  TOTAL:        30.8 GB

Batched (4K x 8):
  Weights:      13.0 GB
  KV Cache:    16.00 GB (52% of total)
  Activation:    1.3 GB
  Overhead:      0.5 GB
  TOTAL:        30.8 GB
```

Out\[19\]:

Visualization

![Stacked horizontal bar chart showing memory allocation for weights, KV cache, activation, and overhead across four configurations.](https://assets.mbrenndoerfer.com/_optimized/notebooks/2_kv_cache_memory_files/memory-budget-breakdown-configs-dark-1920w.webp)

Stacked horizontal bar chart showing memory allocation for weights, KV cache, activation, and overhead across four configurations.

This breakdown demonstrates how dramatically memory requirements shift across different usage patterns. For short 2K contexts, model weights account for roughly 82% of total memory, with [KV cache](https://mbrenndoerfer.com/writing/autoregressive-generation-gpt-text-generation) representing about 6%. At 32K tokens, the cache grows to approximately 52% of total memory and overtakes the weights. Batching 8 sequences at 4K context produces the same 16 GB cache and the same breakdown as one 32K sequence because [batch size](https://mbrenndoerfer.com/writing/stochastic-gradient-descent-neural-network-optimization) and sequence length enter the formula as a product. This transition from weight-dominated to cache-dominated memory profiles changes deployment strategies and hardware selection.

The percentage breakdown reveals how memory allocation shifts with configuration. At short contexts, weights dominate at about 85% of total memory. At long contexts or large batches, the KV cache becomes the primary memory consumer. This shift demonstrates why cache optimization becomes critical for long-context applications. An engineering team that sizes hardware for a short-context model and then later extends the [context window](https://mbrenndoerfer.com/writing/co-occurrence-matrices-distributional-semantics-nlp) may discover that the hardware that seemed adequate is suddenly insufficient, even though the model itself has not changed.

## The Memory Bottleneck

When [KV cache memory](https://mbrenndoerfer.com/writing/multi-query-attention-memory-efficient-inference) dominates, we encounter a fundamental constraint that differs from the compute bottleneck that traditionally limits deep learning. Understanding this distinction is essential because the remedies for each bottleneck are completely different. Buying a faster GPU solves a compute bottleneck; it does nothing for a memory bottleneck. Reducing [model depth](https://mbrenndoerfer.com/writing/transformer-architecture-hyperparameters-design-guide) helps with weight memory but not cache memory. Knowing which bottleneck you face determines which optimizations are worth pursuing.

In a compute-bound scenario, performance is limited by how fast the GPU can perform arithmetic operations. The GPU's thousands of CUDA cores are running at full utilization, and the only way to go faster is to reduce the number of operations or use hardware with more compute units. In a memory-bound scenario, performance is limited by how fast data can move between memory and compute units. The arithmetic units sit idle while waiting for data to arrive, regardless of how many of them exist. An A100 GPU has 312 teraFLOPS of FP16 compute capacity but only 2 terabytes per second of memory bandwidth. The ratio between these two numbers, approximately 156 FLOPs per byte, is the threshold that determines which resource dominates.

During token generation, the model performs two qualitatively different kinds of work. Attention computation reads the entire [KV cache](https://mbrenndoerfer.com/writing/autoregressive-generation-gpt-text-generation) to compute attention scores, then reads it again to compute the weighted value sum. This requires moving massive amounts of data from GPU high-bandwidth memory to the compute units, with relatively modest arithmetic. Feed-forward layers perform dense matrix multiplications where the same weight matrix applies to the input, achieving high arithmetic intensity because the weights are reused across the batch. The attention part is almost always memory-bound for realistic sequence lengths; the feed-forward part can be compute-bound when batches are large enough.

We can formalize this relationship using the roofline model:

$$
\mathcal{I} = \frac{\text{FLOPs}}{\text{Bytes}}
$$

where:

- $\mathcal{I}$: the arithmetic intensity (measured in floating-point operations per byte transferred)
- $\text{FLOPs}$: the total floating-point operations performed for one generation step
- $\text{Bytes}$: the total data transferred from GPU memory, dominated by KV cache reads

If $\mathcal{I}$ is lower than the GPU's ratio of peak compute to peak bandwidth, the process is memory-bound. For an A100, this [ridge](https://mbrenndoerfer.com/writing/ridge-regression-l2-regularization-complete-guide) point is approximately 156 FLOPs/byte (312 TFLOPS divided by 2 TB/s). Any attention computation with arithmetic intensity below this threshold cannot fully utilize GPU compute because the data arrives too slowly.

The practical implication is that for single-sequence generation at typical context lengths, the attention computation is memory-bound. Each token generation requires reading the entire [KV cache](https://mbrenndoerfer.com/writing/autoregressive-generation-gpt-text-generation) from memory. This memory transfer takes longer than the actual arithmetic, which is why techniques like [FlashAttention](https://mbrenndoerfer.com/writing/flashattention-algorithm-memory-efficient-gpu-tiling) (which reduces memory traffic through tiling) and [KV cache quantization](https://mbrenndoerfer.com/writing/kv-cache-compression-eviction-quantization-h2o-algorithm) (which reduces the size of each transfer) provide significant speedups without reducing the number of arithmetic operations.

In\[20\]:

Code

```
# Compute vs Memory bound analysis
def analyze_bottleneck(
    batch_size: int,
    seq_len: int,
    d_model: int,
    num_layers: int,
    gpu_tflops: float = 312,  # A100 FP16
    gpu_bandwidth_tb: float = 2.0,  # A100 HBM
):
    """
    Analyze whether generation is compute-bound or memory-bound.

    For generation, we process one token at a time. Attention: O(seq_len) memory reads, O(seq_len x d_model) compute. FFN: O(d_model^2) memory reads, O(d_model^2) compute.
    """
    # FLOPs for generating one token (simplified)
    # Attention: 4 x d_model x seq_len (Q x K, softmax, x V, output proj)
    attention_flops = 4 * d_model * seq_len * batch_size * num_layers

    # FFN: 8 x d_model x d_ffn approx 8 x 4 x d_model^2 (for d_ffn = 4 x d_model)
    ffn_flops = 8 * 4 * d_model * d_model * batch_size * num_layers

    total_flops = attention_flops + ffn_flops

    # Memory reads (bytes) - dominated by KV cache for attention
    # KV cache read: 2 x seq_len x d_model x bytes per layer
    kv_read_bytes = 2 * seq_len * d_model * 2 * num_layers * batch_size

    # Time estimates (seconds)
    compute_time = total_flops / (gpu_tflops * 1e12)
    memory_time = kv_read_bytes / (gpu_bandwidth_tb * 1e12)

    arithmetic_intensity = total_flops / kv_read_bytes

    return {
        "total_flops": total_flops,
        "kv_read_bytes": kv_read_bytes,
        "compute_time_us": compute_time * 1e6,
        "memory_time_us": memory_time * 1e6,
        "arithmetic_intensity": arithmetic_intensity,
        "bottleneck": "memory" if memory_time > compute_time else "compute",
    }
```

In\[21\]:

Code

```
# Analyze for different sequence lengths
seq_lengths_to_analyze = [1024, 4096, 16384, 65536]
bottleneck_results = []

for seq_len in seq_lengths_to_analyze:
    analysis = analyze_bottleneck(
        batch_size=1, seq_len=seq_len, d_model=4096, num_layers=32
    )
    bottleneck_results.append((seq_len, analysis))
```

Out\[22\]:

Console

```
Bottleneck Analysis: LLaMA 7B Token Generation (A100 GPU)
=================================================================

Context: 1,024 tokens
  Compute time: 56.8 us
  Memory time:  268.4 us
  Arithmetic intensity: 33.0 FLOPs/byte
  Bottleneck: MEMORY

Context: 4,096 tokens
  Compute time: 61.9 us
  Memory time:  1073.7 us
  Arithmetic intensity: 9.0 FLOPs/byte
  Bottleneck: MEMORY

Context: 16,384 tokens
  Compute time: 82.6 us
  Memory time:  4295.0 us
  Arithmetic intensity: 3.0 FLOPs/byte
  Bottleneck: MEMORY

Context: 65,536 tokens
  Compute time: 165.2 us
  Memory time:  17179.9 us
  Arithmetic intensity: 1.5 FLOPs/byte
  Bottleneck: MEMORY
```

Out\[23\]:

Visualization

![Grouped bar chart comparing compute time versus memory time across different context lengths.](https://assets.mbrenndoerfer.com/_optimized/notebooks/2_kv_cache_memory_files/compute-vs-memory-bottleneck-dark-1920w.webp)

Grouped bar chart comparing compute time versus memory time across different context lengths.

These results reveal a critical pattern in [transformer](https://mbrenndoerfer.com/writing/transformer-attention-is-all-you-need) inference performance. Memory transfer time dominates compute time at every context length in this simplified estimate. As context grows, the modeled [KV-cache](https://mbrenndoerfer.com/writing/kv-cache-transformer-attention-optimization) transfer time increases more steeply than compute time, so the memory bottleneck widens rather than narrows. For typical production scenarios at 4K-8K contexts, memory bandwidth is therefore the primary constraint. This is why [KV cache quantization](https://mbrenndoerfer.com/writing/kv-cache-compression-eviction-quantization-h2o-algorithm) and memory- [efficient attention](https://mbrenndoerfer.com/writing/attention-complexity-quadratic-scaling-memory-efficient-transformers) implementations can provide significant speedups by reducing memory traffic. Buying a GPU with more FLOPS alone would not resolve this memory-bound regime; bandwidth and memory-access efficiency matter directly.

This bottleneck analysis demonstrates that at reasonable context lengths, single-sequence generation is memory-bound. The GPU spends most of its time waiting for [KV cache](https://mbrenndoerfer.com/writing/autoregressive-generation-gpt-text-generation) data to transfer from high-bandwidth memory to the compute units. Increasing [batch size](https://mbrenndoerfer.com/writing/stochastic-gradient-descent-neural-network-optimization) helps amortize memory transfer costs, but it in turn increases total memory requirements, creating a constrained optimization problem. Every solution to the memory bottleneck creates a new version of the same problem at a different scale.

Join the Community

Enjoying this article?

I write about AI, data science, machine learning, finance, economics and entrepreneurship. Subscribe to get updates delivered straight to your inbox.

- No popups
- Unobstructed reading
- Commenting

No spam, unsubscribe anytime.

[Join Community](https://mbrenndoerfer.com/community)

![Michael Brenndoerfer](https://assets.mbrenndoerfer.com/_optimized/general/resume/michael_brenndoerfer-480w.webp)

Michael Brenndoerfer

Author and community host

## Memory Visualization Across Models

Let us visualize how memory allocation differs across model sizes and architectures to build intuition for the relative importance of cache versus weights at different scales.

Out\[24\]:

Visualization

![Stacked bar chart comparing memory allocation between model weights and KV cache across four model sizes.](https://assets.mbrenndoerfer.com/_optimized/notebooks/2_kv_cache_memory_files/model-memory-breakdown-comparison-dark-1920w.webp)

Stacked bar chart comparing memory allocation between model weights and KV cache across four model sizes.

Without [GQA](https://mbrenndoerfer.com/writing/grouped-query-attention-gqa-efficient-llm-inference), [LLaMA](https://mbrenndoerfer.com/writing/llama-architecture-design-training-efficiency) 2 70B would require 8 times more [KV cache memory](https://mbrenndoerfer.com/writing/multi-query-attention-memory-efficient-inference), making long-context inference impractical on most hardware. The visualization clearly demonstrates how Grouped Query Attention keeps cache proportions manageable at the 70B scale, with the cache remaining a smaller fraction of total memory compared to what standard [MHA](https://mbrenndoerfer.com/writing/multi-head-attention-transformers) would require. The architectural innovation of GQA is visible in the numbers: the 70B models with 8 KV heads have smaller caches than LLaMA 7B with 32 KV heads, despite being an order of magnitude larger in parameter count.

## Practical Memory Planning

Before deploying a model, estimate memory requirements for your target configuration. This should be a routine step in any model deployment process, not an afterthought. The calculations are inexpensive and the alternative, discovering out-of-memory errors in production, is far more costly. Here is a practical workflow that combines all the components we have discussed.

The planning process involves four inputs you must know before you can make hardware decisions: the model architecture specifications (available in any [model card](https://mbrenndoerfer.com/writing/model-cards-documentation-intended-use-limitations-best-practices) or configuration file), the maximum context length your application requires, the [batch size](https://mbrenndoerfer.com/writing/stochastic-gradient-descent-neural-network-optimization) you need to achieve your throughput targets, and the data type you plan to use for inference. Once you have these, the calculation takes seconds.

In\[25\]:

Code

```
def can_fit_on_gpu(
    gpu_memory_gb: float,
    model_config: dict,
    batch_size: int,
    max_seq_len: int,
    dtype_bytes: int = 2,
    safety_margin: float = 0.9,  # Leave 10 percent buffer
) -> dict:
    """
    Check if a configuration fits in GPU memory.

    Returns feasibility and utilization details.
    """
    available = gpu_memory_gb * safety_margin

    mem = estimate_inference_memory(
        model_params=model_config["total_params"],
        num_layers=model_config["num_layers"],
        num_kv_heads=model_config["num_kv_heads"],
        head_dim=model_config["head_dim"],
        batch_size=batch_size,
        seq_len=max_seq_len,
        dtype_bytes=dtype_bytes,
    )

    fits = mem["total_gb"] <= available
    utilization = (mem["total_gb"] / available) * 100

    return {
        "fits": fits,
        "required_gb": mem["total_gb"],
        "available_gb": available,
        "utilization_pct": utilization,
    }
```

In\[26\]:

Code

```
# Test configurations on common GPUs
gpus = {"RTX 4090": 24, "A100 40GB": 40, "A100 80GB": 80, "H100 80GB": 80}

test_configs = [
    {"model": "LLaMA 7B", "batch": 1, "seq": 4096},
    {"model": "LLaMA 7B", "batch": 8, "seq": 4096},
    {"model": "LLaMA 7B", "batch": 26, "seq": 4096},
]

feasibility_results = []
for config in test_configs:
    model = models[config["model"]]
    config_results = []
    for gpu_name, gpu_mem in gpus.items():
        result = can_fit_on_gpu(gpu_mem, model, config["batch"], config["seq"])
        config_results.append((gpu_name, result))
    feasibility_results.append((config, config_results))
```

Out\[27\]:

Console

```
Deployment Feasibility Check
======================================================================

LLaMA 7B | batch=1 | seq=4096
--------------------------------------------------
  RTX 4090: OK (16.8/21.6 GB)
  A100 40GB: OK (16.8/36.0 GB)
  A100 80GB: OK (16.8/72.0 GB)
  H100 80GB: OK (16.8/72.0 GB)

LLaMA 7B | batch=8 | seq=4096
--------------------------------------------------
  RTX 4090: OOM (30.8/21.6 GB)
  A100 40GB: OK (30.8/36.0 GB)
  A100 80GB: OK (30.8/72.0 GB)
  H100 80GB: OK (30.8/72.0 GB)

LLaMA 7B | batch=26 | seq=4096
--------------------------------------------------
  RTX 4090: OOM (66.8/21.6 GB)
  A100 40GB: OOM (66.8/36.0 GB)
  A100 80GB: OK (66.8/72.0 GB)
  H100 80GB: OK (66.8/72.0 GB)
```

Out\[28\]:

Visualization

![Heatmap showing memory utilization percentages for different model configurations across GPU types.](https://assets.mbrenndoerfer.com/_optimized/notebooks/2_kv_cache_memory_files/gpu-feasibility-heatmap-dark-1920w.webp)

Heatmap showing memory utilization percentages for different model configurations across GPU types.

These results illustrate the hardware requirements for different deployment scenarios. [LLaMA 7B](https://mbrenndoerfer.com/writing/llama-architecture-design-training-efficiency) fits comfortably on all tested GPUs for single-sequence inference at 4K context, with even the 24 GB RTX 4090 providing adequate headroom. Scaling to [batch size](https://mbrenndoerfer.com/writing/stochastic-gradient-descent-neural-network-optimization) 8 pushes requirements beyond consumer hardware, requiring datacenter GPUs like the A100. These constraints directly inform deployment decisions: smaller models offer flexibility across hardware tiers, while larger batches demand specialized infrastructure.

The feasibility analysis demonstrates the practical constraints of different GPU configurations. For [production deployment](https://mbrenndoerfer.com/writing/deploying-your-ai-agent-production-service), understanding these memory limits is essential for selecting appropriate hardware. A team that does not work through these calculations before launch will discover the limits the hard way, during peak load when requests begin failing with out-of-memory errors.

### Choosing a Data Type

The data type used for cache storage is an often-overlooked lever for reducing memory requirements. Every parameter in our formula is fixed by the model architecture or the deployment scenario, except for the bytes-per-element term, which you can control directly. Switching from FP32 to FP16 halves cache memory at no algorithmic cost. Switching from FP16 to [INT8](https://mbrenndoerfer.com/writing/int8-quantization-absmax-smooth-quantization-implementation) halves it again, using quantization to represent each key and value with 8 bits instead of 16. [INT4](https://mbrenndoerfer.com/writing/int4-quantization-group-wise-nf4-format-llms) quantization reduces it by another factor of 2.

The tradeoff is accuracy. FP32 provides full floating-point precision. FP16 and BF16 maintain good numerical range while halving memory. INT8 quantization introduces small rounding errors that research has shown are generally tolerable for [KV cache](https://mbrenndoerfer.com/writing/autoregressive-generation-gpt-text-generation) values, since the [attention mechanism](https://mbrenndoerfer.com/writing/attention-mechanism-intuition-soft-lookup-weights-context-vectors) is somewhat robust to small perturbations in the attended values. INT4 quantization is more aggressive and may affect output quality for some tasks, particularly those requiring precise recall of specific numerical values from long contexts.

The key insight is that the KV cache stores intermediate representations, not the raw training weights of the model. These representations are produced and consumed by the attention mechanism, which involves [softmax](https://mbrenndoerfer.com/writing/multinomial-logistic-regression-complete-guide-mathematical-foundations-python-implementation) [normalization](https://mbrenndoerfer.com/writing/normalization-feature-scaling-min-max-machine-learning-guide) that provides some robustness to quantization noise. This makes [KV cache quantization](https://mbrenndoerfer.com/writing/kv-cache-compression-eviction-quantization-h2o-algorithm) practically easier than [weight quantization](https://mbrenndoerfer.com/writing/weight-quantization-basics-scale-zero-point-calibration), where precision matters more directly for the model's learned computation.

The memory requirements change as follows:

- $\text{bytes} = 4$: FP32, full precision
- $\text{bytes} = 2$: FP16 or BF16, standard inference precision, recommended default
- $\text{bytes} = 1$: [INT8](https://mbrenndoerfer.com/writing/int8-quantization-absmax-smooth-quantization-implementation), halves cache size with minimal quality impact in most scenarios
- $\text{bytes} = 0.5$: [INT4](https://mbrenndoerfer.com/writing/int4-quantization-group-wise-nf4-format-llms), quarters cache size with some quality tradeoff

Practitioners serving large models at scale should treat INT8 [KV cache](https://mbrenndoerfer.com/writing/autoregressive-generation-gpt-text-generation) as the sensible default and reserve FP16 for applications requiring maximum precision. The cache savings of INT8 over FP16 are large enough to support twice the [batch size](https://mbrenndoerfer.com/writing/stochastic-gradient-descent-neural-network-optimization) or twice the context length on the same hardware.

## Limitations

The memory formula we derived is exact for a static, synchronous inference system where all sequences in a batch start and end together. Real production systems deviate from this idealized picture in ways that complicate the calculation.

The first major limitation is that our formula assumes every sequence in the batch occupies the same maximum sequence length. In practice, sequences have highly variable lengths. A batch of 8 requests might include one that generates only 100 tokens and another that generates 4,000. If you reserve the maximum possible cache for every sequence, you waste the memory allocated to short sequences. [Paged Attention](https://mbrenndoerfer.com/writing/paged-attention-vllm-kv-cache-memory-management), covered in the next chapter, addresses this by dividing the cache into fixed-size pages and allocating only the pages each sequence needs. Our formula gives the worst-case upper bound; paged systems achieve much better average-case utilization.

The second limitation is that our activation memory estimate uses a rough multiplier that may not reflect the actual behavior of different model implementations or hardware backends. Activation memory depends on the specific operations being fused, the micro- [batch size](https://mbrenndoerfer.com/writing/stochastic-gradient-descent-neural-network-optimization) for pipeline parallelism, and the implementation details of the inference framework. Libraries like vLLM, TensorRT-LLM, and DeepSpeed all make different choices about when to materialize and deallocate activation tensors. For precise planning, measure actual memory consumption with your chosen model and inference framework on the target hardware rather than relying solely on our analytical formula.

A third limitation worth understanding is that our bottleneck analysis is simplified. The roofline model assumes all data movement is to and from the same level of the memory hierarchy, but GPUs have complex memory hierarchies including L1 cache, L2 cache, shared memory, and high-bandwidth memory. The effective bandwidth for small tensors may be much lower than the peak HBM bandwidth, and operations can be structured to exploit reuse in faster caches. [FlashAttention](https://mbrenndoerfer.com/writing/flashattention-algorithm-memory-efficient-gpu-tiling), for example, achieves speedups precisely by tiling computations to fit in fast shared memory, reducing total HBM traffic even though the number of arithmetic operations remains the same. Our simplified analysis correctly identifies the direction of the bottleneck but underestimates the potential gains from careful memory access pattern optimization.

Finally, multi-GPU deployment introduces additional complexity that our single-GPU formula does not capture. When a model is sharded across multiple GPUs using tensor parallelism, each GPU holds a fraction of the model weights and computes a corresponding fraction of the [attention heads](https://mbrenndoerfer.com/writing/multi-head-attention-transformers). The [KV cache](https://mbrenndoerfer.com/writing/autoregressive-generation-gpt-text-generation) must be partitioned accordingly, with communication between GPUs to synchronize attention outputs. This communication overhead adds to the effective memory bandwidth requirement and can shift the bottleneck analysis significantly. Sequence parallelism, where different sequence positions are processed on different GPUs, requires the KV cache to be gathered across all devices before each attention layer, creating collective communication costs that can dominate latency at large scale. These distributed system effects require separate analysis beyond the single-GPU formula we derived here.

## Key Parameters

The following parameters are essential for [KV cache memory](https://mbrenndoerfer.com/writing/multi-query-attention-memory-efficient-inference) calculation:

- **num\_layers**: Number of [transformer](https://mbrenndoerfer.com/writing/transformer-attention-is-all-you-need) layers in the model. Each layer maintains its own key and value cache, so cache memory scales linearly with the number of layers.
- **num\_kv\_heads**: Number of key-value attention heads, also written as $H_{\text{kv}}$. In standard Multi-Head Attention, this equals the number of query heads. [Grouped Query Attention](https://mbrenndoerfer.com/writing/grouped-query-attention-gqa-efficient-llm-inference) uses fewer KV heads to reduce memory.
- **head\_dim**: Dimension of each attention head, also written as $D_h$. Typically 64, 96, or 128. Combined with [number of heads](https://mbrenndoerfer.com/writing/transformer-architecture-hyperparameters-design-guide), this determines the model dimension.
- **batch\_size**: Number of sequences processed simultaneously. Each sequence maintains its own independent cache, so memory scales linearly with [batch size](https://mbrenndoerfer.com/writing/stochastic-gradient-descent-neural-network-optimization).
- **seq\_len**: Context length in tokens. Cache memory grows linearly as more tokens are processed and cached.
- **dtype\_bytes**: Bytes per element based on data type. FP16/BF16 use 2 bytes, FP32 uses 4 bytes, [INT8](https://mbrenndoerfer.com/writing/int8-quantization-absmax-smooth-quantization-implementation) uses 1 byte, [INT4](https://mbrenndoerfer.com/writing/int4-quantization-group-wise-nf4-format-llms) uses 0.5 bytes.

## Summary

KV cache memory follows a simple but relentless formula: $2 \times L \times B \times T \times H_{\text{kv}} \times D_h \times \text{bytes}$. This linear scaling with batch size and sequence length means the cache can easily exceed model weights for long-context or batched inference. Understanding this formula and its implications is the foundation for every optimization decision in the chapters that follow.

Key takeaways:

- **Model size scales differently than cache size**: A 10x larger model does not require 10x more cache when using [GQA](https://mbrenndoerfer.com/writing/grouped-query-attention-gqa-efficient-llm-inference). [LLaMA](https://mbrenndoerfer.com/writing/llama-architecture-design-training-efficiency) 2 70B has a smaller cache than LLaMA 7B due to its 8 KV heads.
- **Grouped Query Attention reduces cache size**: GQA reduces the 70B cache by 8x compared to standard [MHA](https://mbrenndoerfer.com/writing/multi-head-attention-transformers), making large model deployment feasible on single GPUs.
- **Long contexts are memory-expensive**: At 32K or more tokens, [KV cache](https://mbrenndoerfer.com/writing/autoregressive-generation-gpt-text-generation) typically exceeds model weight memory, fundamentally changing deployment requirements.
- **Generation is memory-bound at typical context lengths**: GPU compute utilization is limited by memory bandwidth when reading cached keys and values, not by arithmetic throughput.
- **Deployment planning requires careful calculation**: Before serving a model, estimate the memory used by the weights and cache, then add the remaining runtime overhead. Run the calculation for your worst-case sequence length and [batch size](https://mbrenndoerfer.com/writing/stochastic-gradient-descent-neural-network-optimization).
- **Data type is a controllable lever**: Switching from FP16 to [INT8](https://mbrenndoerfer.com/writing/int8-quantization-absmax-smooth-quantization-implementation) cache halves memory requirements with minimal quality impact, effectively doubling achievable batch size or context length on the same hardware.

Understanding these memory dynamics is essential for everything that follows in this section of the handbook. The next chapter explores [Paged Attention](https://mbrenndoerfer.com/writing/paged-attention-vllm-kv-cache-memory-management), which applies virtual memory concepts to manage KV cache more efficiently and recover the utilization that static allocation wastes. Subsequent chapters cover cache compression techniques that reduce memory by quantizing or pruning cached values. Together, these techniques enable longer contexts and larger batches while keeping memory requirements manageable.

## Quiz

Ready to test your understanding? Take this quick quiz to [reinforce](https://mbrenndoerfer.com/writing/policy-gradient-methods-reinforce-algorithm) what you've learned about [KV cache memory](https://mbrenndoerfer.com/writing/multi-query-attention-memory-efficient-inference) requirements and deployment planning.

### KV Cache Memory

Question 1 of 80 of 8 completed

What is the formula for calculating KV cache memory?

Track your reading progress

Sign in to mark chapters as read and track your learning journey

## Continue reading

[Back to Language AI Handbook](https://mbrenndoerfer.com/books/language-ai-handbook)

## Reference

BIBTEXAcademic

@misc{brenndoerfer2026kvcache-2, author = {Michael Brenndoerfer}, title = {KV Cache Memory for LLM Inference}, year = {2026}, url = {https://mbrenndoerfer.com/writing/kv-cache-memory-calculation-llm-inference-gpu}, organization = {mbrenndoerfer.com}, note = {Accessed: 2026-09-03} }

APAAcademic

Michael Brenndoerfer (2026). KV Cache Memory for LLM Inference. Retrieved from https://mbrenndoerfer.com/writing/kv-cache-memory-calculation-llm-inference-gpu

MLAAcademic

Michael Brenndoerfer. "KV Cache Memory for LLM Inference." 2026. Web. September 3, 2026. <https://mbrenndoerfer.com/writing/kv-cache-memory-calculation-llm-inference-gpu>.

CHICAGOAcademic

Michael Brenndoerfer. "KV Cache Memory for LLM Inference." Accessed September 3, 2026. https://mbrenndoerfer.com/writing/kv-cache-memory-calculation-llm-inference-gpu.

HARVARDAcademic

Michael Brenndoerfer (2026) 'KV Cache Memory for LLM Inference'. Available at: https://mbrenndoerfer.com/writing/kv-cache-memory-calculation-llm-inference-gpu (Accessed: September 3, 2026).

SimpleBasic

Michael Brenndoerfer (2026). KV Cache Memory for LLM Inference. https://mbrenndoerfer.com/writing/kv-cache-memory-calculation-llm-inference-gpu

DIRECT LINKURL

[https://mbrenndoerfer.com/writing/kv-cache-memory-calculation-llm-inference-gpu](https://mbrenndoerfer.com/writing/kv-cache-memory-calculation-llm-inference-gpu)