#!/bin/bash
# tp-dp-pp-sweep.sh — PROTOTYPE, not wired into the build.
#
# Two tables, using ONLY config.json and model.safetensors.index.json:
#
#   Table 1 — weight loading. Sweeps power-of-2 GPU counts (TP×PP product)
#   within a node budget — NOT individual (TP, PP, DP) triples, since none
#   of them independently affect this table's numbers: weight sharding is
#   `total_bytes / (TP×PP)` and the activation-overhead model below depends
#   only on the model architecture and batch size, so only the PRODUCT
#   TP×PP matters, and DP never enters at all (see WIRING NOTE #13). Filters
#   out GPU counts where weights alone don't fit (no "DOES NOT FIT" rows
#   printed — they're just absent), and for the survivors shows GB used per
#   GPU and GB remaining at gpu-memory-utilization 0.90 and 0.95, EACH at
#   three assumed max-num-batched-tokens settings (4K/8K/16K) — because that
#   setting sizes vLLM's startup memory-profiling forward pass and
#   materially changes how much is left for KV cache (confirmed against a
#   real production log this session: dropping max-num-batched-tokens from
#   ~16K to 8192 freed ~6.2 GiB — see WIRING NOTE #11 for the formula and
#   its derivation/calibration). No KV cache numbers here — that's table 2.
#
#   Table 2 — KV cache sizing. For every distinct (TP, PP) that survived
#   table 1 (DP doesn't affect per-GPU KV cache at all, so it isn't swept
#   again here), computes KV cache GB needed at five max-model-len targets
#   (65536, 131072, 262144, 524288, 1048576 — i.e. 64K/128K/256K/512K/1024K)
#   for both bf16 and fp8 kv-cache-dtype. Non-hybrid models use vLLM's exact
#   per-layer formula (confirmed against vllm/v1/kv_cache_interface.py and
#   vllm/config/model.py, see WIRING NOTES #9-#10). Hybrid (mamba-containing)
#   models use vLLM's actual GROUP-PADDED allocator formula instead (WIRING
#   NOTE #12) — every layer-type group (attention, mamba) gets padded up to
#   the LARGEST group's layer count, which a naive per-layer sum badly
#   undershoots for real hybrid models (confirmed ~2.7x low against a real
#   Nemotron-3-Ultra production run). Flags whether each combination fits
#   within table 1's Remain@0.95/8K-batch for that (TP, PP).
#
# Split into two tables because these are genuinely different questions —
# weight fit is a property of (TP, PP) alone; KV cache is a property of
# (TP, PP, max_model_len, kv-cache-dtype) — and per Rob's own findings
# testing this against a real Nemotron-3-Ultra run, gpu-memory-utilization
# and kv-cache-dtype are the two levers that actually move the KV budget,
# so both need to be visible without cluttering the weight-fit question.
#
# RECOMMENDATION for the skill this feeds: prefer `max-model-len: auto` in
# the generated yaml over a hardcoded computed number. vLLM resolves `auto`
# to whatever actually fits at startup (live-profiled, more accurate than
# any config.json-derived formula — see WIRING NOTES #7/#9, a real ~10 GiB
# gap between this script's (now source-verified-correct) formula and a
# real production run persists even after fixing every bug found this
# session), and critically it still *prevents* a client from overriding it
# upward into an unsafe value. Table 2 is for RANKING TP/PP/kv-cache-dtype
# options against each other and sanity-checking a target context length is
# plausible, not for picking an exact number to hardcode.
#
# Intended as a building block for skills/generate-vllm-config/ — today the
# skill scrapes total parameter count from model-card prose (fragile, see
# SKILL.md:251/258) and uses a fixed
# `kv_per_token ≈ num_kv_heads × head_dim × 2 bytes × num_layers` formula
# missing the ×2 factor for storing both K and V (a real bug, found while
# designing this). This script fixes both: total weight bytes comes from
# model.safetensors.index.json's `metadata.total_size` (exact,
# architecture-agnostic, no formula needed), and the KV-cache formula
# includes the K+V factor explicitly.
#
# Usage: tp-dp-pp-sweep.sh <hf-model-name> <max-nodes>
# Example: tp-dp-pp-sweep.sh Qwen/Qwen2.5-0.5B-Instruct 2
#
# Requires: hf (huggingface_hub CLI — `uv tool install huggingface_hub`), jq, awk.
#
# KNOWN SIMPLIFICATIONS (deliberate, for a prototype — see WIRING NOTES at
# the bottom for the full history and what a real integration needs to add):
#   - Weight sharding is modeled as a flat `total_bytes / (TP × PP)` — this
#     is accurate for dense models. For MoE models it overstates how well
#     TP alone shards expert weights (real deployments use expert-parallel
#     for that, a separate dimension not modeled here) — treat the
#     per-GPU weight estimate for MoE models as optimistic/a floor, not
#     exact. The script prints an explicit note when MoE fields are found.
#   - Table 2's attention-layer KV cache is computed for attention-type
#     layers, PP-partitioned (`ceil(L_FULL / PP)`, assuming even
#     partitioning — real vLLM PP balancing may deviate). Non-attention
#     layers (linear/gated-attention variants other than true mamba2 — e.g.
#     Qwen4-exp/MiniMax-M3-style) are NOT modeled at all: no generic formula
#     exists for them from config.json field names alone (see WIRING NOTES
#     #9's explanation of why this is architecture-specific, not
#     genericizable). Mamba2-style layers (Nemotron-H) ARE modeled, gated
#     behind recognizing that architecture's specific field names — see the
#     MAMBA_* extraction below — using vLLM's real group-padded allocator
#     formula (WIRING NOTE #12), not a simple additive term. Both
#     modeled/unmodeled cases print an explicit note.
#   - `num_key_value_heads` per rank is `max(1, H_kv // TP)` — confirmed
#     directly against vLLM's own source (`ModelConfig.get_num_kv_heads()`,
#     vllm/config/model.py, and nemotron_h.py's own attention layer doing
#     the identical division inline) after this script briefly (and
#     wrongly) changed it to "replicate to the full H_kv on every rank" —
#     see WIRING NOTES #9. Per-GPU KV memory plateaus at ONE head's worth
#     once TP exceeds H_kv; it does not grow back up to the full count.
#   - GPU memory budget (96 GiB usable) is hardcoded to this project's own
#     GH200/Isambard convention (see skills/generate-vllm-config/SKILL.md)
#     — override via GPU_MEM_GB for other hardware. Table 1 always shows
#     both 0.90 and 0.95 utilization; override neither is needed there.
#   - Activation-memory overhead (per WIRING NOTE #11) is now MODELED, not a
#     flat guess: for MoE architectures it's
#     `max_num_batched_tokens × topk × (2×hidden + moe_intermediate) × 2
#     bytes`, for dense it's `max_num_batched_tokens × (hidden +
#     intermediate) × 2 bytes` — both first-principles guesses at what
#     vLLM's startup profile_run() actually measures (a real dummy forward
#     pass, not something it estimates from a formula either — see WIRING
#     NOTE #11). ACTIVATION_OVERHEAD_GB (default 2) is now only a FALLBACK
#     for architectures where neither shape is recognized.
#     ACTIVATION_BASELINE_GB (default 0.5) is a small fixed addend for
#     CUDA-context/NCCL buffers not captured by either formula. Real
#     overhead is kernel-implementation-dependent (tiling, latent-space MoE
#     dispatch, etc.) — treat this as a guess to sanity-check against, not a
#     substitute for reading vLLM's own "Available KV cache memory" log line
#     at your actual chosen max-num-batched-tokens.

set -euo pipefail

MODEL="${1:?usage: $0 <hf-model-name> <max-nodes>}"
MAX_NODES="${2:?usage: $0 <hf-model-name> <max-nodes>}"

GPUS_PER_NODE="${GPUS_PER_NODE:-4}"           # GH200 nodes on Isambard
GPU_MEM_GB="${GPU_MEM_GB:-96}"                # usable HBM3e per GPU (generate-vllm-config skill's own figure)
ACTIVATION_OVERHEAD_GB="${ACTIVATION_OVERHEAD_GB:-2}"   # fallback only — used when the architecture doesn't match either activation formula below
ACTIVATION_BASELINE_GB="${ACTIVATION_BASELINE_GB:-0.5}" # small fixed addend (CUDA context/NCCL buffers) on top of either formula
BLOCK_SIZE="${BLOCK_SIZE:-128}"               # matches vLLM's cdiv(max_model_len, block_size) rounding; 128 is this project's common default for sparse/hybrid models

MAX_LENS=(65536 131072 262144 524288 1048576)   # 64K 128K 256K 512K 1024K
BATCH_TOKENS=(4096 8192 16384)                  # 4K/8K/16K max-num-batched-tokens scenarios for table 1's activation-overhead columns

for tool in hf jq awk; do
    command -v "$tool" >/dev/null 2>&1 || { echo "FATAL: $tool not found on PATH" >&2; exit 1; }
done

# `hf download` prints its result as `path=<local path>` on the last
# non-warning stdout line — strip the prefix rather than assume bare output.
hf_download_path() {
    hf download "$1" "$2" 2>/dev/null | sed -n 's/^path=//p' | tail -n1
}

echo "=== fetching config.json for $MODEL ==="
CONFIG_PATH=$(hf_download_path "$MODEL" config.json) || true
[[ -z "$CONFIG_PATH" ]] && {
    echo "FATAL: could not fetch config.json for $MODEL (private/gated repo, or network policy — see design/references/ for manual-download fallback)" >&2
    exit 1
}

echo "=== fetching model.safetensors.index.json for $MODEL (optional — used for exact weight size) ==="
INDEX_PATH=$(hf_download_path "$MODEL" model.safetensors.index.json) || true
[[ -z "$INDEX_PATH" ]] && echo "  not found — either a single-shard model (no index needed) or genuinely unavailable; weight-size estimate will be skipped below"

# Fields can be at the top level (dense models) or nested under text_config
# (multimodal models — see the Qwen4-exp config this was designed against).
jqget() {
    jq -r "(.text_config.$1 // .$1) // empty" "$CONFIG_PATH"
}

HIDDEN_SIZE=$(jqget hidden_size)
NUM_LAYERS=$(jqget num_hidden_layers)
NUM_HEADS=$(jqget num_attention_heads)
NUM_KV_HEADS=$(jqget num_key_value_heads)
[[ -z "$NUM_KV_HEADS" ]] && NUM_KV_HEADS="$NUM_HEADS"   # no GQA field → plain MHA
HEAD_DIM=$(jqget head_dim)
[[ -z "$HEAD_DIM" ]] && HEAD_DIM=$(awk -v h="$HIDDEN_SIZE" -v n="$NUM_HEADS" 'BEGIN{print int(h/n)}')
NATIVE_MAX_LEN=$(jqget max_position_embeddings)
[[ -z "$NATIVE_MAX_LEN" ]] && { echo "WARNING: max_position_embeddings not found, defaulting to 4096" >&2; NATIVE_MAX_LEN=4096; }

# Hybrid-architecture detection: how many layers are actually full-attention
# (KV-cache-bearing) vs linear/mamba/ssm/gated/moe-FFN-only. The
# per-layer-type field name is NOT standardized across model families —
# `layer_types` (Qwen4-exp-style) and `layers_block_type` (NVIDIA
# Nemotron-H-style) are the two seen so far, add more as found. Nemotron-H
# also has a THIRD role, "moe" — an FFN-only layer with no attention/mamba
# mixer at all — excluded from the full-attention count alongside the
# mixer-replacement types.
LAYER_TYPES_JQ='(.text_config.layer_types // .layer_types // .text_config.layers_block_type // .layers_block_type // empty)'
LAYER_TYPES_TOTAL=$(jq -r "$LAYER_TYPES_JQ | if type == \"array\" then length else empty end" "$CONFIG_PATH")
if [[ -n "$LAYER_TYPES_TOTAL" ]]; then
    [[ -z "$NUM_LAYERS" ]] && NUM_LAYERS="$LAYER_TYPES_TOTAL"   # some configs (Nemotron-H) have no num_hidden_layers at all
    L_FULL=$(jq -r "$LAYER_TYPES_JQ | map(select(test(\"linear|mamba|ssm|conv|recurrent|gated|moe|expert\";\"i\") | not)) | length" "$CONFIG_PATH")
    L_MAMBA=$(jq -r "$LAYER_TYPES_JQ | map(select(test(\"mamba\";\"i\"))) | length" "$CONFIG_PATH")
    HYBRID_NOTE=" [HYBRID: $L_FULL/$LAYER_TYPES_TOTAL layers are full-attention (+$L_MAMBA mamba) — see table 2 notes]"
else
    L_FULL="$NUM_LAYERS"
    L_MAMBA=0
    HYBRID_NOTE=""
fi
if [[ -z "$L_FULL" || "$L_FULL" == "0" ]]; then
    KV_UNKNOWN=1
    [[ -z "$HYBRID_NOTE" ]] && HYBRID_NOTE=" [KV cache estimate UNAVAILABLE: could not determine full-attention layer count from config.json]"
else
    KV_UNKNOWN=0
fi

# Mamba2-style fixed-size recurrent state (Nemotron-H specific field names
# — see MambaStateShapeCalculator.mamba2_state_shape() in
# vllm/model_executor/layers/mamba/mamba_utils.py, confirmed this session).
# Only the dominant "temporal state" term is modeled (num_heads/TP ×
# head_dim × state_size); the smaller conv-state term is omitted. Other
# hybrid architectures' linear-attention layers (Qwen4-exp/MiniMax-M3-style
# GatedDeltaNet) use a DIFFERENT state shape entirely and are NOT modeled —
# genuinely architecture-specific, not derivable from field names alone.
MAMBA_NUM_HEADS=$(jqget mamba_num_heads)
MAMBA_HEAD_DIM=$(jqget mamba_head_dim)
MAMBA_STATE_SIZE=$(jqget ssm_state_size)
MAMBA_DTYPE_BYTES=4   # mamba_ssm_cache_dtype is float32 in every case seen so far
MAMBA_MODELED=0
if [[ -n "$MAMBA_NUM_HEADS" && -n "$MAMBA_HEAD_DIM" && -n "$MAMBA_STATE_SIZE" && "$L_MAMBA" -gt 0 ]]; then
    MAMBA_MODELED=1
fi

# MoE detection (informational + the per-GPU weight caveat noted above).
NUM_EXPERTS=$(jqget num_experts)
[[ -z "$NUM_EXPERTS" ]] && NUM_EXPERTS=$(jqget num_local_experts)
[[ -z "$NUM_EXPERTS" ]] && NUM_EXPERTS=$(jqget n_routed_experts)
MOE_NOTE=""
[[ -n "$NUM_EXPERTS" ]] && MOE_NOTE=" [MoE: $NUM_EXPERTS experts — per-GPU weight estimate assumes flat TP sharding, real deployments typically use expert-parallel for this, treat as optimistic]"

# Activation-memory model (WIRING NOTE #11): guesses the size of vLLM's
# startup profile_run() dummy forward pass at max_num_batched_tokens tokens,
# which is what actually determines "Available KV cache memory" — NOT a
# flat constant. MoE dispatch/expert-intermediate buffers dominate when
# present (scales with topk, not just num_experts); dense models fall back
# to the MLP intermediate-activation term.
MOE_INTERMEDIATE_SIZE=$(jqget moe_intermediate_size)
INTERMEDIATE_SIZE=$(jqget intermediate_size)
TOPK=$(jqget num_experts_per_tok)
[[ -z "$TOPK" ]] && TOPK=$(jqget moe_topk)
[[ -z "$TOPK" ]] && TOPK=$(jqget top_k)

ACT_MODE="unknown"
if [[ -n "$NUM_EXPERTS" && -n "$TOPK" && -n "$MOE_INTERMEDIATE_SIZE" ]]; then
    ACT_MODE="moe"
elif [[ -n "$INTERMEDIATE_SIZE" ]]; then
    ACT_MODE="dense"
fi

# activation_gb <num-tokens> — GB estimate for one profile_run()-style dummy
# forward pass over that many tokens, bf16 (2 bytes/elem) even for FP8-
# quantized weights (GEMM outputs/activations still accumulate at higher
# precision). Both formulas model ONE layer's peak working set (assuming
# sequential per-layer execution frees the previous layer's temporaries)
# plus a small fixed baseline — see WIRING NOTE #11 for derivation and the
# real-world calibration check.
activation_gb() {
    local t="$1"
    case "$ACT_MODE" in
        moe)
            awk -v t="$t" -v k="$TOPK" -v h="$HIDDEN_SIZE" -v i="$MOE_INTERMEDIATE_SIZE" -v base="$ACTIVATION_BASELINE_GB" \
                'BEGIN{printf "%.2f", (t*k*(2*h+i)*2)/1e9 + base}'
            ;;
        dense)
            awk -v t="$t" -v h="$HIDDEN_SIZE" -v i="$INTERMEDIATE_SIZE" -v base="$ACTIVATION_BASELINE_GB" \
                'BEGIN{printf "%.2f", (t*(h+i)*2)/1e9 + base}'
            ;;
        *)
            echo "$ACTIVATION_OVERHEAD_GB"
            ;;
    esac
}

TOTAL_WEIGHT_BYTES=""
if [[ -n "$INDEX_PATH" ]]; then
    TOTAL_WEIGHT_BYTES=$(jq -r '.metadata.total_size // empty' "$INDEX_PATH")
fi

echo
echo "Model: $MODEL$MOE_NOTE$HYBRID_NOTE"
echo "  hidden_size=$HIDDEN_SIZE num_layers=$NUM_LAYERS (full-attention: $L_FULL, mamba: $L_MAMBA) num_heads=$NUM_HEADS num_kv_heads=$NUM_KV_HEADS head_dim=$HEAD_DIM"
echo "  native max_position_embeddings=$NATIVE_MAX_LEN"
if [[ -n "$TOTAL_WEIGHT_BYTES" ]]; then
    echo "  total weight bytes (from safetensors index) = $TOTAL_WEIGHT_BYTES ($(awk -v b="$TOTAL_WEIGHT_BYTES" 'BEGIN{printf "%.1f", b/1e9}') GB)"
else
    echo "  total weight bytes: UNKNOWN (no safetensors index) — both tables below will be skipped"
fi
if (( MAMBA_MODELED )); then
    echo "  mamba state modeled: mamba_num_heads=$MAMBA_NUM_HEADS mamba_head_dim=$MAMBA_HEAD_DIM ssm_state_size=$MAMBA_STATE_SIZE"
elif [[ "$L_MAMBA" -gt 0 ]]; then
    echo "  mamba/linear-attention layers present but NOT modeled (unrecognized field names for this architecture) — table 2 KV numbers are a lower bound"
fi
case "$ACT_MODE" in
    moe)   echo "  activation-overhead model: MoE dispatch (topk=$TOPK, moe_intermediate_size=$MOE_INTERMEDIATE_SIZE) — see WIRING NOTE #11, this is a rough guess" ;;
    dense) echo "  activation-overhead model: dense MLP (intermediate_size=$INTERMEDIATE_SIZE) — see WIRING NOTE #11, this is a rough guess" ;;
    *)     echo "  activation-overhead model: UNRECOGNIZED architecture fields — falling back to flat ACTIVATION_OVERHEAD_GB=${ACTIVATION_OVERHEAD_GB}GB (no batch-size sensitivity)" ;;
esac
echo

if [[ -z "$TOTAL_WEIGHT_BYTES" ]]; then
    echo "Cannot proceed without weight size. Exiting."
    exit 0
fi

# Powers of 2 up to the total GPU budget (max_nodes × GPUS_PER_NODE).
mapfile -t POW2 < <(p=1; max=$((MAX_NODES*GPUS_PER_NODE)); while (( p <= max )); do echo "$p"; p=$((p*2)); done)

USABLE_90_GB=$(awk -v m="$GPU_MEM_GB" 'BEGIN{printf "%.2f", m*0.90}')
USABLE_95_GB=$(awk -v m="$GPU_MEM_GB" 'BEGIN{printf "%.2f", m*0.95}')

echo "── Table 1: weight loading (GPU_MEM_GB=$GPU_MEM_GB) — one row per unique TP×PP product (see WIRING NOTE #13: DP doesn't affect any number here, and neither weight sharding nor the activation-overhead model here distinguishes TP from PP — only their product matters) — Remain columns net out weight + a guessed activation overhead at each max-num-batched-tokens scenario (WIRING NOTE #11) ──"
printf "%-5s %-6s %-14s %-9s %-9s %-9s %-9s %-9s %-9s\n" \
    "GPUs" "Nodes" "Weight/GPU" "R90@4K" "R95@4K" "R90@8K" "R95@8K" "R90@16K" "R95@16K"

declare -A SEEN_TP_PP=()        # dedupe key "$TP,$PP" → 1, for table 2 (populated below per surviving product — table 2 CANNOT collapse by product, see WIRING NOTE #13)
declare -A REMAIN95_8K_OF=()    # "$TP,$PP" → Remain@0.95/8K-batch(GB), reused by table 2's fit check (8K = vLLM's own OPENAI_API_SERVER default on >70GB GPUs)
declare -A WEIGHT_OF=()         # "$TP,$PP" → Weight/GPU(GB), for reference

for GPUS in "${POW2[@]}"; do
    NODES=$(( (GPUS + GPUS_PER_NODE - 1) / GPUS_PER_NODE ))
    (( NODES > MAX_NODES )) && continue

    WEIGHT_GB_PER_GPU=$(awk -v b="$TOTAL_WEIGHT_BYTES" -v g="$GPUS" 'BEGIN{printf "%.2f", b/g/1e9}')

    ACT_4K=$(activation_gb "${BATCH_TOKENS[0]}")
    ACT_8K=$(activation_gb "${BATCH_TOKENS[1]}")
    ACT_16K=$(activation_gb "${BATCH_TOKENS[2]}")

    R90_4K=$(awk -v u="$USABLE_90_GB" -v w="$WEIGHT_GB_PER_GPU" -v a="$ACT_4K" 'BEGIN{printf "%.2f", u-w-a}')
    R95_4K=$(awk -v u="$USABLE_95_GB" -v w="$WEIGHT_GB_PER_GPU" -v a="$ACT_4K" 'BEGIN{printf "%.2f", u-w-a}')
    R90_8K=$(awk -v u="$USABLE_90_GB" -v w="$WEIGHT_GB_PER_GPU" -v a="$ACT_8K" 'BEGIN{printf "%.2f", u-w-a}')
    R95_8K=$(awk -v u="$USABLE_95_GB" -v w="$WEIGHT_GB_PER_GPU" -v a="$ACT_8K" 'BEGIN{printf "%.2f", u-w-a}')
    R90_16K=$(awk -v u="$USABLE_90_GB" -v w="$WEIGHT_GB_PER_GPU" -v a="$ACT_16K" 'BEGIN{printf "%.2f", u-w-a}')
    R95_16K=$(awk -v u="$USABLE_95_GB" -v w="$WEIGHT_GB_PER_GPU" -v a="$ACT_16K" 'BEGIN{printf "%.2f", u-w-a}')

    # "Valid" = fits at the cheapest scenario (0.95 util, smallest 4K
    # batch — least activation overhead) with some headroom left for KV
    # cache (Remain@0.95/4K > 0). Rows that don't even clear that bar
    # are dropped entirely, not printed as "DOES NOT FIT" placeholders —
    # per Rob's ask. A row surviving this filter may still show negative
    # Remain at 8K/16K batch — that's the point of showing all three.
    if ! awk -v r="$R95_4K" 'BEGIN{exit !(r>0)}'; then
        continue
    fi

    printf "%-5s %-6s %-14s %-9s %-9s %-9s %-9s %-9s %-9s\n" \
        "$GPUS" "$NODES" "$WEIGHT_GB_PER_GPU" \
        "$R90_4K" "$R95_4K" "$R90_8K" "$R95_8K" "$R90_16K" "$R95_16K"

    # Table 2 needs every individual (TP, PP) factor pair of this product,
    # not just the product — populate SEEN_TP_PP here without printing a
    # table 1 row per pair (see WIRING NOTE #13). PP is capped at NODES
    # (WIRING NOTE #14): pipelining across GPUs within the same node is
    # never a good idea (wastes the fast intra-node interconnect on
    # latency-sensitive pipeline handoffs instead of tensor-parallel
    # all-reduces) — PP > NODES would necessarily put more than one
    # pipeline stage on the same node, since NODES here already = the
    # minimum nodes this GPU count needs.
    for TP in "${POW2[@]}"; do
        (( TP > GPUS || GPUS % TP != 0 )) && continue
        PP=$((GPUS / TP))
        (( PP > NODES )) && continue
        KEY="$TP,$PP"
        if [[ -z "${SEEN_TP_PP[$KEY]:-}" ]]; then
            SEEN_TP_PP[$KEY]=1
            REMAIN95_8K_OF[$KEY]="$R95_8K"
            WEIGHT_OF[$KEY]="$WEIGHT_GB_PER_GPU"
        fi
    done
done
echo

if (( KV_UNKNOWN )); then
    echo "Table 2 (KV cache) skipped: could not determine full-attention layer count from config.json."
    exit 0
fi

echo "── Table 2: KV cache size PER GPU, per surviving (TP, PP) from table 1, at each max-model-len (BLOCK_SIZE=$BLOCK_SIZE) ──"
echo "  (per-GPU, not total: LOCAL_KV_HEADS is already TP-sharded and LOCAL_L_FULL/LOCAL_L_MAMBA are already PP-partitioned — matches how vLLM itself reports 'Available KV cache memory' per-worker, and matches table 1's Remain columns being per-GPU too, so the MaxSeqs@.95 comparison below is apples-to-apples.)"
if (( MAMBA_MODELED )); then
    echo "  (hybrid architecture: using vLLM's group-padded allocator formula, not a per-layer sum — see WIRING NOTE #12. Deliberately overestimates rather than undershoots.)"
fi
printf "%-5s %-5s %-10s %-16s %-16s %-14s %-14s\n" \
    "TP" "PP" "max-len" "KV-GB/GPU(bf16)" "KV-GB/GPU(fp8)" "MaxSeqs@.95 bf16" "MaxSeqs@.95 fp8"

# Sort keys for stable, readable output (TP then PP, numeric).
for KEY in $(printf '%s\n' "${!SEEN_TP_PP[@]}" | sort -t, -k1,1n -k2,2n); do
    TP="${KEY%,*}"
    PP="${KEY#*,}"
    REMAIN_95="${REMAIN95_8K_OF[$KEY]}"

    # max(1, H_kv // TP) — see header note and WIRING NOTES #9.
    LOCAL_KV_HEADS=$(awk -v h="$NUM_KV_HEADS" -v tp="$TP" 'BEGIN{v=int(h/tp); print (v<1?1:v)}')
    # Layers are partitioned roughly evenly across PP stages; only this
    # stage's share of full-attention layers needs KV cache here.
    LOCAL_L_FULL=$(awk -v l="$L_FULL" -v pp="$PP" 'BEGIN{printf "%d", (l+pp-1)/pp}')

    LOCAL_L_MAMBA=0
    LOCAL_MAMBA_HEADS=0
    if (( MAMBA_MODELED )); then
        LOCAL_L_MAMBA=$(awk -v l="$L_MAMBA" -v pp="$PP" 'BEGIN{printf "%d", (l+pp-1)/pp}')
        LOCAL_MAMBA_HEADS=$(awk -v h="$MAMBA_NUM_HEADS" -v tp="$TP" 'BEGIN{v=int(h/tp); print (v<1?1:v)}')
    fi

    # kv_gb_hybrid <dtype_bytes> — vLLM's ACTUAL hybrid-model KV-cache
    # allocator formula (WIRING NOTE #12), not a per-layer sum: every
    # layer-TYPE group (attention, mamba) gets padded up to the LARGEST
    # group's layer count, and both groups are forced to share one uniform
    # page byte-size. Deliberately used only when MAMBA_MODELED — for
    # non-hybrid models this padding never happens, per-layer-sum stays exact.
    kv_gb_hybrid() {
        local dtype_bytes="$1"
        # page_size is the UNIFORM per-block byte size (unify_kv_cache_spec_page_size)
        # — compared using attention's PER-BLOCK bytes (BLOCK_SIZE tokens), not its
        # total-sequence bytes (t tokens). blocks_attn then divides the FULL
        # sequence's attn bytes by that resolved page — these are two different
        # quantities, conflating them was a real bug caught by re-testing.
        awk -v lf="$LOCAL_L_FULL" -v lm="$LOCAL_L_MAMBA" -v h="$LOCAL_KV_HEADS" -v d="$HEAD_DIM" -v db="$dtype_bytes" \
            -v bs="$BLOCK_SIZE" -v t="$ROUNDED_LEN" \
            -v mh="$LOCAL_MAMBA_HEADS" -v mhd="$MAMBA_HEAD_DIM" -v ms="$MAMBA_STATE_SIZE" -v mdb="$MAMBA_DTYPE_BYTES" \
            'BEGIN{
                attn_block_bytes = 2*h*d*db*bs
                attn_total_bytes = 2*h*d*db*t
                mamba_bytes = mh*mhd*ms*mdb
                page = (attn_block_bytes > mamba_bytes) ? attn_block_bytes : mamba_bytes
                blocks_attn = int((attn_total_bytes + page - 1)/page)
                blocks_mamba = int((mamba_bytes + page - 1)/page)
                group = (lf > lm) ? lf : lm
                total = group * page * (blocks_attn + blocks_mamba)
                printf "%.2f", total/1e9
            }'
    }

    for MAX_LEN in "${MAX_LENS[@]}"; do
        # Non-hybrid: vLLM's exact per-layer formula (FullAttentionSpec.
        # max_memory_usage_bytes, vllm/v1/kv_cache_interface.py — WIRING
        # NOTE #9): cdiv(max_len, block_size) × block_size × 2(K+V) ×
        # num_kv_heads × head_size × dtype_size, summed over this stage's
        # full-attention layers.
        # Hybrid (MAMBA_MODELED): vLLM's group-padded allocator formula
        # instead (WIRING NOTE #12) — a plain per-layer sum undershoots by
        # ~2.7x for real hybrid models, this deliberately overestimates by
        # ~1.5x instead, which is the safer direction per Rob's own call.
        ROUNDED_LEN=$(awk -v m="$MAX_LEN" -v bs="$BLOCK_SIZE" 'BEGIN{printf "%d", int((m+bs-1)/bs)*bs}')

        if (( MAMBA_MODELED )); then
            KV_GB_BF16=$(kv_gb_hybrid 2)
            KV_GB_FP8=$(kv_gb_hybrid 1)
        else
            KV_GB_BF16=$(awk -v l="$LOCAL_L_FULL" -v h="$LOCAL_KV_HEADS" -v d="$HEAD_DIM" -v t="$ROUNDED_LEN" \
                'BEGIN{printf "%.2f", (l*2*h*d*2*t)/1e9}')
            KV_GB_FP8=$(awk -v l="$LOCAL_L_FULL" -v h="$LOCAL_KV_HEADS" -v d="$HEAD_DIM" -v t="$ROUNDED_LEN" \
                'BEGIN{printf "%.2f", (l*2*h*d*1*t)/1e9}')
        fi

        # Concurrency ratio (Remain@0.95/8K ÷ this row's KV-GB/GPU) — how many
        # max_len-length sequences worth of KV cache fit, i.e. the same
        # quantity vLLM itself logs as "Maximum concurrency for %s tokens per
        # request: %.2fx" (kv_cache_utils.py:2178-2182). "-" when it's ≤1
        # (doesn't even fit one full-length sequence — a plain fail, not a
        # concurrency number worth reporting).
        MAXSEQS_BF16=$(awk -v k="$KV_GB_BF16" -v r="$REMAIN_95" 'BEGIN{v=r/k; if(v>1) printf "%.2f", v; else print "-"}')
        MAXSEQS_FP8=$(awk -v k="$KV_GB_FP8" -v r="$REMAIN_95" 'BEGIN{v=r/k; if(v>1) printf "%.2f", v; else print "-"}')

        MAX_LEN_LABEL="${MAX_LEN}"
        case "$MAX_LEN" in
            65536) MAX_LEN_LABEL="64K" ;;
            131072) MAX_LEN_LABEL="128K" ;;
            262144) MAX_LEN_LABEL="256K" ;;
            524288) MAX_LEN_LABEL="512K" ;;
            1048576) MAX_LEN_LABEL="1024K" ;;
        esac

        printf "%-5s %-5s %-10s %-16s %-16s %-14s %-14s\n" \
            "$TP" "$PP" "$MAX_LEN_LABEL" "$KV_GB_BF16" "$KV_GB_FP8" "$MAXSEQS_BF16" "$MAXSEQS_FP8"
    done
done

echo
echo "MaxSeqs@.95 = Remain@0.95/8K (table 1, same TP/PP) ÷ this row's KV-GB/GPU — how many max_len-length sequences worth of KV cache fit, the same quantity vLLM logs as 'Maximum concurrency for N tokens per request: X.XXx'. Shown as '-' when it's ≤1 (doesn't even fit one full-length sequence). If your actual max-num-batched-tokens is 4K or 16K (or something else), recompute against table 1's matching Remain column instead — the fit picture can change meaningfully, see WIRING NOTE #11."
echo "Prefer max-model-len: auto in the generated yaml over hardcoding a value from this table — the intended use here is picking a starting max-num-batched-tokens/max-model-len to try, then letting an actual run with max-model-len: auto tell you the real number (see WIRING NOTE #12)."

# ── WIRING NOTES (how this would become part of generate-vllm-config) ──────
#
# 1. Fix the existing bug this script's header also flags: SKILL.md:497's
#    `kv_per_token ≈ num_kv_heads × head_dim × 2 bytes × num_layers` is
#    missing the ×2 for K+V — it's currently `H_kv × D_h × bytes × L`
#    instead of `2 × H_kv × D_h × bytes × L`, undercounting by exactly 2x.
#
# 2. Add a `model.safetensors.index.json` fetch alongside the existing
#    model-card fetch (SKILL.md step that determines parameter count) and
#    prefer `metadata.total_size` over model-card prose when available —
#    fixes the documented "ask the user" fallback (SKILL.md:258) for models
#    whose card doesn't state a clean total.
#
# 3. This script's tables only cover weight-fit and KV-cache sizing. A real
#    integration also needs the skill's existing MoE-backend-selection
#    logic (SKILL.md's Backend Selection Guide), the num_key_value_heads-vs-TP
#    divisibility warning (shared-layer-replication rule), and the
#    hybrid-architecture "keep full context" exception (SKILL.md:503) — this
#    script's hybrid detection is a generalization of that single
#    hand-written exception into a pattern match on `layer_types`, intended
#    to replace it rather than duplicate it.
#
# 4. NETWORK POLICY NOTE from testing this session: `huggingface.co` itself
#    needed allowlisting (now done). For repos whose weights are served via
#    HF's newer Xet storage backend, actual file content additionally comes
#    from `cas-server.xethub.hf.co` and/or region CDN hosts like
#    `us.aws.cdn.hf.co` — both blocked by default deny, confirmed while
#    testing this script against cyankiwi/MiniMax-M3-AWQ-INT4. `config.json`
#    and `model.safetensors.index.json` are small enough that this may not
#    matter for THIS script specifically (metadata-only fetches sometimes
#    route differently), but a real venv/model download would hit this.
#
# 5. Tested this session: the config.json-fetch and "no index found"
#    graceful-fallback path both verified against a real, small, public
#    model (Qwen/Qwen2.5-0.5B-Instruct — single safetensors shard, no
#    index). The "index found, compute real weight bytes" path was
#    validated against a hand-written synthetic index file (network policy
#    blocked fetching a real multi-shard model's index this session, see
#    note 4) — re-verify against a real large model once that's unblocked.
#
# 6. Real crash found and fixed testing against
#    RedHatAI/NVIDIA-Nemotron-3-Ultra-550B-A55B-FP8-dynamic (a real,
#    already-deployed model in this project — see
#    examples/nemotron-3-ultra-550B-A55B-FP8.yaml): `model_type: nemotron_h`
#    has NO `num_hidden_layers` field at all, and uses `layers_block_type`
#    (not `layer_types`) for its per-layer roles — neither was recognized,
#    so L_FULL came out empty, and the KV-cache math divided by zero mid-sweep.
#    Also: this architecture has a THIRD per-layer role, `"moe"` (an
#    FFN-only layer with no attention/mamba mixer at all, interleaved
#    1:1 with the mixer layers — 48 mamba + 48 moe + 12 attention = 108
#    total), which the original exclude-pattern didn't know about either —
#    would have silently counted those 48 moe-only layers as full-attention
#    too (8x overcount: 96 instead of the real 12) had the crash not caught
#    it first. Fixed by (a) trying multiple known layer-type field names,
#    (b) deriving NUM_LAYERS from that list's length when
#    num_hidden_layers is absent, (c) adding moe/expert to the
#    non-attention exclude pattern, and (d) guarding against a
#    still-undetermined layer count. This is inherently a whack-a-mole
#    problem — expect more field-name/layer-role variants from future
#    model families.
#
# 7. Second real-world discrepancy, same Nemotron-3-Ultra model, TP=8/PP=1:
#    a live production run needed 16.42 GiB KV cache for one sequence at
#    max-model-len=1048576 (note: 4x this model's config.json
#    max_position_embeddings of 262144 — a rope-scaling extension this
#    script has no way to see from config.json alone, flagged as a real
#    limitation, not fixed). CUDA-graph/profiling overhead is also real and
#    approximate even in vLLM's own accounting — that same run's log said
#    "--gpu-memory-utilization=0.9500 is equivalent to 0.9398... increase
#    to 0.9602 to maintain the same effective KV cache size".
#
# 8. (Superseded by #10's table split, kept for history) Originally output
#    was reframed from "estimated max-model-len" to "KV pool token
#    capacity" + "implied max-num-seqs at native max-model-len", per Rob's
#    observation that vLLM's actual paged KV cache is one shared block pool
#    (continuous batching), not a per-sequence static reservation.
#
# 9. Checked vLLM's actual source for the exact formula behind the
#    ValueError in point 7 (Rob, 2026-09-04) — `vllm/v1/core/kv_cache_utils.py`
#    (`check_enough_kv_cache_memory`/`_check_enough_kv_cache_memory`, sums
#    `spec.max_memory_usage_bytes()` over every layer's `KVCacheSpec`) and
#    `vllm/v1/kv_cache_interface.py`. Findings, all now incorporated into
#    table 2 above:
#    - `FullAttentionSpec.max_memory_usage_bytes()` is exactly
#      `cdiv(max_model_len, block_size) × 2 × block_size × num_kv_heads ×
#      head_size × dtype_size` per layer — confirms this script's per-layer
#      formula (2 × H_kv × D_h × bytes × T, K+V factor and now block-size
#      rounding both included) is exactly right.
#    - `num_kv_heads` in that formula comes from
#      `ModelConfig.get_num_kv_heads()`: `max(1, total_num_kv_heads //
#      tensor_parallel_size)` — floor to at least 1, NOT replicate to the
#      full head count. This DIRECTLY CONTRADICTS what point 6 briefly
#      changed it to earlier this session (a wrong assumption, not sourced
#      at the time) — reverted, now confirmed both generically and in
#      nemotron_h.py's own attention layer (`self.num_kv_heads = max(1,
#      self.total_num_kv_heads // tp_size)`, line 433), which does the
#      identical division inline rather than going through the generic
#      helper.
#    - `MambaSpec.max_memory_usage_bytes()` (mamba/SSM layers) is real and,
#      unlike attention, does NOT scale with max_model_len at all — it's
#      `page_size_bytes × (1 + num_speculative_blocks)`, a fixed per-layer
#      state size from `MambaStateShapeCalculator.mamba2_state_shape()`
#      (vllm/model_executor/layers/mamba/mamba_utils.py). Now modeled in
#      table 2 (the MAMBA_* extraction and MAMBA_GB term) for Nemotron-H
#      specifically; genuinely small (~1 MB/layer here) so its ABSENCE
#      wasn't what caused the ~10 GiB gap noted in point 7, but it's now
#      included anyway since it's cheap to compute correctly.
#    - Even with every fix above applied and block-size rounding included,
#      table 2's TP=8/PP=1/1048576/bf16 figure was still well short of the
#      real 16.42 GiB a production run needed (~2.7x low) — root-caused and
#      fixed in point 12 below (a per-layer sum is fundamentally the wrong
#      model for hybrid architectures, not missing a small term).
#
# 10. Restructured into two tables (Rob, 2026-09-04): a weights-only table
#     filtered to valid combinations (no "DOES NOT FIT" placeholder rows)
#     showing remaining GB at both 0.90 and 0.95 utilization, and a
#     separate KV-cache sweep across the 5 max-model-len targets and both
#     kv-cache-dtype options, using vLLM's exact formula per point 9 rather
#     than an approximation. Reused across both tables: table 1 populates
#     SEEN_TP_PP/REMAIN95_OF/WEIGHT_OF associative arrays for every
#     (TP, PP) pair that clears the 0.95-utilization bar (any DP), which
#     table 2 iterates directly rather than re-sweeping and re-filtering.
#     (REMAIN95_OF was later renamed REMAIN95_8K_OF — see point 11.)
#
# 11. Modeled activation-memory overhead instead of a flat guess (Rob,
#     2026-09-04) — motivated by a real production observation: dropping
#     --max-num-batched-tokens from ~16K to 8192 on the SAME Nemotron-3-
#     Ultra TP=8/PP=1 deployment (identical 67.39 GiB weight load) moved
#     vLLM's own reported "Available KV cache memory" from 8.36 GiB to
#     14.57 GiB — a 6.2 GiB swing from ONE scheduler setting nobody would
#     normally think of as a memory knob.
#
#     Root cause, traced through vLLM source: `determine_available_memory()`
#     (vllm/v1/worker/gpu_worker.py:448-545) computes
#     `available_kv_cache_memory = requested_memory - non_kv_cache_memory -
#     cudagraph_memory_estimate`, where `non_kv_cache_memory` includes
#     `torch_peak_increase` — the ACTUAL MEASURED peak allocator usage from
#     a real dummy forward pass, `profile_run()` →
#     `self._dummy_run(self.max_num_tokens, is_profile=True)`
#     (gpu_model_runner.py:6424-6426), where `self.max_num_tokens =
#     scheduler_config.max_num_batched_tokens` (gpu_model_runner.py:504).
#     So the size of that one dummy step's token batch IS the size of the
#     activation-memory reservation — not an estimate, a real profiled peak.
#
#     Derived a first-principles guess for what that peak scales with
#     (config.json fields only, no vLLM run needed): for MoE architectures,
#     the dominant term is the fused-MoE dispatch/expert-intermediate/
#     combine buffers, each token replicated to its `topk` selected experts:
#     `T × topk × (2×hidden_size + moe_intermediate_size) × 2 bytes`. For
#     Nemotron-3-Ultra (hidden=8192, moe_intermediate=5120,
#     num_experts_per_tok=22 — an unusually high topk, which is WHY MoE
#     dominates here) that's ~924 KiB/token. Solving the real 6.2 GiB
#     observed delta for the prior max-num-batched-tokens value gives
#     ΔT ≈ 7,040 → T_before ≈ 15,200 — startlingly close to 16384, vLLM's
#     own LLM_CLASS default max_num_batched_tokens for >70GB GPUs
#     (arg_utils.py:2463). That's real corroboration (the formula was
#     derived from config.json fields alone, not fit to this data point),
#     not proof — see caveats below.
#
#     Dense models (no MoE fields) fall back to the analogous single-layer
#     MLP-intermediate term: `T × (hidden_size + intermediate_size) × 2
#     bytes` — no topk multiplier, so typically ~2 orders of magnitude
#     smaller per token than the MoE case above.
#
#     CAVEATS (real, acknowledged, not fixed):
#     - Assumes the fused-MoE kernel materializes full T×topk-scale buffers
#       rather than tiling internally — a more memory-efficient grouped-GEMM
#       kernel implementation would show flatter, sub-linear scaling, and
#       this guess would overestimate the true overhead.
#     - Nemotron-3-Ultra's config.json also has `moe_latent_size=2048`
#       (unmodeled here) — if the kernel actually dispatches tokens in that
#       compressed latent space rather than full hidden_size width, the real
#       MoE term could be ~4x smaller than this formula predicts.
#     - This is a PRIOR to sanity-check a config against before running
#       anything, not a substitute for reading vLLM's own "Available KV
#       cache memory" log line at your actual chosen
#       max-num-batched-tokens — that number is always the ground truth.
#     - ACTIVATION_BASELINE_GB (0.5 GB default) is a guess at the fixed,
#       batch-size-independent floor (CUDA context, NCCL buffers) — not
#       derived from anything, just carried over as a small constant so the
#       formula doesn't imply zero overhead as max-num-batched-tokens → 0.
#
# 12. Root-caused and fixed the ~2.7x undershoot flagged as unresolved in
#     point 9 (Rob, 2026-09-04) — it was never a missing term, it was the
#     wrong MODEL. Traced vLLM's actual hybrid-model KV-cache sizing
#     (`_max_memory_usage_bytes_from_groups`, general-case branch,
#     `vllm/v1/core/kv_cache_utils.py:1832-1892`, plus
#     `unify_kv_cache_spec_page_size` at line 1067): it does NOT sum each
#     layer's independent KV memory. It buckets layers into per-TYPE
#     "groups" (all full-attention layers → one group, all mamba layers →
#     another), forces every group's per-layer page to share one uniform
#     byte-size (mamba's fixed-size state, being independent of block_size,
#     usually wins and attention's block_size gets scaled up to match — this
#     cancels out for a full-length single sequence, doesn't change the
#     final byte count), and then — the actual dominant effect, straight
#     from the function's own docstring ("if a model has 8 full attention
#     layers and 9 sliding window layers, they will be padded to 9 full + 9
#     sliding window for uniform group sizes") — PADS EVERY GROUP'S LAYER
#     COUNT UP TO THE LARGEST GROUP'S COUNT:
#       group_size = max(len(group) for group in kv_cache_groups)
#       memory = group_size × uniform_page_size × total_blocks_needed
#     For Nemotron-3-Ultra the mamba group (48 layers) dwarfs the attention
#     group (12, or 13 counting the MTP head's own attention layer — see
#     point 6/7, now also confirmed via `mtp_layers_block_type` +
#     `num_nextn_predict_layers=1` in the real config.json). So vLLM
#     reserves memory as if there were 48 attention-equivalent layers, not
#     13 — this dwarfs anything a per-layer sum could ever produce.
#
#     Implemented as `kv_gb_hybrid()`: for the MAMBA_MODELED branch only
#     (non-hybrid models never hit this path in vLLM, per-layer-sum stays
#     exact for them), compute one attention layer's bytes and one mamba
#     layer's bytes for this (TP, PP, max_len, dtype), take the uniform page
#     as the max of the two, block-ify both against that page, and multiply
#     the summed block count by max(LOCAL_L_FULL, LOCAL_L_MAMBA) — i.e.
#     vLLM's exact formula, not an approximation of it.
#
#     Validated against the real Nemotron-3-Ultra TP=8/PP=1/max_len=1048576
#     production ValueError ("16.42 GiB KV cache is needed"): this formula
#     gives ~24.05 GiB — overestimates by ~46%, versus the old per-layer-sum
#     formula's ~6.49 GB (undershoots by ~2.7x, the wrong direction). The
#     residual ~46% is plausibly the MTP layer forming its own separate
#     (small) group rather than joining the 12-layer attention group, and/or
#     num_speculative_blocks nuances not derivable from config.json alone —
#     not chased further. Per Rob's explicit call: "better to be closer and
#     conservatively wrong" — this table is meant to inform a starting
#     max-num-batched-tokens/max-model-len guess, with the real number then
#     coming from an actual run with `max-model-len: auto`, so overshooting
#     is the safe failure direction and undershooting (risking a real OOM
#     the operator didn't expect) is the one worth actively avoiding.
#
# 13. Collapsed table 1 from (TP, PP, DP) triples to unique TP×PP products,
#     and dropped DP entirely (Rob, 2026-09-04) — asked directly whether DP
#     ever changes table 1's math, and whether table 2 reduces to the
#     product too. Checked both against the actual code and real output:
#     - Table 1: `WEIGHT_GB_PER_GPU = total_bytes / (TP×PP)` and
#       `activation_gb()` (WIRING NOTE #11) depend only on the model's
#       architecture fields and the assumed batch size — TP and PP never
#       appear individually, and DP doesn't appear AT ALL. Confirmed
#       empirically: before this fix, every (TP,PP) pair sharing a product
#       (1×8, 2×4, 4×2, 8×1, all product=8) printed byte-for-byte identical
#       Weight/GPU and Remain columns. So table 1 now sweeps GPU-count
#       (TP×PP product) directly — one row per achievable per-replica GPU
#       count — instead of enumerating every redundant factorization and
#       every DP multiple of it. DP (data-parallel replica count) is a
#       deployment-scaling decision made independently of per-GPU memory
#       fit, not something this table needs to model at all.
#     - Table 2: the OPPOSITE is true — `LOCAL_KV_HEADS` depends on TP
#       ALONE (head-count sharding) and `LOCAL_L_FULL`/`LOCAL_L_MAMBA`
#       depend on PP ALONE (layer-depth partitioning); these are different
#       sharding axes with materially different effects, not
#       interchangeable via their product. Confirmed from real output: at
#       product=8, TP=8/PP=1/1024K/bf16 needs 25.82 GB/GPU while
#       TP=1/PP=8/1024K/bf16 needs only 6.49 GB/GPU — a ~4x difference for
#       the SAME product. So table 2 cannot be collapsed by product; it
#       still enumerates every individual (TP, PP) factor pair.
#     Net result: table 1's GPUS loop now also enumerates each surviving
#     product's (TP, PP) factor pairs internally (not printed as table 1
#     rows) purely to populate SEEN_TP_PP/REMAIN95_8K_OF/WEIGHT_OF for
#     table 2's use — table 1 stays collapsed-by-product on screen, table 2
#     still gets full (TP, PP) granularity underneath.
#
# 14. Capped PP at NODES (Rob, 2026-09-04): "I don't think it would ever be
#     a good idea to make GPUs in node pipeline parallel." Since NODES here
#     is already the minimum node count this GPU total needs
#     (`ceil(GPUS/GPUS_PER_NODE)`), any factor pair with PP > NODES would
#     necessarily place more than one pipeline stage on the same node —
#     e.g. for GPUS_PER_NODE=4, TP=1/PP=4 needs only 1 node but splits it
#     into 4 separate pipeline stages, each stage's single GPU sharing a
#     node with 3 GPUs from OTHER stages. Pipeline parallelism trades
#     latency-sensitive point-to-point handoffs for reduced communication
#     volume; tensor parallelism uses the same GPUs for the opposite
#     (bandwidth-heavy, latency-tolerant all-reduces) — so intra-node's fast
#     interconnect should go to TP, and PP should only ever cross node
#     boundaries. Added `(( PP > NODES )) && continue` to the (TP, PP)
#     factor-pair enumeration that feeds table 2 — table 1 itself is
#     unaffected (it was already collapsed by GPUS/product, see WIRING NOTE
#     #13), this only prunes which (TP, PP) splits table 2 considers.
