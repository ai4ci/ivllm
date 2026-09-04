#!/usr/bin/env python3
"""Stand-in for `vllm serve` — just enough surface to demonstrate the patch
mechanism without needing an actual vLLM install. See ../README.md (this
directory's run-plain.sh / run-bwrap-overlay.sh) for how it's used.

Deliberately dumb: one function whose behavior is what a model-specific
patch would change, called once at import/run time, like a model file's
top-level behavior (imports, class bodies) in the real vLLM case this
prototype stands in for.
"""

MODEL_NAME = "cyankiwi/MiniMax-M3-AWQ-INT4"


def load_model_layer():
    """Stand-in for e.g. MinimaxM3QKVParallelLinearWithIndexer's fused
    load path — unpatched, this is the "wrong for this checkpoint" behavior."""
    return "single fused qkv+indexer GEMM (breaks on mixed-precision checkpoints)"


def serve():
    layer = load_model_layer()
    print(f"vLLM stand-in serving {MODEL_NAME}")
    print(f"  attention layer: {layer}")


if __name__ == "__main__":
    serve()
