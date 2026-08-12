# CPU Offload and Unified Memory on GH200

Source: [NVIDIA Developer Blog - CPU-GPU Memory Sharing](https://developer.nvidia.com/blog/accelerate-large-scale-llm-inference-and-kv-cache-offload-with-cpu-gpu-memory-sharing/)

## GH200 Unified Memory Architecture

The NVIDIA GH200 Grace Hopper Superchip connects CPU and GPU via **NVLink-C2C** — a 900 GB/s, memory-coherent interconnect that delivers 7× the bandwidth of PCIe Gen 5.

Key benefits:
- Creates a single unified memory address space shared by CPU and GPU
- Enables transparent access to CPU memory from GPU without explicit data transfers
- Eliminates redundant memory copies between CPU and GPU

### Memory Capacity

| Component | Memory |
|-----------|--------|
| GPU (H100) | 96 GB HBM3 |
| CPU (Grace) | 460-480 GB LPDDR |
| Interconnect | NVLink-C2C at 900 GB/s |

This allows large models to load even when they exceed GPU memory limits — the GPU can access CPU memory transparently.

## vLLM CPU Offload (`--cpu-offload-gb`)

### How it Works

The `--cpu-offload-gb` flag reserves CPU memory space per GPU for model weights. Weights are loaded from CPU memory to GPU memory on-the-fly during each forward pass using **UVA (Unified Virtual Addressing)** for zero-copy access.

### Effect

Think of it as virtual GPU memory:
```
Virtual GPU capacity = Physical GPU memory + --cpu-offload-gb

Example: 24 GB GPU + 10 GB offload = 34 GB virtual capacity
```

### Configuration Options

| Option | Description | Default | Use Case |
|--------|-------------|---------|----------|
| `--cpu-offload-gb` | CPU memory (GiB) per GPU for weights | 0 | Basic offload using UVA |
| `--offload-backend` | "auto", "prefetch", or "uva" | auto | "uva" for cpu-offload-gb |
| `--offload-group-size` | Group N layers, offload last N | 0 | Async prefetch (not UVA) |
| `--offload-num-in-group` | Layers to offload per group | 1 | Must be ≤ offload-group-size |

### Example: Llama 3 70B

From NVIDIA's article, Llama 3 70B at FP16 requires ~140 GB:
```yaml
model: meta-llama/Llama-3.1-70B
# Fails with OOM on single 96 GB GPU without offload
tensor-parallel-size: 1

# With offload:
# --cpu-offload-gb: 50  # reserves 50 GB CPU memory
# GPU memory used: ~90 GB (140 - 50)
# GPU has 96 GB available, so gpu-memory-utilization: 0.90 is sufficient
```

### Requirements

- **Hardware**: Requires fast CPU-GPU interconnect (NVLink-C2C on GH200)
- **vLLM version**: ≥ 0.6.0 (OffloadConfig introduced in this version)

### When to Use

✅ Good for:
- Models that exceed per-GPU VRAM but fit on single node with offload
- Testing single-node deployment before committing to multi-node
- Reducing inter-node communication latency

❌ Not ideal for:
- Very large models (multi-node is better)
- Models that fit at fp8 quantization
- Non-NVLink architectures (PCIe too slow for effective offload)

### Performance Considerations

- UVA zero-copy enables transparent access but still has memory bandwidth limitations
- NVLink-C2C (900 GB/s) is fast enough that offload latency is minimal
- Prefetch backend can hide transfer latency for specific layer groups
- Memory bandwidth is the bottleneck, not latency (unlike PCIe)

## Key Takeaways for Isambard AI

Isambard AI's GH200 nodes are ideal for CPU offload because:
1. **NVLink-C2C** provides the necessary 900 GB/s coherent interconnect
2. **460 GB CPU memory** per node provides ample offload space
3. **4 × 96 GB GPUs** means offload can help each GPU individually

This makes single-node deployment feasible for larger models (e.g., Llama-3-70B) that would otherwise require multi-node setup.
