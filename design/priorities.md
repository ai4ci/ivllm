# Priorities

one line todo list to keep track of different directions:
NOT FOR UPDATES: link to active_issues.md if needed

N.b. `[ ]` means not started `[/]` in progress, `[X]` complete

* [ ] Multi-model router - requirements
* [ ] VLLM 0.29.0 - Qwen3.8-flash-next and GLM5.3-flash support - on hold
* [ ] Review disabled level 4 (torch/cuda-gdb) diagnostics - on hold
* [ ] Agent launcher - review prototype code from old versions - requirements
* [/] Formalise patching method in engine (e.g. solar-open2 support) - design
* [/] Hardware diagnostics / optimise network outside of vllm - NCCL / libfabric / NIXL / DeepEP / Humming on isambard - prototype
* [ ] JIT compiler diagnostics - PTXAS CUDA compiler performance on isambard - requirements
* [/] Benchmarking - e2e testing
* [ ] generate-vllm-config skill updating - maintenance
* [ ] Testing cpu-offloading and kv-value-offloading - benchmarking
* [ ] Integrate Isambard_containers as a backend - design
* [ ] VLLM docker containers as backend - investigation
* [/] Scheduling jobs - test and document
* [/] 4 node jobs on isambard - needs retesting
* [/] Minimax-M3-FP8 support (2 node) - on hold
* [/] GLM-5.2-FP8 support - on hold
* [ ] Mistral128 \[THINK\] blocks showing in pi - debug
* [ ] Nemotron-super <|im_end|> blocks parsed incorrectly in pi - debug


## Model specific investigation - details
* [ ] Re-test NCCL_CUMEM_ENABLE now NCCL is 2.30.4 - may be a stale pre-2.22.3 workaround (vllm#5091/#24141) - benchmarking
* [ ] Read vllm#45198, #8410, #46097, #41725, #29740, #40926, #41530 - closely-matched community multi-node hang reports - research
* [ ] Resolve whether vLLM's TP/EP collectives use pynccl or torch.distributed's NCCL backend (VLLM_DISABLE_PYNCCL) - determines if TORCH_NCCL_DESYNC_DEBUG etc actually apply - active issue
* [ ] Test VLLM_DISTRIBUTED_USE_SPLIT_GROUP=1 on GLM-5.2 - independent PyTorch Gloo/split_group hang mechanism found (pytorch/pytorch#145376) - active issue
* [ ] Check whether GLM-5.2 hangs correlate with idle-then-burst traffic pattern, not just sustained load (vllm#42742) - active issue
* [ ] Watch for vllm#40926-style MTP deadlock (sustained low-rate traffic, sample_tokens RPC timeout) when GLM-5.2-FP8+MTP testing resumes on 4 nodes - active issue
* [ ] Try VLLM_GPU_NIC_PCIE_MAPPING/VLLM_NIC_SELECTION_VARS for GPU-NIC RDMA affinity on Slingshot (vllm PR#42083, real feature) - benchmarking

## Complete
* [X] Tested disable-custom-all-reduce on GLM-5.2 (20260812_213446) - hang still occurs, vllm#8410 mechanism ruled out - active issue
* [X] Confirmed NCCL_DEBUG=INFO verbose logging exhausted as a diagnostic - logs everything up to the freeze then goes totally silent during the hang, same as libfabric logging - active issue
* [X] Confirmed our own GLM-5.2 logs already show "No available shared memory broadcast block found in 60 seconds" - matches vllm#51593/#45198 verbatim - active issue
* [X] Configuration knowledge base - supported flags for VLLM, NCCL, FI_CXI, vllm serve - documentation
* [X] ivllm-environment  - documentation
* [X] GLM-5.2-AWQ-INT4
* [X] MiniMax-M3-AWQ-INT4
* [X] JIT caching and diagnostics
* [X] Verify VLLM_USE_V2_MODEL_RUNNER=0 actually took effect in GLM-5.2 runs, not silently overridden (vllm#44697) - active issue
