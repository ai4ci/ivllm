# Priorities

one line todo list to keep track of different directions:
N.b. `[ ]` means not started `[/]` in progress, `[X]` complete

* [ ] Multi-model router - requirements
* [ ] Re-write AI torch flight recorder capture prototype slop - look at TODOs in utils.sh. Needs to be node local (in `monitor_node`, née `wait_report`) and triggered by a node local condition probably. Still relevant, 2026-09-03: `report_torch()`'s own header comment now says outright "this does not currently result in anything informative... the torch output is nonsense" — the underlying problem this TODO describes was never fixed, just confirmed.
* [x] GLM-5.2-AWQ-INT4 establish cause of hang with debug monitor - **RESOLVED 2026-09-02/03**: was a silently-reverted NCCL version pin, not a vLLM bug — see `active-issues.md`'s resolved GLM-5.2 entry. Follow-on work opened instead: a live `NCCL_GDRCOPY_ENABLE` regression found during the fix's documentation pass (top of `active-issues.md`'s Known Issues, needs a decision), and a batch of config knobs (`numa-bind`, `no-async-scheduling`, `disable-custom-all-reduce`, `fuse_allreduce_rms`) worth testing for removal now the real cause is fixed.
* [ ] Agent launcher - review prototype code from old versions - requirements
* [/] Formalise patching method in engine (e.g. solar-open2 support) - design
* [/] Hardware diagnostics / optimise network outside of vllm - NCCL / libfabric / NIXL / DeepEP / Humming on isambard - requirements. **Concrete next step, 2026-09-03**: evolve `design/prototype/slingshot-tp-reprex.sh` (built during the GLM-5.2 hang investigation — a standalone 2-node/8-GPU NCCL AllGather stress test, no vLLM/Ray, matching real job resource shape) into a proper standalone multinode network benchmarking harness. This is now blocking several genuinely unresolved config questions that can only be settled by measurement, not code-reading: whether `NCCL_GDRCOPY_ENABLE`/`FI_HMEM_CUDA_USE_GDRCOPY` actually help or hurt (see `active-issues.md`'s top Known Issue — conflicting evidence from Isambard's own container def vs. this project's earlier libfabric-bug-reading pass), and whether `FI_CXI_RDZV_EAGER_SIZE`/other Slingshot vendor-default flags do anything measurable on this fabric. Much cheaper to iterate on than a full vLLM job (no model load, results in ~a minute).
* [ ] JIT compiler diagnostics - PTXAS CUDU compiler performance on isambard - requirements
* [/] Benchmarking - e2e testing
* [ ] generate-vllm-config skill updating - maintenance
* [ ] Testing cpu-offloading and kv-value-offloading - benchmarking
* [ ] Integrate Isambard_containers as a backend - design
* [ ] Scheduling jobs - requirements
* [ ] Refactor `is_startable()` (`utils.sh`) vs `ivllm-serve.sh`'s duplicated inline status-guard block - the two already drifted once (`warmup` case added only to the inline copy) - maintenance
* [ ] JIT caching and diagnostics - active issue
* [/] 4 node jobs on isambard - awaiting support response
* [/] Minimax-M3 support - active issue blocked
* [/] GLM-5.2-FP8 support - active issue blocked (4 node job)


## Model specific investigation - details
* [X] Verify VLLM_USE_V2_MODEL_RUNNER=0 actually took effect in GLM-5.2 runs, not silently overridden (vllm#44697) - active issue
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
