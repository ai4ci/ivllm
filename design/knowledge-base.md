# Isambard VLLM configuration knowledge base

This document must be actively maintained after model debugging or benchmarking.

There are an overwhelming number of possible enviroment flags and vllm serve cli options.

This document tells us what is available, what is relevant, what ivllm is defaulting
to and what has been tested and works / does not work / makes no difference in
the vllm testing and benchmarking we have done so far. For every flag we need a
recommendation.

The flags or options **must be proven to exist in recent versions of the relevant software**
and not be a hallucination, large amounts of web sourced content is unreliable or
outdated.

Referenced:
* The vllm source code - gold standard
* VLLM env vars: https://docs.vllm.ai/en/stable/configuration/env_vars/
* VLLM serve: https://docs.vllm.ai/en/stable/cli/serve/
* NCCL: https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2307/user-guide/docs/env.html
* NCCL (Brics): https://docs.isambard.ac.uk/user-documentation/guides/nccl/
* Modules (Brics): https://docs.isambard.ac.uk/user-documentation/guides/modules/
* VLLM (Brics): https://docs.isambard.ac.uk/user-documentation/tutorials/distributed-inference/
* Libfabric: https://ofiwg.github.io/libfabric/main/man/onepage.html
* Slingshot libfabric: https://ofiwg.github.io/libfabric/main/man/fi_cxi.7.html
* CUDA: https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/environment-variables.html
* Torch: https://docs.pytorch.org/docs/2.13/torch_environment_variables.html

The scope of this is limited to flags that may materially affect the stability
or performance, or behaviour of vllm on isambard, and might be set as options in
a `vllm.yaml` configuration, or are defaulted in `vllm-env.sh`. It also lists
flags or variables that can't affect Isambard or are irrelevant and why, so
that we don't waste time investigating dead ends.

Out of scope: purely mechanistic e.g. cache directories, ports, IPs that are
set by ivllm itself, debugging options, hard coded library paths that might be found in
`common-env.sh`. for these see [ivllm-environment](design/ivllm-environment.md).

* Variable / flag
* Recommended value, or range of values, or json value for complex flags, or
heuristic for selecting (within a context)
* IVLLM default: value set in vllm-env.sh or common-env.sh
* Impact: How does this affect stability, performance, tuning, behaviour?
* Strength of recommendation: always, sometimes, never.
* Strength of evidence:
    - proven (reproducible debugging or benchmarking output)
    - recommendation (Isambard or GH200 specific: Brics, isamabrd containers, doubleword.ai)
    - documentation (official documentation only),
    - recipe (vllm recipes or cookbooks - non GH200 specific)
    - inferred (web searches)
* Context: specific models / attention mechanisms / quantisation schemes / vllm version if relevant

TODO: start populating with variables from vllm-env.sh and existing examples
TODO: analyse log failures and identify specific broken or high value flags
TODO: research CUDA and TORCH flags (particularly NCCL related) for things related
to potential JIT compiler hangs.
TODO: check whether vllm-env.sh is still correct.

## Enviroment variables

### VLLM

### NCCL / Libfabric

### Compliers (e.g. CUDA, Torch etc)

### Backends (e.g. DeepGEMM)

### Irrelevant to Isambard



## Vllm serve configuration

### Backends

### Offloading

### Irrelevant to Isambard
