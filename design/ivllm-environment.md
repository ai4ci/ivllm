# Ivllm environment

This document describes the paths and configuration environment variables that
are set in ivllm and perform functional roles such as compiler paths and are
set either by the brics nccl, libfabric modules themselves or by
[common-env.sh](src/engine/lib/common-env.sh).

They assume the naming conventions in the [resolve functions](src/engine/lib/utils.sh)

e.g. `$nvhpc_dir` in this document is the same as a call to `resolve_nvhpc_dir` in `utils.sh`
and `$IVLLM_PROJECTDIR` is set and is the project root. `$HOME` for user home

* Variable / flag
* Value: value set in common-env.sh or inherited from brics modules
* Strength of evidence:
    - proven (known failure without this value)
    - recommendation (Isambard or GH200 specific: Brics, isamabrd containers, doubleword.ai)
    - documentation (official documentation only)
    - recipe (vllm recipes or cookbooks - non GH200 specific)
    - inferred (web searches)

References:
* https://docs.nvidia.com/hpc-sdk/installation-guide/index.html#end-user-environment-settings


TODO: start populating with variables from common-env.sh
TODO: research debugging options in whole stack for improving ivllm debug logging

## Compiler / library paths

## Networking configuration

## Cache setup

## Debugging flags
