# Dedicated NVidia Blackwell Capacity for Sovereign Agentic AI

## Summary

Alongside the existing GH200 (Hopper) capacity on Isambard AI, provision a dedicated block of Blackwell Ultra GPUs, run as a standard always-on Ubuntu Linux environment with InfiniBand between nodes — no SLURM, no batch queue — sized to serve the largest current open-weight models as an interactive, always-available service.

## The problem this solves

The sovereign inferencing on Isambard AI pilot proved the core idea works: open-weight models, self-hosted on our own HPC, driven by standard agent harnesses, are a useful substitute for a commercial AI subscription.

There were 4 pain points caused by the Isambard AI architecture:
* Hopper class GPUs like those on Isambard are one generation behind the state of the art. They have about 1/3rd the GPU memory and do not support newer features, which means measured in terms of the size of a AI model they can support, they are at least six times smaller than the current Blackwell class GPUs.
* Isambard nodes are 4 GPU trays connected with NVLink. This compares to Blackwell class GPUs which are 8 GPU units. The per tray model size is at least 12 times smaller. Any per-GPU memory overhead eats into free space faster on hopper GPUs that in does on Blackwell.
* Between nodes Isambard is connected with Slingshot. Blackwell class use Infiniband to connect nodes, and this is where the effort is going in software support. Many bugs we identified came down to poor slingshot support.
* Isambard AI is designed and setup for high throughput large scale batch training not smaller scale interactive inferencing.

The result of this is Hopper on Isambard can comfortably support models up to 200 billion parameters, at a stretch can be pushed to 600 billion parameters, but over that it starts to be complex. These models still perform very well in most scenarios.

The current state of the art open weight models such as Kimi-K3, Qwen3.8, GLM5.3 range between 1 and 3 trillion parameters. They are not going to be simple to run on Isambard AI and hard to scale. It is almost inevitable that larger models will be released soon that rival the current Anthropic models.

Isambard AI is excellent at what it's actually for (large-scale batch training), but not designed specifically for an interactive, always-on inferencing service.

## The proposal

A dedicated Blackwell GPU deployment, run as its own environment rather than another SLURM partition:

- **Standard Ubuntu Linux, no SLURM.** Managed like any other always-on service (containers/systemd, not a job queue), matching how inference services are normally operated in industry. This directly removes the batch/interactive mismatch above — a user's request is served immediately by an already-running instance, not queued behind unrelated jobs.
- **B300 GPUs or GB300 GPUs**, alongside — not instead of — the existing GH200 capacity. GH200 remains the right, cost-effective choice for small-to-mid models and high-concurrency workloads; GB300 is specifically for the largest models and the lowest-friction path to serving them.
- **InfiniBand between nodes**, the standard, best-supported fabric for this workload class — removes an entire category of fabric-specific debugging the pilot had to absorb on Slingshot.

## Why Blackwell specifically

- **Native FP4 (NVFP4) support.** Confirmed during the pilot that Hopper (GH200) has no hardware support for 4-bit floating point at all — the largest models can currently only run at 8-bit or above on our existing hardware. Blackwell's native FP4 tensor cores roughly halve the memory footprint of the biggest models again versus FP8, on top of already having more memory per GPU (below) — the difference between "technically won't fit" and "runs comfortably."
- **More memory per GPU, and a larger single fast-interconnect domain.** Both apply regardless of which unit size below is chosen — more HBM per GPU than GH200 either way. The interconnect-domain benefit scales with unit size: B300/GB300 ships either as NVL72 rack-scale units (72 GPUs sharing one NVLink domain) or as smaller 8-GPU HGX-style units — both are still a materially larger single fast-interconnect domain than GH200's 4-GPUs-per-node NVLink-C2C, with Slingshot in between for anything larger. For the very largest models, that means substantially more of the model can stay on the fast interconnect before ever needing to cross a slower network link at all, and when it does it is better supported in sofware.

## Options:

1) Start small and scale.

If we are looking at the cheaper option it is recommended to stick to fewer larger GPUs in the blackwell class. 8x B300 GPUs with x86 CPUs with total of 2.3 Tb GPU memory is the starter unit. Systems come in around £500-700K. Can be rack mounted and added to over time, or single unit pods also available. This would serve the largest of the current generation of open weight model with ease, and additional units would help scale to larger models, over well supported inter-node InfiniBand connection.

2) Go large.

A GB300 NVL72 (72 GPUs) in specialised liquid cooled rack. The state of the art, prices difficult to get but estimate £4-8M. This will be future proof for the medium term, and provides capability to scale to serve large numbers of users.

## Open questions

- We don't yet have real demand data to size this based on usage.
- Procurement/hosting route (in-house vs. a managed GB300 offering) isn't addressed here — this is a case for the capability, not a specific vendor path.
- Access control and multi-tenancy on a non-SLURM environment needs its own design.
