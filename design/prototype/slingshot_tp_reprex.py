#!/usr/bin/env python3
# design/prototype/slingshot_tp_reprex.py — PROTOTYPE, not production (see
# AGENTS.md: scripts in design/ are instructional examples, rewrite before
# shipping).
#
# Companion to slingshot-tp-reprex.sh. Drives a plain torch.distributed NCCL
# AllGather in a tight loop across an 8-rank / 2-node group, sized and shaped
# to match the exact collective vLLM's MoE/EP dispatch path
# (AgRsAll2AllManager.dispatch(), vendor/vllm-0.26.0/vllm/distributed/
# device_communicators/all2all.py) issues per layer during real GLM-5.2
# serving — no vLLM, no Ray, no model weights involved at all.
#
# Why this shape specifically: logs/glm52q/20260901_164849's nccl-debug.log
# (NCCL_DEBUG=INFO) showed every hang bottoming out in an AllGather with
# `count 19360 datatype 9` (bf16), and — the actual finding this reprex
# exists to test in isolation — NCCL's own topology-aware channel builder
# elects exactly one rank per physical node (local slot 3 in every run
# observed so far, on 4 different physical nodes across 2 independent hangs)
# as the sole inter-node network bridge for that collective; every other
# rank's traffic for the same call stays on-node via NVLink/PCI P2P and
# never touches Slingshot at all. If this is a real Slingshot/CXI network
# reliability issue rather than anything vLLM-specific, hammering the
# identical 8-rank/2-node AllGather shape with no model compute in between
# should reproduce the same rank-3/rank-7 divergence, much faster and much
# cheaper than a full vLLM startup + warmup cycle.
#
# Usage: launched once per SLURM task by slingshot-tp-reprex.sh (which sets
# RANK/WORLD_SIZE/LOCAL_RANK/MASTER_ADDR/MASTER_PORT from SLURM_* env vars).
# Not meant to be run directly.

import os
import socket
import statistics
import sys
import time
from datetime import timedelta

import torch
import torch.distributed as dist

ELEMS = int(os.environ.get("REPREX_ELEMS", "19360"))  # matches observed AllGather count
ITERS = int(os.environ.get("REPREX_ITERS", "20000"))
HEARTBEAT_EVERY = int(os.environ.get("REPREX_HEARTBEAT_EVERY", "500"))
STALL_WARN_SECS = float(os.environ.get("REPREX_STALL_WARN_SECS", "2.0"))
INIT_TIMEOUT_SECS = float(os.environ.get("REPREX_INIT_TIMEOUT_SECS", "120"))

RANK = int(os.environ["RANK"])
WORLD_SIZE = int(os.environ["WORLD_SIZE"])
LOCAL_RANK = int(os.environ["LOCAL_RANK"])
HOST = socket.gethostname()


def log(msg):
    now = time.strftime("%H:%M:%S")
    print(f"[{now}] [rank {RANK} local {LOCAL_RANK} {HOST}] {msg}", flush=True)


def main():
    log(
        f"starting: world_size={WORLD_SIZE} elems={ELEMS} (bf16, "
        f"{ELEMS * 2} bytes/rank) iters={ITERS} stall_warn={STALL_WARN_SECS}s"
    )

    torch.cuda.set_device(LOCAL_RANK)
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        rank=RANK,
        world_size=WORLD_SIZE,
        timeout=timedelta(seconds=INIT_TIMEOUT_SECS),
    )
    log("process group initialised — NCCL communicator constructed")

    # Static buffers, reused every iteration — mirrors AgRsAll2AllManager's
    # real dispatch/combine tensors being fixed-shape per decode step, and
    # keeps this loop allocation-free so it stresses only the collective.
    send_buf = torch.full((ELEMS,), float(RANK), dtype=torch.bfloat16, device="cuda")
    recv_buf = torch.empty(ELEMS * WORLD_SIZE, dtype=torch.bfloat16, device="cuda")

    dist.barrier()
    log("barrier cleared — all ranks present, starting AllGather loop")

    latencies_ms = []
    last_heartbeat = time.monotonic()
    loop_start = time.monotonic()

    for i in range(1, ITERS + 1):
        t0 = time.monotonic()
        dist.all_gather_into_tensor(recv_buf, send_buf)
        torch.cuda.synchronize()
        dt = time.monotonic() - t0
        latencies_ms.append(dt * 1000)

        if dt > STALL_WARN_SECS:
            log(
                f"⚠️  STALL at iteration {i}/{ITERS}: this call took {dt:.2f}s "
                f"(threshold {STALL_WARN_SECS}s) — this rank is the straggler"
            )

        if i % HEARTBEAT_EVERY == 0:
            now = time.monotonic()
            recent = latencies_ms[-HEARTBEAT_EVERY:]
            log(
                f"heartbeat: iter={i}/{ITERS} "
                f"since_last={now - last_heartbeat:.2f}s "
                f"last_{HEARTBEAT_EVERY}_mean={statistics.mean(recent):.3f}ms "
                f"max={max(recent):.3f}ms"
            )
            last_heartbeat = now

    total = time.monotonic() - loop_start
    summary = (
        f"DONE iters={ITERS} total={total:.2f}s "
        f"mean={statistics.mean(latencies_ms):.3f}ms "
        f"median={statistics.median(latencies_ms):.3f}ms "
        f"p99={sorted(latencies_ms)[int(len(latencies_ms) * 0.99)]:.3f}ms "
        f"max={max(latencies_ms):.3f}ms"
    )
    log(summary)

    # Gather every rank's one-line summary onto rank 0 so the final log
    # output has a single, directly comparable per-rank table — no need to
    # go hunting through interleaved multi-rank stdout to spot a straggler.
    payload = summary.encode("utf-8").ljust(256, b"\0")
    send_t = torch.tensor(list(payload), dtype=torch.uint8, device="cuda")
    recv_t = torch.empty(256 * WORLD_SIZE, dtype=torch.uint8, device="cuda")
    dist.all_gather_into_tensor(recv_t, send_t)

    if RANK == 0:
        log("=== per-rank summary table ===")
        chunks = recv_t.cpu().numpy().tobytes()
        for r in range(WORLD_SIZE):
            chunk = chunks[r * 256 : (r + 1) * 256].split(b"\0")[0].decode("utf-8")
            print(f"  rank {r}: {chunk}", flush=True)

    dist.barrier()
    dist.destroy_process_group()
    log("clean exit")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log("FATAL — exception in reprex loop, re-raising after flushing:")
        import traceback

        traceback.print_exc()
        sys.stdout.flush()
        raise
