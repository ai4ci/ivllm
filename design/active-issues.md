# Active Issues

This document tracks known, pre-existing bugs in the `isambard-vllm` repository.

## Known Issues

*(No active issues currently tracked)*

## Resolved

- **Console log tailing dropped/garbled lines** (missing `-n` history window on `tail -f`, raw `\r` progress bars overwriting console lines, chunk-not-line stdout echoing in `runRemote`, and a racy double-tail handoff between job startup and `watchLog`). Fixed by: explicit `-a`/from-now flag on `ivllm-show-log.sh` replacing the ambiguous default `tail -f` window; `stdbuf -oL tr '\r' '\n'` normalization in both tail pipelines; line-buffering in `SshRemoteOps.runRemote()`; and removing `requestStart()`'s own log monitor in favour of a single `watchLog()` call in `cmdConnect`/`cmdCancel` gated by status polling (`isStarting`/`isStopped`) instead of log-text markers.
