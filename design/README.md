# Design Documents — isambard-vllm v3

This directory contains the architectural design for the v3 migration.

## ** CURRENT STATE **

The current implementation has been rewritten from V2 by hand. The binary has
been renamed `ivllm2` in this branch. Eventually we will decide whether to keep
original as old `ivllm`

A bubblewrap sandbox test harness is now in place, running bash tests in
isolated environments with real subprocess/signal semantics against the real
installed yq 3.4.1 and jq 1.7 binaries. All 13 documented implementation
issues (design/issues.md) have been identified and resolved. Test status:
40/40 individual assertions pass across 5 test files (lockfile, cache,
config, vllm-env, monitor-head). Zero failures.

Key next steps (tomorrow):

- [X] Review existing code. Identify obvious defects and inconsistent documentation. Record in design/issues.md.
- [X] Setup bash bubblewrap testing environment with shims for mocking HPC login and compute nodes.
- [X] Unit tests for bash utilities (refactor existing).
- [X] Bash lockfile lifecycle test and monitor tests using shims.
- [ ] Expand bash test coverage: monitor_startup, monitor_worker, exit trap/signal tests, login-node handoff tests against shim call log.
- [ ] Refactor DryRunRemoteOps and create MockBackend to decouple CLI testing from backend
- [ ] Typescript tests (refactor or write from scratch)

## Active documents

| Document | Description |
|----------|-------------|
| `architecture.md` | System architecture: layers, lockfile protocol, monitor triad, lifecycle |
| `adr.md` | Architecture Decision Records for the v3 redesign |
| `roadmap.md` | Step-by-step migration plan with 7 phases (M1–M7) |
| `coding-standards.md` | Coding conventions: TypeScript OptionParser, object-oriented patterns, bash function design |
| `testing.md` | Test architecture: 4 layers, mock infrastructure, handoff testing, scenario matrix, phase-by-phase plan |

## Reference and support

| Directory | Description |
|-----------|-------------|
| `prototype/` | Working prototype of the bash framework and test harness |
| `references/` | External reference docs (CUDA, vLLM serve, Slingshot, storage notes) |
| `old/` | Archived design docs from v1/v2 (kept for reference) |

## Quick start for new contributors

1. Read `architecture.md` for the high-level picture
2. Read `roadmap.md` for what's being built and in what order
3. For specific decisions, see `adr.md`
4. For the prototype bash framework, see `prototype/prototype.sh`
5. Archived v2 docs are in `old/` if you need context on past decisions
