# Design Documents — isambard-vllm v3

This directory contains the architectural design for the v3 migration.

## ** CURRENT STATE **

The current implementation has been rewritten from V2 by hand. The binary has
been renamed `ivllm2` in this branch. Eventually we will decide whether to keep
original as old `ivllm`

It has not been tested, and the test framework has not been setup. Code in test directory is LEGACY and will need to be rewritten,

Design decisions were made during refactoring that will invalidate existing tests and may not be properly documented.

Key next steps:

- [X] Review existing code. Identify obvious defects and inconsistent documentation. Record in design/issues.md.
- [ ] Setup bash bubblewrap testing enviroment with shims for mocking HPC login and compute nodes.
- [ ] Unit tests for bash utilities (refactor existing)
- [ ] Bash lockfile lifecycle test and monitor tests using shims.
- [ ] Refactor DryRunRemoteOps and create MockBackend to decouple CLI testing from backend
- [ ] Typescript tests (refactor existing ro write from scratch).

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
