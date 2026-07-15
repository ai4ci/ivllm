# Design Documents — isambard-vllm v3

This directory contains the architectural design for the v3 migration.

## Active documents

| Document | Description |
|----------|-------------|
| `architecture.md` | System architecture: layers, lockfile protocol, monitor triad, lifecycle |
| `adr.md` | Architecture Decision Records for the v3 redesign |
| `roadmap.md` | Step-by-step migration plan with 7 phases (M1–M7) |
| `coding-standards.md` | Coding conventions: TypeScript OptionParser, object-oriented patterns, bash function design |
| `testing.md` | Test architecture: 4 layers, mock infrastructure, handoff testing, scenario matrix, phase-by-phase plan |
| `implementation-plan.md` | **Day 1** — Phase M1 bash framework (foundation, additive) |
| `implementation-plan-2.md` | **Day 2** — Phase M2 new CLI commands (types, paths, metadata, connect/cancel scaffold) |
| `implementation-plan-3.md` | **Day 3** — Phase M3 self-managed lifecycle (SSH ops, real connect/cancel, remove old code) |

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
