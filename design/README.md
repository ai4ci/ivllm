# Design Documents — isambard-vllm v3

This directory contains the architectural design for the v3 migration.

## ** CURRENT STATE **

The current implementation has been rewritten from V2 by hand. The binary is
named `ivllm2` in package.json; the name is provisional pending a final decision
on whether to rename the original `ivllm` binary.

A bubblewrap (bwrap) sandbox test harness runs bash tests in isolated environments
with real subprocess/signal semantics against the real installed `yq 3.4.1` and
`jq 1.7` binaries. All 13 documented implementation issues (design/issues.md)
have been identified and resolved.

**Test status:**
- **Bash**: 74 assertions across 10 test files (1 unit + 9 sandboxed), all green
- **TypeScript**: 62 assertions across 6 test files, all green
- **Total**: 136 assertions, 0 failures

Key completed work:

- [X] Review existing code. Identify defects and document in design/issues.md.
- [X] Build bash bubblewrap testing environment with PATH shims.
- [X] Rewrite bash unit tests (lockfile, cache, config, vllm-env, monitor).
- [X] Add bash login-node handoff, monitor startup/worker, and exit trap tests.
- [X] Rewrite TypeScript tests from scratch — Backend unit tests, MockRemoteOps, local-ops, semver, CLI lifecycle integration.
- [X] Wire bash tests into `bun test` via integration wrapper.
- [X] Full documentation pass: added JSDoc/bash doc headers to all source files.
  - `utils.sh` — 44+ functions documented (path resolvers, lockfile state machine, monitors, traps, semver, cache).
  - `Backend.ts`, `index.ts`, `RemoteOps.ts`, `SshRemoteOps.ts`, `utils.ts` — JSDoc added or improved.
  - Login wrapper scripts (`ivllm-*.sh`) and SLURM templates (`slurm-*.sh`) — doc headers added.
- [X] Bug fixes discovered and applied during documentation:
  - `get_job_status_setting()`: `$job` → `$1` in error message (undefined variable).
  - `get_job_config_setting()`: `$job` → `$1` in error message (undefined variable).
  - `update_status_slurm_id()`: `-z` → `-n` in condition (was writing SLURM ID when empty).
  - `Backend.ts`: Lifecycle methods (`isStopped`, `isStartable`) intentionally return `true` on missing lockfile — job was never started, therefore startable.

## Outstanding work

The following items are pending and can be addressed after e2e testing is established:

- [ ] **CLI handler unit tests** — `cmdConnect`, `cmdSetup`, `cmdConfig`, `cmdStatus`, `cmdCancel` in `src/index.ts` have no unit tests. Only the Backend contract is covered in integration tests.
- [ ] **TypeScript type check** — No `tsc --noEmit` or `--strict` check in the test pipeline. Add as a fast gate before `bun test`.
- [ ] **ESLint / linting** — No linting configured in the test workflow.
- [ ] **Monitor idle timeout unit test** — `monitor_head`'s log-parsing idle check works end-to-end but has no focused unit test for the time-pattern matching logic.
- [ ] **Semver module convergence** — `src/semver.ts` (TypeScript) and `utils.sh` (bash) both implement semver parsing/comparison. Consider documenting the divergence or converging to a single source of truth.

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
