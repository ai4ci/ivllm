/**
 * tests/unit/bash-integration.test.ts — Runs the full bash test suite
 * from within `bun test` so a single `bun test` command exercises both
 * TypeScript and bash tests.
 *
 * Delegates to `bash tests/bash/run.sh` which runs unit/ then sandboxed/
 * test files. The bun:test harness asserts on the exit code.
 */
import { describe, it, expect } from 'bun:test';
import { execSync } from 'child_process';
import { join } from 'path';

const skipLong = !process.env.RUN_LONG_TESTS;
const root = process.cwd();
const bashRun = join(root, 'tests', 'bash', 'run.sh');

describe('Bash test suite — unit only', () => {
    // it()'s own timeout must be >= the execSync timeout below, otherwise
    // bun:test's default 5s per-test timeout kills the assertion before the
    // child process can finish (regardless of how much time execSync itself
    // was given).
    it(
        'runs unit/ tests and exits 0',
        () => {
            const output = execSync(`bash ${bashRun} unit`, {
                encoding: 'utf-8',
                timeout: 30_000,
            });
            expect(output).toMatch(/0 failed/);
        },
        { timeout: 30_000 },
    );
});

describe('Bash test suite — sandboxed only', () => {
    // Same requirement as above — 300s bun:test timeout to match the 300s
    // execSync timeout for the heavier bwrap-sandboxed suite.
    it.skipIf(skipLong)(
        'runs sandboxed/ tests and exits 0',
        () => {
            const output = execSync(`bash ${bashRun} sandboxed`, {
                encoding: 'utf-8',
                timeout: 300_000, // 5 min — heavy sandboxed tests
            });
            expect(output).toMatch(/0 failed/);
        },
        { timeout: 300_000 },
    );
});
