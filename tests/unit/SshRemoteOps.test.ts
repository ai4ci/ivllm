/**
 * tests/unit/SshRemoteOps.test.ts — Permission-handling tests for
 * SshRemoteOps.copyDirectory()/copyFile().
 *
 * These exercise the REAL `rsync` binary (no mocking) against a fake `ssh`
 * placed first on PATH. Real ssh joins every argv word after the target
 * host into one string and runs it via the remote shell — the fake `ssh`
 * below replicates exactly that, then runs the joined string locally with
 * `bash -c`, so rsync's `-e ssh` / `--rsync-path` machinery (and our own
 * runRemote/runRemoteSync command strings) behave identically to talking to
 * a real login node, just against a local directory instead of over the
 * network.
 *
 * We simulate "a contributor's machine with a normal umask" by setting the
 * test process's own umask to 0o022 (no group-write) and giving source
 * files/dirs mode 644/755 — i.e. exactly what a fresh `git clone` produces
 * under that umask, since git itself only tracks the executable bit, not
 * group-write.
 *
 * See design/active-issues.md for:
 *  - why the directory-level test below is currently `.skip`ped (a bun:test
 *    runtime limitation, not a bug in the production code — confirmed
 *    working via plain `bun run` and real Isambard usage)
 *  - the still-open file-level permission gap the second test documents
 */
import { describe, it, expect, beforeEach, afterEach } from 'bun:test';
import fs from 'fs';
import os from 'os';
import path from 'path';
import { SshRemoteOps } from '../../src/ops/SshRemoteOps';
import type { Credentials } from '../../src/types';

const FAKE_SSH_SCRIPT = `#!/bin/bash
# Test fixture — replicates real ssh's "join trailing argv words into one
# string, run via remote shell" behaviour, locally, so rsync/scp argv
# semantics can be tested without a network connection or real sshd.
args=("$@")
i=0
n=\${#args[@]}
remote_words=()
found_target=0
while (( i < n )); do
    arg="\${args[$i]}"
    if (( found_target )); then
        remote_words+=("$arg")
        (( i++ ))
        continue
    fi
    case "$arg" in
        -o|-l) (( i += 2 )) ;;          # -o KEY=VALUE / -l user (both take a value)
        -*) (( i++ )) ;;                # any other bare flag
        *) found_target=1; (( i++ )) ;; # first non-flag word is the target host
    esac
done
if [[ \${#remote_words[@]} -eq 0 ]]; then
    exit 0
fi
exec bash -c "\${remote_words[*]}"
`;

const CREDS: Credentials = {
    loginHost: 'fake-host',
    username: 'fake-user',
    projectDir: '/irrelevant',
};

describe('SshRemoteOps permission handling (real rsync/ssh argv, no network)', () => {
    let tmpRoot: string;
    let fakeSshDir: string;
    let originalPath: string | undefined;
    let originalUmask: number;

    beforeEach(() => {
        tmpRoot = fs.mkdtempSync(path.join(os.tmpdir(), 'ivllm-sshops-test-'));
        fakeSshDir = path.join(tmpRoot, 'bin');
        fs.mkdirSync(fakeSshDir);
        fs.writeFileSync(path.join(fakeSshDir, 'ssh'), FAKE_SSH_SCRIPT, {
            mode: 0o755,
        });

        originalPath = process.env.PATH;
        process.env.PATH = `${fakeSshDir}:${originalPath}`;

        // A normal contributor umask — no group-write by default.
        originalUmask = process.umask(0o022);
    });

    afterEach(() => {
        process.umask(originalUmask);
        process.env.PATH = originalPath;
        fs.rmSync(tmpRoot, { recursive: true, force: true });
    });

    // Historical note: an earlier version of this test relied solely on
    // `--rsync-path 'umask 002 && rsync'` for directory group-write, and had
    // to be `.skip`ped because bun:test's runtime interferes with umask
    // propagation to spawned child processes (see design/active-issues.md,
    // "bun:test cannot exercise umask-dependent child-process behaviour",
    // for the full investigation — confirmed as a test-runner artifact, not
    // a code bug, since the identical assertion passed via plain `bun run`
    // and matches real Isambard usage). copyDirectory now follows the rsync
    // with an explicit `chmod -R g+rwX` on the destination, which is
    // deterministic and umask-independent — so this test no longer needs
    // to be skipped.
    it('copyDirectory leaves the destination directory group-writable even under a restrictive local umask', async () => {
        const src = path.join(tmpRoot, 'src');
        const dest = path.join(tmpRoot, 'dest');
        fs.mkdirSync(src);
        fs.writeFileSync(path.join(src, 'file.txt'), 'hello', { mode: 0o644 });
        fs.mkdirSync(dest); // pre-existing remote project dir, as in real usage

        const ops = new SshRemoteOps(CREDS);
        await ops.copyDirectory(src, dest, 'up');

        const destSrcDir = path.join(dest, path.basename(src));
        const dirMode = fs.statSync(destSrcDir).mode & 0o777;

        expect(dirMode & 0o020).toBe(0o020);
    });

    // Regression test for the file-level gap (see design/active-issues.md,
    // "copyDirectory doesn't make copied FILES group-writable when the
    // source lacks the bit"): without --perms, rsync (like cp without -p)
    // uses a copied file's own SOURCE mode bits as its base request — a
    // umask can only strip bits from that, never add ones the source
    // lacked. So a file that is locally 644 (no group-write — exactly what
    // `git clone` produces under a normal 022 umask, since git only tracks
    // the executable bit) used to stay 644 on the destination no matter
    // what `--rsync-path 'umask 002 && rsync'` requested. The follow-up
    // `chmod -R g+rwX` in copyDirectory now closes this deterministically,
    // independent of source mode or umask.
    it('copyDirectory leaves destination FILES group-writable even when the source file lacks the bit', async () => {
        const src = path.join(tmpRoot, 'src');
        const dest = path.join(tmpRoot, 'dest');
        fs.mkdirSync(src);
        fs.writeFileSync(path.join(src, 'file.txt'), 'hello', { mode: 0o644 });
        fs.mkdirSync(dest);

        const ops = new SshRemoteOps(CREDS);
        await ops.copyDirectory(src, dest, 'up');

        const destSrcDir = path.join(dest, path.basename(src));
        const fileMode =
            fs.statSync(path.join(destSrcDir, 'file.txt')).mode & 0o777;

        expect(fileMode & 0o020).toBe(0o020); // group-writable file
    });

    it('copyFile writes a group-writable destination file even under a restrictive local umask', async () => {
        const localFile = path.join(tmpRoot, 'local.yaml');
        const remoteFile = path.join(tmpRoot, 'remote.yaml');
        // copyFile streams bytes through `umask 002 && cat > remoteFile`
        // rather than preserving source permissions, so — unlike
        // copyDirectory's rsync-based file copies — it isn't subject to
        // the source-mode gap above.
        fs.writeFileSync(localFile, 'model: test\n', { mode: 0o644 });

        const ops = new SshRemoteOps(CREDS);
        await ops.copyFile(localFile, remoteFile);

        const mode = fs.statSync(remoteFile).mode & 0o777;
        expect(mode & 0o020).toBe(0o020); // group-writable
        expect(fs.readFileSync(remoteFile, 'utf8')).toBe('model: test\n');
    });
});
