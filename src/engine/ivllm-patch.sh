#!/bin/bash
# ivllm-patch.sh — apply (or revert) a local source
# patch against an already-installed vLLM venv, keyed by vLLM version.
#
# First step towards making patching a first-class ivllm concept rather than
# one-off scripts per fix (like hybrid-trtllm.sh) — a single generic
# apply/revert tool driven by ordinary unified-diff files under
# patches/<version>/. See design/active-issues.md for the
# background on why each patch exists, and this directory's README.md for
# per-patch notes.
#
# Usage:
#   ivllm-patch.sh -v <vllm-version> -p <patch-file> [-r]
#
# -v <vllm-version>  e.g. 0.25.1 — must already be installed via slurm-vllm-setup.sh
# -p <patch-file>    unified diff, git-style a/ b/ paths, rooted at the vLLM
#                 package's site-packages directory (e.g. a path of
#                 "vllm/model_executor/layers/foo.py" inside the diff),
#                 applied via `patch -p1`. Producing one:
#                   diff -u original.py patched.py \
#                     | sed 's|^--- original.py|--- a/vllm/path/to/original.py|;
#                            s|^+++ patched.py|+++ b/vllm/path/to/original.py|'
# -r    reverse-apply the patch instead of applying it
#
# Idempotent either way: applying an already-applied patch, or reverting one
# that isn't applied, is a clean no-op rather than an error. Requires
# $IVLLM_PROJECTDIR to already be set (same as the rest of the engine scripts)
# — sources utils.sh to reuse resolve_vllm_version_dir() rather than
# reconstructing the venv path here.
#
# Maintains a plain-text manifest of currently-applied patches at
# <vllm_version_dir>/.ivllm-patches-applied (one basename per line) — outside
# site-packages so it's clearly ivllm's own bookkeeping, not vLLM code. Any
# job-launch script can check this to announce in its own log whether the
# vLLM venv it's about to run is patched, and with what, without vLLM itself
# needing to know or care.

ivllm_patch_usage() {
    echo "Usage: $0 <vllm-version> <patch-file> [--revert]" >&2
    echo "  -v version  e.g. 0.25.1 — must already be installed via slurm-vllm-setup.sh" >&2
    echo "  -p <patch-file>    unified diff, git-style a/ b/ paths, rooted at site-packages" >&2
    echo "  -r    reverse-apply the patch instead of applying it" >&2
    exit 1
}

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$here/lib/utils.sh"

export IVLLM_VERSION=""
export IVLLM_PATCH=""
export IVLLM_REVERT=0
export LOG=""

OPTIND=1
while getopts "v:p:rh" opt; do
    case $opt in
        v) IVLLM_VERSION="$OPTARG" ;;
        p) IVLLM_PATCH="$OPTARG" ;;
        r) IVLLM_REVERT=1 ;;
        h) ivllm_setup_usage ;;
        \?) echo "Error: Invalid option -$OPTARG" >&2; ivllm_setup_usage ;;
        :)  echo "Error: Option -$OPTARG requires an argument" >&2; ivllm_setup_usage ;;
    esac
done

# Args: $1 - patch args array name (nameref); dry-run in the given direction.
# Returns: 0 if the dry run succeeds (i.e. that direction is currently clean
# to apply), 1 otherwise. Usage: dry_run_ok patch_args_array_name
dry_run_ok() {
    local patch="$1"
    local -n args_ref="$2"
    patch "${args_ref[@]}" --dry-run --force <"$patch" >/dev/null 2>&1
}

main() {
    local version="${1:-}"
    local patch_file="${2:-}"
    local revert="${3:-}"

    [[ -f "$patch_file" ]] || { echo "[patch] patch file not found: $patch_file" >&2; exit 1; }
    : "${IVLLM_PROJECTDIR:?IVLLM_PROJECTDIR must be set — run inside a job/setup context first}"

    local vllm_version_dir site_packages
    vllm_version_dir=$(resolve_vllm_version_dir "$version")
    site_packages="$vllm_version_dir/lib/python3.12/site-packages"

    [[ -d "$site_packages" ]] || {
        echo "[patch] no installed vLLM found at $site_packages" >&2
        exit 1
    }

    local forward_args=(-p1 -d "$site_packages")
    local reverse_args=(-p1 -d "$site_packages" -R)

    local currently_applied=1
    dry_run_ok "$patch_file" reverse_args && currently_applied=0   # reverse applies cleanly => patch is in
    local currently_clean=1
    dry_run_ok "$patch_file" forward_args && currently_clean=0     # forward applies cleanly => patch is out

    if (( currently_applied == 0 && currently_clean == 0 )); then
        echo "[patch] ERROR: both directions dry-run clean — patch file doesn't look like it targets this tree. Aborting." >&2
        exit 1
    fi

    local manifest="$vllm_version_dir/.ivllm-patches-applied"
    local patch_name
    patch_name="$(basename "$patch_file")"

    # Patches are named <descriptive-name>.v<vllm-version>.v<revision>.patch —
    # warn if a *different* revision of this same patch is already recorded
    # as applied, since reapplying a changed .patch file under a fresh
    # revision suffix over an older applied revision is exactly the scenario
    # the dry-run check below can't detect on its own (see README.md).
    local base_name="${patch_name%.v[0-9]*.patch}"
    if [[ -f "$manifest" ]]; then
        local other_revisions
        other_revisions=$(grep -F "${base_name}.v" "$manifest" 2>/dev/null | grep -vxF "$patch_name" || true)
        if [[ -n "$other_revisions" ]]; then
            echo "[patch] WARNING: a different revision of this same patch is already recorded as applied:" >&2
            echo "$other_revisions" | sed 's/^/[patch]   /' >&2
            echo "[patch] revert it first if it's still applied, to avoid a partial/mixed state." >&2
        fi
    fi

    if (( revert == 1 )); then
        if (( currently_applied != 0 )); then
            echo "[patch] $patch_name is not currently applied to vLLM $version — nothing to revert."
            exit 0
        fi
        patch "${reverse_args[@]}" <"$patch_file"
        [[ -f "$manifest" ]] && grep -vxF "$patch_name" "$manifest" > "$manifest.tmp" && mv "$manifest.tmp" "$manifest"
        echo "[patch] reverted $patch_name from vLLM $version at $site_packages"
    else
        if (( currently_clean != 0 )); then
            echo "[patch] $patch_name is already applied to vLLM $version — nothing to do."
            exit 0
        fi
        patch "${forward_args[@]}" <"$patch_file"
        grep -qxF "$patch_name" "$manifest" 2>/dev/null || echo "$patch_name" >> "$manifest"
        echo "[patch] applied $patch_name to vLLM $version at $site_packages"
    fi
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then main "$IVLLM_VERSION" "$IVLLM_PATCH" "$IVLLM_REVERT"; fi
