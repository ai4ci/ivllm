#!/bin/bash
# shellcheck disable=SC2016
# tests/bash/sandboxed/test-config.sh — vllm.yaml config-reading tests.
#
# Runs against the REAL yq binary (3.4.1 — matching what is installed on
# the HPC) rather than a shimmed/idealised stand-in. This is deliberate:
# running against the real binary is how issues 7-9 in design/issues.md
# were discovered in the first place — utils.sh's config-reading functions
# were originally written assuming yq v4's jq-style filter syntax, but the
# real installed yq is v3, which has a completely different (and, in one
# case, differently-ordered) CLI. Those issues have been fixed; the tests
# exercise the corrected behaviour and will catch any future regressions.

source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../lib" && pwd)/sandbox.sh"

_SETUP_MINIMAL='
    mkdir -p "$IVLLM_PROJECTDIR/engine/jobs/job1"
    cp /work/fixtures/minimal.yaml "$(resolve_job_config job1)"
'

_SETUP_WITH_ENV='
    mkdir -p "$IVLLM_PROJECTDIR/engine/jobs/job1"
    cp /work/fixtures/with-env.yaml "$(resolve_job_config job1)"
'

# --- get_job_config_setting: RED, see design/issues.md Issue 7 -------------
# yq v3's `read` subcommand takes (file, path) — utils.sh calls it
# (path, file), i.e. reversed, so every read silently returns "".

sandbox_run_test "get_job_config_setting_model" login "
    $_SETUP_MINIMAL
    value=\$(get_job_config_setting job1 \".model\")
    assert_eq \"\$value\" \"test-org/test-model-7b\" \"model\" || exit 1
"

sandbox_run_test "get_job_config_setting_idle_timeout" login "
    $_SETUP_MINIMAL
    value=\$(get_job_config_setting job1 \".idle-timeout\")
    assert_eq \"\$value\" \"30\" \"idle-timeout\" || exit 1
"

sandbox_run_test "get_job_config_setting_tensor_parallel" login "
    $_SETUP_MINIMAL
    value=\$(get_job_config_setting job1 \".tensor-parallel-size\")
    assert_eq \"\$value\" \"1\" \"tensor-parallel-size\" || exit 1
"

# --- resolve_stripped_job_config: RED, see design/issues.md Issue 8 --------
# utils.sh calls `yq 'del(.env, .nnodes, ...)' file` — yq v4 jq-filter
# syntax, unsupported by the installed yq v3 (which has a `delete`/`d`
# subcommand taking exactly one path per invocation, not a filter pipeline).

sandbox_run_test "resolve_stripped_job_config_strips_env_and_metadata" login '
    '"$_SETUP_WITH_ENV"'
    clean=$(resolve_stripped_job_config job1)
    assert_file_exists "$clean" || exit 1

    if yq r "$clean" "env" 2>/dev/null | grep -qv "^null$"; then
        echo "FAIL: env: block should have been stripped"
        exit 1
    fi
    if yq r "$clean" "min-vllm-version" 2>/dev/null | grep -qv "^null$"; then
        echo "FAIL: min-vllm-version should have been stripped"
        exit 1
    fi
    if yq r "$clean" "metadata" 2>/dev/null | grep -qv "^null$"; then
        echo "FAIL: metadata block should have been stripped"
        exit 1
    fi

    model=$(yq r "$clean" "model")
    assert_eq "$model" "test-org/test-model-with-env" "model (after stripping)" || exit 1
'

# --- get_job_config_exports: RED, see design/issues.md Issue 9 ------------
# Same root cause as Issue 8: `( .env // {} ) | to_entries | ...` is yq v4
# syntax, unsupported by the installed yq v3.

sandbox_run_test "get_job_config_exports_produces_export_lines" login '
    '"$_SETUP_WITH_ENV"'
    # get_job_config_exports pipes through jq'"'"'s @sh format, which
    # single-quotes values (safer than manual double-quoting for arbitrary
    # shell-unsafe content) — build the expected quote char without needing
    # a literal single-quote in this already-single-quoted test body.
    apos=$(printf "\047")
    exports=$(get_job_config_exports job1)
    assert_contains "$exports" "export FOO=${apos}bar${apos}" "exports" || exit 1
    assert_contains "$exports" "export BAZ=${apos}qux${apos}" "exports" || exit 1
'

sandbox_run_test "get_job_config_exports_empty_env_block" login "
    $_SETUP_MINIMAL
    exports=\$(get_job_config_exports job1)
    assert_eq \"\$exports\" \"\" \"exports (no env: block)\" || exit 1
"

exit "$FAIL"
