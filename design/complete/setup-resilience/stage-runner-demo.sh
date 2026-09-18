#!/bin/bash
# design/prototype/setup-resilience/stage-runner-demo.sh — PROTOTYPE, not
# production (see AGENTS.md: scripts in design/ are instructional examples,
# must be completely rewritten for production).
#
# Question this answers: can `src/engine/lib/slurm-vllm-setup.sh` guarantee it
# reaches its final `uv pip install --upgrade nvidia-nccl-cu12==2.30.4` pin
# when arbitrary optional stages fail — and if not, exactly what stops it?
#
# Why this matters: the NCCL pin (slurm-vllm-setup.sh:388-389) is the one
# stage in the install with a documented *correctness* rationale rather than a
# performance one — vLLM issue #46097 traces a multi-node TP collective-desync
# deadlock to the bundled nvidia-nccl-cu12 build, and 2.30.4 is verified clean
# here by design/prototype/nccl-probe.sh section [7]. It is also the last
# statement in a ~390-line script running `set -euo pipefail` (line 17) over
# eight third-party `git clone`/`make`/`python3 -m build` stages. Any one of
# them failing — network blip, upstream repo move, aarch64 compile error, a
# path that only exists on compute nodes — aborts the script and silently loses
# the pin, leaving a half-built venv that looks healthy.
#
# Four modes, all faked (`uv`/`git`/`cmake`/`make` are shell functions) — no
# network, no SLURM allocation, no venv, no /opt, nothing to clean up:
#
#   fragile   — today's shape. One optional stage fails, `set -e` aborts, the
#               pin never runs.
#   naive     — the obvious fix (run each stage with errexit suspended).
#               REACHES the pin, but silently reports a partially-failed stage
#               as having succeeded. Demonstrated, not theorised.
#   resilient — the wrapper PLUS a post-condition probe per stage. Pin always
#               runs, mandatory failures still abort, every partial failure is
#               actually caught and reported.
#   landmine-{1,2,3} — one statement each, in a real child shell, that aborts
#               the script even when its stage is meant to be optional.
#
# `bash stage-runner-demo.sh` runs the self-check across all of them and exits
# non-zero if any expectation is wrong. Run it before believing the README —
# the README's central claim (that the naive wrapper is insufficient) came out
# of this file failing, not out of reading the man page.

set -uo pipefail # deliberately NOT -e: every mode below is a fresh child
                 # process that sets its own options, so this file can drive
                 # the failures and observe them from outside.

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SELF="$HERE/stage-runner-demo.sh"

# $STATE is the fake filesystem-of-record for "is it installed?" — the
# equivalent of the venv's site-packages. $LOG records every tool call in
# order; order is the whole point, so the log IS the assertion.
STATE=${STATE:-$(mktemp -d)}
LOG=${LOG:-$(mktemp)}
BROKEN_PKG=${BROKEN_PKG:-uccl} # which package's `uv pip install` should fail
WITH_FLASHINFER=${WITH_FLASHINFER:-1}
VENV_DIR=${VENV_DIR:-$STATE/venv}
RDMA_DIR=${RDMA_DIR:-$STATE/rdma}
EMPTY_DIR=${EMPTY_DIR:-$STATE/no-h200-configs}
MANIFEST=$STATE/installed.txt
export LOG BROKEN_PKG WITH_FLASHINFER VENV_DIR RDMA_DIR EMPTY_DIR MANIFEST
mkdir -p "$EMPTY_DIR"

# ─────────────────────────────────────────────────────────────────────────────
# Fakes. Four tools, same call shapes as the real script.
#
# The fake `uv` matters more than the others: it keeps a manifest of what
# actually got installed, so `uv pip show <pkg>` means something. That is what
# makes post-condition probing testable rather than tautological — a stage can
# fail its post-condition even when it exits 0, exactly as it does for real.
# ─────────────────────────────────────────────────────────────────────────────
uv() {
  case "${1:-}" in
  pip)
    case "${2:-}" in
    show)
      grep -qxF "${3:-}" "$MANIFEST" 2>/dev/null && {
        echo "uv: pip show ${3} -> installed" >>"$LOG"
        return 0
      }
      echo "uv: pip show ${3} -> NOT installed" >>"$LOG"
      return 1
      ;;
    list)
      # `uv pip list --format=json | jq …` is how the real script discovers the
      # flashinfer version. Models the flashinfer-absent case.
      [[ ${WITH_FLASHINFER:-1} == 1 ]] && echo 9.9.9
      return 0
      ;;
    venv)
      echo "uv: venv ${3:-}" >>"$LOG" && return 0
      ;;
    install)
      shift 2
      local arg pkgs=() pkg
      for arg in "$@"; do
        case $arg in
        --*) continue ;;
        esac
        # `pkg==` with an empty version: what `uv pip install X=="$EMPTY"`
        # actually looks like on the command line, and uv rejects it.
        case $arg in
        *==)
          echo "uv: ERROR invalid version specifier in '$arg'" >>"$LOG"
          return 1
          ;;
        "$BROKEN_PKG"*)
          echo "uv: pip install $* -> FAIL" >>"$LOG"
          return 1
          ;;
        esac
        # Normalise to a distribution name: strip any version specifier, then
        # any trailing `-<version>…` wheel suffix, so `uv pip show uccl` can
        # find what `uv pip install uccl-0.1-py3-none-any.whl` put there.
        pkg="${arg%%[<>=!]*}"
        pkgs+=("${pkg%%-[0-9]*}")
      done
      for arg in "${pkgs[@]}"; do echo "$arg" >>"$MANIFEST"; done
      echo "uv: $* -> ok" >>"$LOG"
      return 0
      ;;
    esac
    ;;
  venv)
    echo "uv: venv ${1:-}" >>"$LOG" && return 0
    ;;
  esac
  echo "uv: $*" >>"$LOG"
  return 0
}

git() { echo "git: $*" >>"$LOG"; return 0; }
cmake() { echo "cmake: $*" >>"$LOG"; return 0; }
make() { echo "make: $*" >>"$LOG"; return 0; }
export -f uv git cmake make

# ─────────────────────────────────────────────────────────────────────────────
# Stages. Same order and same mandatory/optional split as the real script,
# stripped to what decides whether the pin is reached.
#
# Each optional stage has a matching `_check` — a *post-condition*, i.e. an
# independent statement that the thing the stage exists to produce now exists.
# In production these are not new code: the script already opens every optional
# block with `uv pip show <pkg>` as its "already installed, skip" guard, so the
# same expression reused afterwards is both the success test and the retry
# test. `fused_moe_check` is a file test because that stage produces a file,
# not a package.
# ─────────────────────────────────────────────────────────────────────────────

# MANDATORY — NVHPC SDK, then the venv + vLLM wheels. Without these the pin is
# pointless, so a failure here must abort the whole script.
stage_nvhpc() { echo "stage: nvhpc" >>"$LOG"; }
stage_venv() {
  uv venv "$VENV_DIR"
  uv pip install "vllm==$1" "ray[default]"
}
venv_check() { uv pip show vllm; }

# OPTIONAL — third-party kernels/backends that vLLM runs fine without.
stage_flashinfer() {
  # real lines 154+160: version discovered by query, then pinned exactly
  local flashinfer
  flashinfer=$(uv pip list --format=json)
  uv pip install "flashinfer-jit-cache==$flashinfer" --index-url https://flashinfer.ai/whl/cu129
}
flashinfer_check() { uv pip show flashinfer-jit-cache; }

stage_deepgemm() {
  # real line 169: the ref-extract pipeline runs BEFORE any build, and its
  # failure is legitimately expected when a release doesn't pin a DeepGEMM ref.
  local ref
  ref=$(printf 'OTHER="x"\n' | grep "DEEPGEMM_GIT_REF=" | sed -n '1p;s@.*="\(.*\)".*@\1@p') || true
  if [[ -z ${ref:-} ]]; then
    echo "WARNING: no DeepGEMM git reference found — not applicable" >>"$LOG"
    return 77 # conventional SKIP: not-a-failure, don't post-condition
  fi
  git clone --recursive --shallow-submodules https://github.com/deepseek-ai/DeepGEMM.git
  uv pip install --no-build-isolation .
}
deepgemm_check() { uv pip show deep_gemm; }

stage_uccl() {
  git clone --recursive --shallow-submodules -b main https://github.com/uccl-project/uccl.git
  make -j
  uv pip install --no-build-isolation "uccl-0.1-py3-none-any.whl"
  uv pip install --no-build-isolation deep_ep_wrapper # never reached: uccl died
}
uccl_check() { uv pip show uccl; }

stage_fused_moe() {
  # real lines 138-145. `nullglob` is NOT set in the real script, so with no
  # H200 configs present the glob stays literal and `cp` fails — mid-body, with
  # the function's last command (`popd`) still succeeding afterwards.
  local f
  pushd "$EMPTY_DIR" >/dev/null
  for f in *device_name=NVIDIA_H200*; do cp "$f" "${f/H200/GH200}"; done
  popd >/dev/null
}
fused_moe_check() { compgen -G "$EMPTY_DIR/*device_name=NVIDIA_GH200*" >/dev/null; }

# MANDATORY, LAST: the reason this file exists.
stage_nccl_pin() {
  uv pip install --upgrade nvidia-nccl-cu12==2.30.4
  echo "PIN-RAN" >>"$LOG"
}

# Stage bodies must be visible to the `bash -c` children that run them. The
# production equivalent does not need this at all: re-executing the same script
# by path (`bash "$0" --stage uccl`) has them all in scope already.
export -f stage_nvhpc stage_venv stage_flashinfer stage_deepgemm stage_uccl \
  stage_fused_moe stage_nccl_pin

# ── mode: fragile — today's script shape ─────────────────────────────────────
fragile() {
  set -euo pipefail
  stage_nvhpc
  stage_venv 0.26.0
  stage_fused_moe
  stage_flashinfer
  stage_deepgemm
  stage_uccl
  stage_nccl_pin
}

# ── mode: naive — the wrapper everyone writes first ─────────────────────────
naive() {
  set -euo pipefail
  STAGES_FAILED=()
  # Subshell + `|| rc=$?` looks airtight: errexit is suspended for the whole
  # call, so the stage cannot abort the script. But "cannot abort" and "aborted
  # correctly" are different properties. Suspending errexit for the call also
  # suspends it *inside* the body — so a failure mid-body no longer stops the
  # body, and the stage's exit status is whatever its last command happened to
  # return. `set -e` re-asserted inside the subshell does NOT help (POSIX: the
  # -e setting is ignored in a subshell created to run a command in a guarded
  # context), and neither does backgrounding + `wait` (measured to return 0 for
  # a failed job). See README §"why the obvious wrapper is not enough".
  naive_stage() {
    local name=$1 rc=0
    shift
    echo "--- optional stage: $name" >>"$LOG"
    ("$@") || rc=$?
    # NB: an `if`, not `((rc != 0)) && { … }` — the latter returns 1 when rc is
    # 0, and as the function's last command that makes the *call* fail, which
    # under a live `set -e` aborts the script. This harness wrote that bug, hit
    # it, and is documenting it.
    if ((rc != 0)); then
      STAGES_FAILED+=("$name")
      echo "WARNING: '$name' failed (rc=$rc)" >>"$LOG"
    fi
  }
  stage_venv 0.26.0
  naive_stage fused-moe stage_fused_moe
  naive_stage flashinfer stage_flashinfer
  naive_stage deepgemm stage_deepgemm
  naive_stage uccl stage_uccl
  stage_nccl_pin
  echo "naive report: failed=[${STAGES_FAILED[*]:-none}]" >>"$LOG"
}

# ── mode: resilient — wrapper + post-condition ───────────────────────────────
STAGES_FAILED=()
STAGES_SKIPPED=()

# Optional stage: `stage <name> <body> <check>`.
#
# Two independent mechanisms, and the demo exists to show you need both:
#
#  1. CONTAINMENT — the body runs in a *fresh shell process*, so it has real
#     errexit/pipefail/nounset of its own (a failure anywhere in it aborts it)
#     while being unable to abort the parent. A fresh process is also the only
#     thing that contains the real stages' `cd`/`pushd`/`export` (the UCCL and
#     NIXL stages export CPATH, CPLUS_INCLUDE_PATH, LIBRARY_PATH,
#     LD_LIBRARY_PATH, TORCH_CUDA_ARCH_LIST, UCCL_STAGING_DIR, and `pushd`
#     without a matching `popd` on their failure paths). It additionally
#     contains a `set -u` unbound-variable trip, which is *fatal* to a
#     non-interactive shell and is NOT excused by errexit being off.
#     Production variant: `bash "$0" --stage "$name"` re-executing this same
#     script, which re-sources common-env.sh (total env containment) and needs
#     no `export -f` of every stage body.
#  2. POST-CONDITION — necessary because containment alone only proves the
#     body stopped early; it says nothing about a body that ran to the end with
#     a broken thing inside it. The check asks the question that actually
#     matters ("is uccl installed?"), not "did the recipe exit cleanly?".
#
# NB: `$?` in the `else` branch of an `if` is 0, not the failure code — hence
# `local rc=0; … || rc=$?` rather than `if ! …`.
stage() {
  local name=$1 body=$2 check=$3 rc=0
  echo "--- optional stage: $name" >>"$LOG"
  bash -c 'set -euo pipefail; "$@"' _ "$body" || rc=$?
  if ((rc == 77)); then
    STAGES_SKIPPED+=("$name")
    echo "skipped: '$name' (not applicable)" >>"$LOG"
  elif ((rc != 0)); then
    STAGES_FAILED+=("$name")
    echo "WARNING: optional stage '$name' failed (rc=$rc) — continuing" >>"$LOG"
  elif ! "$check"; then
    # The case the naive wrapper misses entirely: body exited 0, artifact absent.
    STAGES_FAILED+=("$name")
    echo "WARNING: optional stage '$name' exited 0 but its post-condition failed — continuing" >>"$LOG"
  else
    echo "ok: '$name'" >>"$LOG"
  fi
}

# Mandatory stage: no wrapper, no subshell. errexit is live inside it, so a
# failure aborts immediately — a venv without vLLM in it must not be allowed to
# reach a success exit code.
required() {
  local name=$1
  shift
  echo "--- mandatory stage: $name" >>"$LOG"
  "$@"
}

resilient() {
  set -euo pipefail
  required nvhpc stage_nvhpc
  required vllm-wheels stage_venv 0.26.0
  if ! venv_check; then
    echo "FATAL: mandatory stage 'vllm-wheels' did not produce an importable vllm" >>"$LOG"
    exit 1
  fi

  stage fused-moe stage_fused_moe fused_moe_check
  stage flashinfer stage_flashinfer flashinfer_check
  stage deepgemm stage_deepgemm deepgemm_check
  stage uccl stage_uccl uccl_check

  # Unconditional, last, mandatory: reached whether or not anything above it
  # failed. That is the entire point of the exercise.
  stage_nccl_pin

  ((${#STAGES_SKIPPED[@]} > 0)) &&
    echo "skipped (not applicable): ${STAGES_SKIPPED[*]}" >>"$LOG"
  if ((${#STAGES_FAILED[@]} > 0)); then
    echo "setup INCOMPLETE: optional stages failed: ${STAGES_FAILED[*]}" >>"$LOG"
    echo "the venv is usable but those kernels/backends are absent" >>"$LOG"
    exit 1
  fi
  echo "setup complete" >>"$LOG"
}

# ── modes: the three landmines, one per child shell ─────────────────────────
# Each runs in a real child with the real script's options, because the point
# is about how `set -e` interacts with the surrounding *script*, which cannot be
# demonstrated from inside a guarded subshell of the same script. Printing
# REACHED means the statement was survived.

landmine1() { # real line 169 — ref-extract pipeline
  set -euo pipefail
  local ref
  ref=$(printf 'OTHER="x"\n' | grep "DEEPGEMM_GIT_REF=" | head -n 1 | sed 's/.*="\(.*\)".*/\1/')
  echo "REACHED"
  [[ -z ${ref:-} ]] && echo "would have skipped the stage"
}
landmine2() { # real lines 154+160 — empty discovered version
  set -euo pipefail
  local flashinfer
  WITH_FLASHINFER=0 flashinfer=$(uv pip list --format=json)
  uv pip install "flashinfer-jit-cache==$flashinfer"
  echo "REACHED"
}
landmine3() { # real lines 138-145 — literal glob with no nullglob
  set -euo pipefail
  stage_fused_moe
  echo "REACHED"
}

# ─────────────────────────────────────────────────────────────────────────────
# Self-check.
# ─────────────────────────────────────────────────────────────────────────────
checks=0
notok=0

expect() { # expect <description> <0-for-pass>
  checks=$((checks + 1))
  if [[ $2 == 0 ]]; then
    echo "ok   — $1"
  else
    echo "FAIL — $1"
    notok=$((notok + 1))
  fi
}

# run_mode <mode> → RC, OUT
run_mode() {
  : >"$LOG"
  : >"$MANIFEST"
  OUT=$(BROKEN_PKG="$BROKEN_PKG" WITH_FLASHINFER="$WITH_FLASHINFER" STATE="$STATE" \
    bash "$SELF" "$1" 2>&1)
  RC=$?
}

selfcheck() {
  # [1] the bug as it stands today.
  BROKEN_PKG=uccl
  WITH_FLASHINFER=1
  run_mode fragile
  [[ $RC != 0 ]] && ! grep -q PIN-RAN "$LOG"
  expect "fragile: optional 'uccl' failure aborts before the pin (rc=$RC)" $?
  grep -q "vllm==0.26.0" "$LOG"
  expect "fragile: mandatory vllm stage had succeeded — the abort is an OPTIONAL stage's fault" $?
  [[ $OUT == *"cp: cannot stat"* ]]
  expect "fragile: it actually died one stage EARLIER, at the fused-MoE glob" $?

  # [2] the naive wrapper reaches the pin but lies about what failed.
  run_mode naive
  grep -q PIN-RAN "$LOG"
  expect "naive: pin reached despite optional failures" $?
  grep -q "naive report: failed=\[deepgemm\]" "$LOG"
  expect "naive: of 4 optional stages it lists exactly one — and it is the false one" $?
  ! fused_moe_check
  expect "naive: FALSE NEGATIVE 1 — the fused-MoE artifact is missing, not reported" $?
  ! uv pip show uccl
  expect "naive: FALSE NEGATIVE 2 — uccl is not installed either, because the command after the one that failed succeeded" $?
  grep -q "'deepgemm' failed (rc=77)" "$LOG"
  expect "naive: FALSE POSITIVE — legitimately-absent DeepGEMM reported as a failure" $?

  # [3] wrapper + post-condition.
  run_mode resilient
  local pin_line stages_line
  pin_line=$(grep -n "nvidia-nccl-cu12==2.30.4 -> ok" "$LOG" | tail -1 | cut -d: -f1)
  stages_line=$(grep -n -- "--- optional stage" "$LOG" | tail -1 | cut -d: -f1)
  [[ -n $pin_line && $pin_line -gt $stages_line ]]
  expect "resilient: NCCL pin ran, after every optional stage had been attempted" $?
  grep -q "optional stages failed: fused-moe uccl" "$LOG"
  expect "resilient: BOTH the mid-body failure and the silent partial failure recorded" $?
  grep -q "skipped (not applicable): deepgemm" "$LOG"
  expect "resilient: a legitimately-absent DeepGEMM is SKIPPED, not failed" $?
  ! grep -q "^uv: flashinfer-jit-cache -> NOT" "$LOG" && grep -q "ok: 'flashinfer'" "$LOG"
  expect "resilient: a stage that genuinely succeeded is not flagged" $?

  # [3b] a mandatory failure must still stop everything.
  BROKEN_PKG=vllm
  run_mode resilient
  [[ $RC != 0 ]] && ! grep -q PIN-RAN "$LOG"
  expect "resilient: mandatory 'vllm-wheels' failure still aborts before the pin" $?

  # [3c] the clean case — needs an H200 config to exist, or fused-MoE can
  # never pass its post-condition.
  BROKEN_PKG=__nothing__
  touch "$EMPTY_DIR/someconfig-device_name=NVIDIA_H200.json"
  run_mode resilient
  [[ $RC == 0 ]] && grep -q "setup complete" "$LOG" &&
    grep -q "skipped (not applicable): deepgemm" "$LOG"
  expect "resilient: clean run exits 0 — a SKIP alone is not a failure" $?
  compgen -G "$EMPTY_DIR/*NVIDIA_GH200*" >/dev/null
  expect "resilient: the fused-MoE GH200 configs really were produced this time" $?
  rm -f "$EMPTY_DIR"/*NVIDIA_H200* "$EMPTY_DIR"/*NVIDIA_GH200*

  # [3d] the flashinfer-absent case, i.e. landmine 2 reached from a real run.
  BROKEN_PKG=__nothing__
  WITH_FLASHINFER=0
  run_mode resilient
  grep -q "optional stages failed:.*flashinfer" "$LOG" && grep -q PIN-RAN "$LOG"
  expect "resilient: empty flashinfer version is recorded, pin still runs" $?
  WITH_FLASHINFER=1

  # [4] the landmines, individually.
  local n
  for n in 1 2 3; do
    run_mode "landmine-$n"
    [[ $RC != 0 && $OUT != *REACHED* ]]
    expect "landmine-$n: the statement aborts a real script (rc=$RC, guard never reached)" $?
  done

  echo
  echo "$((checks - notok))/$checks checks passed"
  [[ $notok == 0 ]]
}

case ${1:-selfcheck} in
  fragile) fragile ;;
  naive) naive ;;
  resilient) resilient ;;
  landmine-1) landmine1 ;;
  landmine-2) landmine2 ;;
  landmine-3) landmine3 ;;
  selfcheck) selfcheck ;;
  *)
    echo "usage: $(basename "$0") [selfcheck|fragile|naive|resilient|landmine-1|landmine-2|landmine-3]" >&2
    exit 2
    ;;
esac
