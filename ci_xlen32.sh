#!/usr/bin/env bash
#
# Run every XLEN=32 test that .github/workflows/ci.yml's `tests` job runs.
#
# Mirrors the matrix exactly: same names, same per-name commands (the
# `regression` name expands to four sub-runs, as in ci.yml). `hip` is
# excluded because chipStar's hipcc is rv64-only.
#
# Each ci/regression.sh invocation is wrapped in `timeout 120`. The script
# never stops on failure — it records the outcome of every test and prints
# a summary at the end.
#
# Usage:
#   ./ci/ci_xlen32.sh                  # run from build32/ — runs everything
#   ./ci/ci_xlen32.sh tensor tensor_sp # run a subset
#   BUILD_DIR=/path/build32 ./ci/ci_xlen32.sh  # run from elsewhere
#   TEST_TIMEOUT=300 ./ci/ci_xlen32.sh # override per-command timeout (seconds)

set -u
set -o pipefail

BUILD_DIR="${BUILD_DIR:-$PWD}"
cd "$BUILD_DIR" || { echo "ERROR: cannot cd to $BUILD_DIR"; exit 2; }

if [[ ! -x ci/regression.sh ]]; then
  echo "ERROR: ci/regression.sh not found under $BUILD_DIR — run from build32/ (or set BUILD_DIR=)" >&2
  exit 2
fi

# Re-export the env vars that ci.yml's "Export tool env" step writes to
# $GITHUB_ENV/$GITHUB_PATH. Keep in sync with ci/{sst,gem5}_install.sh.in.
VORTEX_HOME="${VORTEX_HOME:-$(dirname "$BUILD_DIR")}"
TOOLDIR="${TOOLDIR:-$VORTEX_HOME/tools}"
export MPIHOME="${MPIHOME:-$TOOLDIR/openmpi_install}"
export MPICC="${MPICC:-mpicc}"
export MPICXX="${MPICXX:-mpicxx}"
export SST_CORE_HOME="${SST_CORE_HOME:-$TOOLDIR/sst-install/sst-core}"
export SST_ELEMENTS_HOME="${SST_ELEMENTS_HOME:-$TOOLDIR/sst-install/sst-elements}"
export GEM5_HOME="${GEM5_HOME:-$TOOLDIR/gem5}"
export PATH="$MPIHOME/bin:$SST_CORE_HOME/bin:$SST_ELEMENTS_HOME/bin:$GEM5_HOME/build/X86:$PATH"

# Per-command timeout (seconds). 120s mirrors the project rule.
TEST_TIMEOUT="${TEST_TIMEOUT:-120}"

# Matrix `name:` list from .github/workflows/ci.yml (XLEN=32 effective).
# `hip` is excluded per the `exclude` block (rv64-only).
# Order matches the ci.yml matrix so logs line up with what CI prints.
ALL_TESTS=(
  regression
  amo
  mpi
  dtm
  opencl
  vulkan
  cache
  config1
  config2
  debug
  scope
  stress
  synthesis
  vm
  rvc
  cupbop
  tensor
  tensor_sp
  gem5
)

# Optional positional args narrow the set to just those names.
if (( $# > 0 )); then
  SELECTED=("$@")
else
  SELECTED=("${ALL_TESTS[@]}")
fi

LOG_DIR="${LOG_DIR:-$BUILD_DIR/ci_xlen32_logs}"
mkdir -p "$LOG_DIR"

# Run one ci/regression.sh invocation under `timeout`; return its status.
# Distinguishes timeout (exit 124) from a real failure in the summary.
run_cmd() {
  local label="$1"; shift
  local log="$LOG_DIR/${label}.log"
  echo "  $ timeout ${TEST_TIMEOUT} ./ci/regression.sh $*" | tee -a "$log"
  timeout "$TEST_TIMEOUT" ./ci/regression.sh "$@" 2>&1 | tee -a "$log"
  return "${PIPESTATUS[0]}"
}

declare -a PASSED=() FAILED=() TIMED_OUT=()
record() {
  local name="$1" rc="$2"
  case "$rc" in
    0)   PASSED+=("$name") ;;
    124) TIMED_OUT+=("$name") ; FAILED+=("$name") ;;
    *)   FAILED+=("$name") ;;
  esac
}

for name in "${SELECTED[@]}"; do
  echo "=================================================================="
  echo "  [$(date +%H:%M:%S)] running: $name"
  echo "=================================================================="
  if [[ "$name" == "regression" ]]; then
    # ci.yml: the `regression` entry expands to four sub-runs; cap each.
    overall=0
    for sub in unittest riscv kernel regression; do
      run_cmd "regression-$sub" --"$sub"
      rc=$?
      (( rc != 0 )) && overall=$rc
    done
    record "$name" "$overall"
  else
    run_cmd "$name" --"$name"
    record "$name" $?
  fi
done

echo
echo "=================================================================="
echo "  Summary  (per-command timeout: ${TEST_TIMEOUT}s)"
echo "=================================================================="
printf "  passed   (%d): %s\n" "${#PASSED[@]}" "${PASSED[*]:-<none>}"
printf "  failed   (%d): %s\n" "${#FAILED[@]}" "${FAILED[*]:-<none>}"
printf "  timed out(%d): %s\n" "${#TIMED_OUT[@]}" "${TIMED_OUT[*]:-<none>}"
echo "  logs:           $LOG_DIR"

(( ${#FAILED[@]} == 0 ))
