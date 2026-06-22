#!/usr/bin/env bash
#
# Run every XLEN=32 test locally, mirroring .github/workflows/ci.yml's `tests` job.
#
# CI v2 runs the pytest catalog (ci/testcases/*.yaml). This script drives the
# SAME catalog, one category at a time via `pytest -m "<category>"`, so it
# stays in sync automatically — categories are DISCOVERED from ci/testcases/,
# never hardcoded (that hardcoding was the drift we removed). xlen-inapplicable
# categories (e.g. hip, cupbop at 32) collect nothing and count as passed.
#
# Each category is wrapped in `timeout`. The script never stops on failure — it
# records every outcome and prints a summary at the end.
#
# Usage:
#   ./ci_xlen32.sh                     # run from build32/ — runs everything
#   ./ci_xlen32.sh tensor tensor_mx    # run a subset of categories
#   BUILD_DIR=/path/build32 ./ci_xlen32.sh     # run from elsewhere
#   TEST_TIMEOUT=1800 ./ci_xlen32.sh   # override per-category timeout (seconds)

set -u
set -o pipefail

XLEN=32
BUILD_DIR="${BUILD_DIR:-$PWD}"
cd "$BUILD_DIR" || { echo "ERROR: cannot cd to $BUILD_DIR"; exit 2; }

if [[ ! -d ci/testcases ]]; then
  echo "ERROR: ci/testcases not found under $BUILD_DIR — run from build${XLEN}/ (or set BUILD_DIR=)" >&2
  exit 2
fi

# Re-export the env vars that ci.yml's "Export tool env" step writes to
# $GITHUB_ENV/$GITHUB_PATH (needed by the sst/gem5/mpi categories).
VORTEX_HOME="${VORTEX_HOME:-$(dirname "$BUILD_DIR")}"
TOOLDIR="${TOOLDIR:-$VORTEX_HOME/tools}"
export MPIHOME="${MPIHOME:-$TOOLDIR/openmpi_install}"
export MPICC="${MPICC:-mpicc}"
export MPICXX="${MPICXX:-mpicxx}"
export SST_CORE_HOME="${SST_CORE_HOME:-$TOOLDIR/sst-install/sst-core}"
export SST_ELEMENTS_HOME="${SST_ELEMENTS_HOME:-$TOOLDIR/sst-install/sst-elements}"
export GEM5_HOME="${GEM5_HOME:-$TOOLDIR/gem5}"
export PATH="$MPIHOME/bin:$SST_CORE_HOME/bin:$SST_ELEMENTS_HOME/bin:$GEM5_HOME/build/X86:$PATH"

# Per-category timeout (seconds).
TEST_TIMEOUT="${TEST_TIMEOUT:-120}"

# Categories = the catalog itself (single source of truth).
if (( $# > 0 )); then
  SELECTED=("$@")
else
  mapfile -t SELECTED < <(ls ci/testcases/*.yaml | xargs -n1 basename | sed 's/\.yaml$//')
fi

LOG_DIR="${LOG_DIR:-$BUILD_DIR/ci_xlen${XLEN}_logs}"
mkdir -p "$LOG_DIR"

# Run one category through the pytest catalog under `timeout`; return its status.
run_cmd() {
  local cat="$1"
  local log="$LOG_DIR/${cat}.log"
  echo "  $ VX_XLEN=${XLEN} timeout ${TEST_TIMEOUT} python3 -m pytest ci -m \"${cat}\"" | tee -a "$log"
  VX_XLEN="${XLEN}" timeout "$TEST_TIMEOUT" python3 -m pytest ci -m "${cat}" --strict-markers -q \
    --junitxml="$LOG_DIR/junit-${cat}.xml" 2>&1 | tee -a "$log"
  return "${PIPESTATUS[0]}"
}

declare -a PASSED=() FAILED=() TIMED_OUT=()
record() {
  local name="$1" rc="$2"
  case "$rc" in
    0)   PASSED+=("$name") ;;
    5)   PASSED+=("$name") ;;   # pytest exit 5 = nothing collected (xlen-filtered) — not a failure
    124) TIMED_OUT+=("$name") ; FAILED+=("$name") ;;
    *)   FAILED+=("$name") ;;
  esac
}

for name in "${SELECTED[@]}"; do
  echo "=================================================================="
  echo "  [$(date +%H:%M:%S)] running: $name (XLEN=${XLEN})"
  echo "=================================================================="
  run_cmd "$name"
  record "$name" $?
done

echo
echo "=================================================================="
echo "  Summary  (per-category timeout: ${TEST_TIMEOUT}s)"
echo "=================================================================="
printf "  passed   (%d): %s\n" "${#PASSED[@]}" "${PASSED[*]:-<none>}"
printf "  failed   (%d): %s\n" "${#FAILED[@]}" "${FAILED[*]:-<none>}"
printf "  timed out(%d): %s\n" "${#TIMED_OUT[@]}" "${TIMED_OUT[*]:-<none>}"
echo "  logs:           $LOG_DIR"

(( ${#FAILED[@]} == 0 ))
