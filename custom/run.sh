#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build"

M=64
N=64
K=64
SPARSITY=2
A_SPARSITY=0.5
B_SPARSITY=0.5
NUM_THREADS=32
ITYPE=fp16
OTYPE=fp32
WARPS=2
BLOCK_M=2
BLOCK_N=16
XBAR_QUEUE_DEPTH=2
PERF_CLASS=""
LOG_FILE=run.log
DO_CLEAN=1
EXTRA_BUILD_CONFIGS=""
EXTRA_APP_ARGS=()

usage() {
  cat <<'EOF'
Usage: ./custom/run.sh [options] [-- extra app args]

Options:
  -m <value>   Matrix M dimension (default: 64)
  -n <value>   Matrix N dimension (default: 64)
  -k <value>   Matrix K dimension (default: 64)
  -s <value>   Sparsity mode: 0, 1, or 2 (default: 2)
  -a <value>   A sparsity ratio in [0.0, 1.0] (default: 0.5)
  -b <value>   B sparsity ratio in [0.0, 1.0] (default: 0.5)
  -t <value>   NUM_THREADS compile-time constant (default: 32)
  -i <value>   ITYPE compile-time type (default: fp16)
  -o <value>   OTYPE compile-time type (default: fp32)
  -w <value>   blackbox --warps value (default: 1)
  -M <value>   FEOP BLOCK_M override (default: 2)
  -N <value>   FEOP BLOCK_N override (default: 16)
  -Q <value>   FEOP XBAR_QUEUE_DEPTH override (default: 2)
  -p <value>   blackbox --perf class (default: disabled)
  -l <path>    blackbox log file relative to build/ (default: run.log)
  -C           Skip make clean
  -h           Show this help

Examples:
  ./custom/run.sh
  ./custom/run.sh -m 128 -n 64 -k 64 -s 1 -a 0.25 -b 0.5
  ./custom/run.sh -m 32 -n 32 -k 64 -s 0 -- -x
EOF
}

while getopts ":m:n:k:s:a:b:t:i:o:w:M:N:Q:p:l:Ch" opt; do
  case "${opt}" in
    m) M="${OPTARG}" ;;
    n) N="${OPTARG}" ;;
    k) K="${OPTARG}" ;;
    s) SPARSITY="${OPTARG}" ;;
    a) A_SPARSITY="${OPTARG}" ;;
    b) B_SPARSITY="${OPTARG}" ;;
    t) NUM_THREADS="${OPTARG}" ;;
    i) ITYPE="${OPTARG}" ;;
    o) OTYPE="${OPTARG}" ;;
    w) WARPS="${OPTARG}" ;;
    M) BLOCK_M="${OPTARG}" ;;
    N) BLOCK_N="${OPTARG}" ;;
    Q) XBAR_QUEUE_DEPTH="${OPTARG}" ;;
    p) PERF_CLASS="${OPTARG}" ;;
    l) LOG_FILE="${OPTARG}" ;;
    C) DO_CLEAN=0 ;;
    h)
      usage
      exit 0
      ;;
    :)
      echo "Missing argument for -${OPTARG}" >&2
      usage >&2
      exit 1
      ;;
    \?)
      echo "Unknown option: -${OPTARG}" >&2
      usage >&2
      exit 1
      ;;
  esac
done

shift $((OPTIND - 1))

if [[ $# -gt 0 ]]; then
  if [[ "$1" == "--" ]]; then
    shift
  fi
  EXTRA_APP_ARGS=("$@")
fi

case "${SPARSITY}" in
  0|1|2) ;;
  *)
    echo "Unsupported sparsity mode: ${SPARSITY}. Expected 0, 1, or 2." >&2
    exit 1
    ;;
esac

if (( BLOCK_M <= 0 || BLOCK_N <= 0 || XBAR_QUEUE_DEPTH <= 0 )); then
  echo "BLOCK_M, BLOCK_N, and XBAR_QUEUE_DEPTH must be positive integers." >&2
  exit 1
fi

if (( 32 % BLOCK_M != 0 || 32 % BLOCK_N != 0 )); then
  echo "BLOCK_M and BLOCK_N must divide the 32x32 FEOP tile dimensions." >&2
  exit 1
fi

if (( (BLOCK_M & (BLOCK_M - 1)) != 0 || (BLOCK_N & (BLOCK_N - 1)) != 0 )); then
  echo "BLOCK_M and BLOCK_N must be powers of two." >&2
  exit 1
fi

BUILD_CONFIGS=(
  "-DTCU_OP"
  "-DNUM_THREADS=${NUM_THREADS}"
  "-DITYPE=${ITYPE}"
  "-DOTYPE=${OTYPE}"
  "-DSGEMM_CONST_M=${M}"
  "-DSGEMM_CONST_N=${N}"
  "-DSGEMM_CONST_K=${K}"
  "-DSGEMM_CONST_SPARSITY=${SPARSITY}"
  "-DSGEMM_CONST_A_SPARSITY=${A_SPARSITY}f"
  "-DSGEMM_CONST_B_SPARSITY=${B_SPARSITY}f"
  "-DTCU_FEOP_BLOCK_M_OVERRIDE=${BLOCK_M}"
  "-DTCU_FEOP_BLOCK_N_OVERRIDE=${BLOCK_N}"
  "-DTCU_FEOP_XBAR_QUEUE_DEPTH_OVERRIDE=${XBAR_QUEUE_DEPTH}"
)

RUNTIME_ARGS=(
  "-m${M}"
  "-n${N}"
  "-k${K}"
  "-s${SPARSITY}"
  "-a${A_SPARSITY}"
  "-b${B_SPARSITY}"
)

if [[ ${#EXTRA_APP_ARGS[@]} -gt 0 ]]; then
  RUNTIME_ARGS+=("${EXTRA_APP_ARGS[@]}")
fi

CONFIGS_STR="${BUILD_CONFIGS[*]}"
APP_ARGS_STR="${RUNTIME_ARGS[*]}"

cd "${BUILD_DIR}"

APP_BUILD_DIR="tests/regression/sgemm_tcu_op"

if [[ ! -d "${APP_BUILD_DIR}" ]]; then
  echo "Missing build wrapper: ${BUILD_DIR}/${APP_BUILD_DIR}" >&2
  echo "Expected out-of-tree build directory for sgemm_tcu_op is not present." >&2
  exit 1
fi

if [[ "${DO_CLEAN}" -eq 1 ]]; then
  make -C "${APP_BUILD_DIR}" clean
fi

echo "Build CONFIGS: ${CONFIGS_STR}"
CONFIGS="${CONFIGS_STR}" make -C "${APP_BUILD_DIR}"

echo "Runtime args: ${APP_ARGS_STR}"
BLACKBOX_CMD=(
  ./ci/blackbox.sh
  --driver=rtlsim
  --app=sgemm_tcu_op
  --warps="${WARPS}"
  --debug=1
  --log="${LOG_FILE}"
  --args="${APP_ARGS_STR}"
  --perf=6
)

if [[ -n "${PERF_CLASS}" ]]; then
  BLACKBOX_CMD+=(--perf="${PERF_CLASS}")
fi

CONFIGS="-DNUM_THREADS=${NUM_THREADS} -DEXT_TCU_ENABLE -DTCU_TYPE_DPI -DTCU_OP -DEXT_DXA_ENABLE -DTCU_FEOP_BLOCK_M_OVERRIDE=${BLOCK_M} -DTCU_FEOP_BLOCK_N_OVERRIDE=${BLOCK_N} -DTCU_FEOP_XBAR_QUEUE_DEPTH_OVERRIDE=${XBAR_QUEUE_DEPTH}" \
"${BLACKBOX_CMD[@]}"
