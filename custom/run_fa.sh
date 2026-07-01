#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/build}"

N=1024
ENABLE_TIMING=0

usage() {
  cat <<'EOF'
Usage: ./custom/run_fa.sh [options]

Options:
  -n <value>   KV sequence length N (default: 1024, must be a multiple of 32)
  -t, --timing Enable FlashAttention phase timing counters
  -h           Show this help

Examples:
  ./custom/run_fa.sh
  ./custom/run_fa.sh -n 128
  ./custom/run_fa.sh -n 1024 -t
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -n)
      if [[ $# -lt 2 ]]; then
        echo "Missing argument for -n" >&2
        usage >&2
        exit 1
      fi
      N="$2"
      shift 2
      ;;
    --timing|-t)
      ENABLE_TIMING=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if ! [[ "${N}" =~ ^[0-9]+$ ]]; then
  echo "N must be a positive integer" >&2
  exit 1
fi

if (( N == 0 || N % 32 != 0 )); then
  echo "N must be a positive multiple of 32 for NUM_THREADS=32" >&2
  exit 1
fi

cd "${BUILD_DIR}"

APP_CONFIGS="-DNUM_THREADS=32 -DK_SEQUENCE_LENGTH=${N} -DEXT_TCU_ENABLE"
if (( ENABLE_TIMING )); then
  APP_CONFIGS="${APP_CONFIGS} -DFA_ENABLE_TIMING"
fi

echo "FlashAttention N=${N}, timing=$([[ ${ENABLE_TIMING} -eq 1 ]] && echo enabled || echo disabled)"
echo "CONFIGS=${APP_CONFIGS}"

make -C tests/regression/flashattention_tcu_virgo clean
CONFIGS="${APP_CONFIGS}" make -C tests/regression/flashattention_tcu_virgo

CONFIGS="-DEXT_TCU_ENABLE" ./ci/blackbox.sh --driver=simx --app=flashattention_tcu_virgo --threads=32
