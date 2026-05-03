#!/usr/bin/env bash

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build"
RUN_SH="${SCRIPT_DIR}/run.sh"
PLOT_PY="${SCRIPT_DIR}/plot.py"

TESTS="${TESTS:-all}"
CSV_FILE="${CSV_FILE:-${BUILD_DIR}/run_all.csv}"
LOG_DIR="${LOG_DIR:-.}"
DO_CLEAN_FLAG="${DO_CLEAN_FLAG:-}"

CONFIGS=(
  # "-m 32  -n 32  -k 256  -s 0 -a 0.0  -b 0.0  -T 32 -i fp8  -o fp32 -w 2 -M 2 -N 16 -Q 4 -p 2 -t sgemm_tcu_op -l run1.log"
  # "-m 32  -n 32  -k 256  -s 0 -a 0.0  -b 0.0  -T 32 -i fp16 -o fp32 -w 2 -M 2 -N 16 -Q 4 -p 2 -t sgemm_tcu_op -l run2.log"
  # "-m 32  -n 32  -k 256  -s 0 -a 0.0  -b 0.0  -T 32 -i fp8  -o fp32 -w 2 -M 2 -N 16 -Q 4 -p 2 -t sgemm_tcu,sgemm_tcu_sp -l run1.log"
  # "-m 32  -n 32  -k 256  -s 0 -a 0.0  -b 0.0  -T 32 -i fp16 -o fp32 -w 2 -M 2 -N 16 -Q 4 -p 2 -t sgemm_tcu,sgemm_tcu_sp -l run2.log"
  
  "-m 64  -n 64  -k 512  -s 0 -a 0.0  -b 0.0  -T 32 -i fp8  -o fp32 -w 2 -M 2 -N 16 -Q 4 -p 2 -t sgemm_tcu_op -l run3.log"
  "-m 64  -n 64  -k 512  -s 2 -a 0.2  -b 0.2  -T 32 -i fp8  -o fp32 -w 2 -M 2 -N 16 -Q 4 -p 2 -t sgemm_tcu_op -l run4.log"
  "-m 64  -n 64  -k 512  -s 2 -a 0.2  -b 0.6  -T 32 -i fp8  -o fp32 -w 2 -M 2 -N 16 -Q 4 -p 2 -t sgemm_tcu_op -l run5.log"
  "-m 64  -n 64  -k 512  -s 2 -a 0.2  -b 0.9  -T 32 -i fp8  -o fp32 -w 2 -M 2 -N 16 -Q 4 -p 2 -t sgemm_tcu_op -l run6.log"
  "-m 64  -n 64  -k 512  -s 2 -a 0.6  -b 0.2  -T 32 -i fp8  -o fp32 -w 2 -M 2 -N 16 -Q 4 -p 2 -t sgemm_tcu_op -l run7.log"
  "-m 64  -n 64  -k 512  -s 2 -a 0.6  -b 0.6  -T 32 -i fp8  -o fp32 -w 2 -M 2 -N 16 -Q 4 -p 2 -t sgemm_tcu_op -l run8.log"
  "-m 64  -n 64  -k 512  -s 2 -a 0.6  -b 0.9  -T 32 -i fp8  -o fp32 -w 2 -M 2 -N 16 -Q 4 -p 2 -t sgemm_tcu_op -l run9.log"
  "-m 64  -n 64  -k 512  -s 2 -a 0.9  -b 0.2  -T 32 -i fp8  -o fp32 -w 2 -M 2 -N 16 -Q 4 -p 2 -t sgemm_tcu_op -l run10.log"
  "-m 64  -n 64  -k 512  -s 2 -a 0.9  -b 0.6  -T 32 -i fp8  -o fp32 -w 2 -M 2 -N 16 -Q 4 -p 2 -t sgemm_tcu_op -l run11.log"
  "-m 64  -n 64  -k 512  -s 2 -a 0.9  -b 0.9  -T 32 -i fp8  -o fp32 -w 2 -M 2 -N 16 -Q 4 -p 2 -t sgemm_tcu_op -l run12.log"

  # "-m 64  -n 64  -k 64  -s 2 -a 0.5  -b 0.5  -T 32 -i fp16 -o fp32 -w 2 -M 2 -N 16 -Q 2"
  # "-m 64  -n 64  -k 128 -s 2 -a 0.5  -b 0.5  -T 32 -i fp16 -o fp32 -w 2 -M 2 -N 16 -Q 4"
  # "-m 128 -n 64  -k 64  -s 2 -a 0.5  -b 0.5  -T 32 -i fp16 -o fp32 -w 2 -M 4 -N 8  -Q 2"
  # "-m 64  -n 128 -k 64  -s 2 -a 0.5  -b 0.5  -T 32 -i fp16 -o fp32 -w 2 -M 4 -N 8  -Q 4"
  # "-m 128 -n 128 -k 64  -s 2 -a 0.5  -b 0.5  -T 32 -i fp16 -o fp32 -w 2 -M 2 -N 16 -Q 8"
  # "-m 32  -n 32  -k 32  -s 0 -a 0.0  -b 0.0  -T 32 -i fp16 -o fp32 -w 2 -M 2 -N 16 -Q 2"
  # "-m 64  -n 64  -k 64  -s 1 -a 0.25 -b 0.5  -T 32 -i fp16 -o fp32 -w 2 -M 2 -N 16 -Q 4"
  # "-m 64  -n 64  -k 64  -s 2 -a 0.25 -b 0.25 -T 32 -i fp16 -o fp32 -w 2 -M 4 -N 8  -Q 8"
)

usage() {
  cat <<'EOF'
Usage: ./custom/run_all.sh

Runs custom/run.sh over a predefined configuration sweep.

Environment overrides:
  TESTS          Default test list passed to run.sh -t when a config omits -t (default: all)
  CSV_FILE       Manifest CSV path (default: build/run_all.csv)
  LOG_DIR        Default log directory relative to build/ when a config omits -l (default: .)
  DO_CLEAN_FLAG  Set to -C to skip make clean in run.sh

Generated logs:
  sgemm_tcu_op -> run1.log, run2.log, ...
  sgemm_tcu    -> run1_ip.log, run2_ip.log, ...
  sgemm_tcu_sp -> run1_ip_sp.log, run2_ip_sp.log, ...
EOF
}

csv_escape() {
  local value="$1"
  value="${value//\"/\"\"}"
  printf '"%s"' "${value}"
}

make_log_path() {
  local run_index="$1"

  if [[ -n "${LOG_DIR}" && "${LOG_DIR}" != "." ]]; then
    printf '%s/run%s.log' "${LOG_DIR}" "${run_index}"
  else
    printf 'run%s.log' "${run_index}"
  fi
}

make_suffixed_log() {
  local log_file="$1"
  local suffix="$2"
  local dir base stem ext

  dir="$(dirname "${log_file}")"
  base="$(basename "${log_file}")"
  if [[ "${base}" == *.* ]]; then
    stem="${base%.*}"
    ext=".${base##*.}"
  else
    stem="${base}"
    ext=""
  fi

  if [[ "${dir}" == "." ]]; then
    printf '%s%s%s\n' "${stem}" "${suffix}" "${ext}"
  else
    printf '%s/%s%s%s\n' "${dir}" "${stem}" "${suffix}" "${ext}"
  fi
}

test_enabled() {
  local test_list="$1"
  local needle="$2"
  local normalized="${test_list//,/ }"
  local test

  for test in ${normalized}; do
    case "${test}:${needle}" in
      all:*) return 0 ;;
      sgemm_tcu_op:sgemm_tcu_op|tcu_op:sgemm_tcu_op|op:sgemm_tcu_op) return 0 ;;
      sgemm_tcu:sgemm_tcu|ip:sgemm_tcu) return 0 ;;
      sgemm_tcu_sp:sgemm_tcu_sp|ip_sp:sgemm_tcu_sp) return 0 ;;
    esac
  done

  return 1
}

validate_tests() {
  local test_list="$1"
  local normalized="${test_list//,/ }"
  local test

  for test in ${normalized}; do
    case "${test}" in
      all|sgemm_tcu_op|tcu_op|op|sgemm_tcu|ip|sgemm_tcu_sp|ip_sp) ;;
      *)
        echo "Unsupported test: ${test}. Expected sgemm_tcu_op, sgemm_tcu, sgemm_tcu_sp, or all." >&2
        exit 1
        ;;
    esac
  done
}

append_csv_row() {
  local file_name="$1"
  local testbench="$2"
  local status="$3"
  local run_index="$4"
  local m="$5"
  local n="$6"
  local k="$7"
  local sparsity="$8"
  local a_sparsity="$9"
  local b_sparsity="${10}"
  local num_threads="${11}"
  local itype="${12}"
  local otype="${13}"
  local warps="${14}"
  local block_m="${15}"
  local block_n="${16}"
  local queue_depth="${17}"

  {
    csv_escape "${file_name}"; printf ','
    csv_escape "${testbench}"; printf ','
    printf '%s,%s,%s,%s,%s,%s,%s,' "${run_index}" "${m}" "${n}" "${k}" "${sparsity}" "${a_sparsity}" "${b_sparsity}"
    printf '%s,' "${num_threads}"
    csv_escape "${itype}"; printf ','
    csv_escape "${otype}"; printf ','
    printf '%s,%s,%s,%s,%s\n' "${warps}" "${block_m}" "${block_n}" "${queue_depth}" "${status}"
  } >> "${CSV_FILE}"
}

run_plot_for_log() {
  local log_file="$1"
  local log_path="${BUILD_DIR}/${log_file}"

  if [[ ! -f "${log_path}" ]]; then
    echo "Skipping stats, missing log: ${log_path}" >&2
    return 1
  fi

  echo "stats: ${log_file}"
  PYTHONDONTWRITEBYTECODE=1 python3 "${PLOT_PY}" "${log_path}"
}

load_config() {
  m=""
  n=""
  k=""
  sparsity=""
  a_sparsity=""
  b_sparsity=""
  num_threads=""
  itype=""
  otype=""
  warps=""
  block_m=""
  block_n=""
  queue_depth=""
  config_tests="${TESTS}"
  log_file=""
  config_args=()

  while [[ $# -gt 0 ]]; do
    case "$1" in
      -m) m="$2"; config_args+=("$1" "$2"); shift 2 ;;
      -n) n="$2"; config_args+=("$1" "$2"); shift 2 ;;
      -k) k="$2"; config_args+=("$1" "$2"); shift 2 ;;
      -s) sparsity="$2"; config_args+=("$1" "$2"); shift 2 ;;
      -a) a_sparsity="$2"; config_args+=("$1" "$2"); shift 2 ;;
      -b) b_sparsity="$2"; config_args+=("$1" "$2"); shift 2 ;;
      -T) num_threads="$2"; config_args+=("$1" "$2"); shift 2 ;;
      -i) itype="$2"; config_args+=("$1" "$2"); shift 2 ;;
      -o) otype="$2"; config_args+=("$1" "$2"); shift 2 ;;
      -w) warps="$2"; config_args+=("$1" "$2"); shift 2 ;;
      -M) block_m="$2"; config_args+=("$1" "$2"); shift 2 ;;
      -N) block_n="$2"; config_args+=("$1" "$2"); shift 2 ;;
      -Q) queue_depth="$2"; config_args+=("$1" "$2"); shift 2 ;;
      -p) config_args+=("$1" "$2"); shift 2 ;;
      -t) config_tests="$2"; config_args+=("$1" "$2"); shift 2 ;;
      -l) log_file="$2"; config_args+=("$1" "$2"); shift 2 ;;
      *)
        echo "Unsupported config option: $1" >&2
        exit 1
        ;;
    esac
  done

  validate_tests "${config_tests}"
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ ! -x "${RUN_SH}" ]]; then
  echo "Missing executable run script: ${RUN_SH}" >&2
  exit 1
fi

if [[ ! -f "${PLOT_PY}" ]]; then
  echo "Missing plot script: ${PLOT_PY}" >&2
  exit 1
fi

mkdir -p "${BUILD_DIR}"
mkdir -p "$(dirname "${CSV_FILE}")"
if [[ -n "${LOG_DIR}" && "${LOG_DIR}" != "." ]]; then
  mkdir -p "${BUILD_DIR}/${LOG_DIR}"
fi

printf 'file,testbench,run,m,n,k,sparsity,a_sparsity,b_sparsity,num_threads,itype,otype,warps,block_m,block_n,queue_depth,status\n' > "${CSV_FILE}"

overall_status=0
run_index=1

for config in "${CONFIGS[@]}"; do
  read -r -a raw_config_args <<< "${config}"
  load_config "${raw_config_args[@]}"
  if [[ -z "${log_file}" ]]; then
    log_file="$(make_log_path "${run_index}")"
    config_args+=("-l" "${log_file}")
  fi
  if [[ "$(dirname "${log_file}")" != "." ]]; then
    mkdir -p "${BUILD_DIR}/$(dirname "${log_file}")"
  fi

  echo "run${run_index}: tests=${config_tests} log=${log_file} m=${m} n=${n} k=${k} sparsity=${sparsity} a=${a_sparsity} b=${b_sparsity} threads=${num_threads} queue=${queue_depth}"

  "${RUN_SH}" \
    -t "${config_tests}" \
    "${config_args[@]}" \
    ${DO_CLEAN_FLAG}
  status=$?

  if [[ "${status}" -ne 0 ]]; then
    overall_status="${status}"
  fi

  if test_enabled "${config_tests}" "sgemm_tcu_op"; then
    append_csv_row "${log_file}" "sgemm_tcu_op" "${status}" "${run_index}" "${m}" "${n}" "${k}" "${sparsity}" "${a_sparsity}" "${b_sparsity}" "${num_threads}" "${itype}" "${otype}" "${warps}" "${block_m}" "${block_n}" "${queue_depth}"
    run_plot_for_log "${log_file}" || overall_status=$?
  fi
  if test_enabled "${config_tests}" "sgemm_tcu"; then
    test_log_file="$(make_suffixed_log "${log_file}" "_ip")"
    append_csv_row "${test_log_file}" "sgemm_tcu" "${status}" "${run_index}" "${m}" "${n}" "${k}" "${sparsity}" "${a_sparsity}" "${b_sparsity}" "${num_threads}" "${itype}" "${otype}" "${warps}" "${block_m}" "${block_n}" "${queue_depth}"
    run_plot_for_log "${test_log_file}" || overall_status=$?
  fi
  if test_enabled "${config_tests}" "sgemm_tcu_sp"; then
    test_log_file="$(make_suffixed_log "${log_file}" "_ip_sp")"
    append_csv_row "${test_log_file}" "sgemm_tcu_sp" "${status}" "${run_index}" "${m}" "${n}" "${k}" "${sparsity}" "${a_sparsity}" "${b_sparsity}" "${num_threads}" "${itype}" "${otype}" "${warps}" "${block_m}" "${block_n}" "${queue_depth}"
    run_plot_for_log "${test_log_file}" || overall_status=$?
  fi

  run_index=$((run_index + 1))
done

echo "Wrote ${CSV_FILE}"
exit "${overall_status}"
