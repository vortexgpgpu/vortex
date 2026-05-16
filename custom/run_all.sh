#!/usr/bin/env bash

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build"
RUN_SH="${SCRIPT_DIR}/run.sh"
PLOT_PY="${SCRIPT_DIR}/plot.py"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/run_all}"
QUEUE_PLOT_PY="${OUTPUT_DIR}/run_q/plot_queue_depth.py"

TESTS="${TESTS:-all}"
CSV_FILE="${CSV_FILE:-${OUTPUT_DIR}/run_all.csv}"
LOG_DIR="${LOG_DIR:-.}"
RUN_GROUP="${RUN_GROUP:-all}"
DO_CLEAN_FLAG="${DO_CLEAN_FLAG:-}"
JOBS="${JOBS:-1}"
WORK_BUILD_ROOT="${WORK_BUILD_ROOT:-${BUILD_DIR}/run_all_workers}"
MAKE_LOCK_FILE="${MAKE_LOCK_FILE:-${BUILD_DIR}/run_all.make.lock}"
CSV_HEADER="file,testbench,run,m,n,k,sparsity,a_sparsity,b_sparsity,num_threads,itype,otype,warps,block_m,block_n,queue_depth,status"
PLOT_PYTHON="${PLOT_PYTHON:-${SCRIPT_DIR}/venv/bin/python}"
if [[ ! -x "${PLOT_PYTHON}" ]]; then
  PLOT_PYTHON="python3"
fi

CONFIGS=(

  # TEST RUNS FOR DEBUG
  # "-m 64  -n 64  -k 512  -s 2 -a 0.6  -b 0.6  -T 32 -i fp8  -o fp32 -w 2 -M 2 -N 16 -Q 4 -p 2 -t sgemm_tcu_op -l run7.log"


  # QUEUE SIZE ANALYSIS: run_q_1.log through run_q_12.log
  # "-m 64  -n 64  -k 512  -s 2 -a 0.5  -b 0.5  -T 32 -i fp8  -o fp32 -w 2 -M 2 -N 16 -Q 1 -p 2 -t sgemm_tcu_op -l run_q_1.log"
  # "-m 64  -n 64  -k 512  -s 2 -a 0.5  -b 0.5  -T 32 -i fp8  -o fp32 -w 2 -M 2 -N 16 -Q 2 -p 2 -t sgemm_tcu_op -l run_q_2.log"
  # "-m 64  -n 64  -k 512  -s 2 -a 0.5  -b 0.5  -T 32 -i fp8  -o fp32 -w 2 -M 2 -N 16 -Q 4 -p 2 -t sgemm_tcu_op -l run_q_3.log"
  # "-m 64  -n 64  -k 512  -s 2 -a 0.5  -b 0.5  -T 32 -i fp8  -o fp32 -w 2 -M 2 -N 16 -Q 8 -p 2 -t sgemm_tcu_op -l run_q_4.log"

  # "-m 64  -n 64  -k 512  -s 2 -a 0.9  -b 0.9  -T 32 -i fp8  -o fp32 -w 2 -M 2 -N 16 -Q 1 -p 2 -t sgemm_tcu_op -l run_q_5.log"
  # "-m 64  -n 64  -k 512  -s 2 -a 0.9  -b 0.9  -T 32 -i fp8  -o fp32 -w 2 -M 2 -N 16 -Q 2 -p 2 -t sgemm_tcu_op -l run_q_6.log"
  # "-m 64  -n 64  -k 512  -s 2 -a 0.9  -b 0.9  -T 32 -i fp8  -o fp32 -w 2 -M 2 -N 16 -Q 4 -p 2 -t sgemm_tcu_op -l run_q_7.log"
  # "-m 64  -n 64  -k 512  -s 2 -a 0.9  -b 0.9  -T 32 -i fp8  -o fp32 -w 2 -M 2 -N 16 -Q 8 -p 2 -t sgemm_tcu_op -l run_q_8.log"

  # "-m 64  -n 64  -k 512  -s 2 -a 0.5  -b 0.5  -T 32 -i fp8  -o fp32 -w 2 -M 4 -N 8 -Q 1 -p 2 -t sgemm_tcu_op -l run_q_9.log"
  # "-m 64  -n 64  -k 512  -s 2 -a 0.5  -b 0.5  -T 32 -i fp8  -o fp32 -w 2 -M 4 -N 8 -Q 2 -p 2 -t sgemm_tcu_op -l run_q_10.log"
  # "-m 64  -n 64  -k 512  -s 2 -a 0.5  -b 0.5  -T 32 -i fp8  -o fp32 -w 2 -M 4 -N 8 -Q 4 -p 2 -t sgemm_tcu_op -l run_q_11.log"
  # "-m 64  -n 64  -k 512  -s 2 -a 0.5  -b 0.5  -T 32 -i fp8  -o fp32 -w 2 -M 4 -N 8 -Q 8 -p 2 -t sgemm_tcu_op -l run_q_12.log"

  # "-m 64  -n 64  -k 512  -s 2 -a 0.9  -b 0.9  -T 32 -i fp8  -o fp32 -w 2 -M 4 -N 8 -Q 1 -p 2 -t sgemm_tcu_op -l run_q_13.log"
  # "-m 64  -n 64  -k 512  -s 2 -a 0.9  -b 0.9  -T 32 -i fp8  -o fp32 -w 2 -M 4 -N 8 -Q 2 -p 2 -t sgemm_tcu_op -l run_q_14.log"
  # "-m 64  -n 64  -k 512  -s 2 -a 0.9  -b 0.9  -T 32 -i fp8  -o fp32 -w 2 -M 4 -N 8 -Q 4 -p 2 -t sgemm_tcu_op -l run_q_15.log"
  # "-m 64  -n 64  -k 512  -s 2 -a 0.9  -b 0.9  -T 32 -i fp8  -o fp32 -w 2 -M 4 -N 8 -Q 8 -p 2 -t sgemm_tcu_op -l run_q_16.log"

  #  COMPARISON WITH BASELINES
  # "-m 32  -n 32  -k 256  -s 0 -a 0.0  -b 0.0  -T 32 -i fp8  -o fp32 -w 2 -M 2 -N 16 -Q 4 -p 2 -t sgemm_tcu_op -l run1.log"
  # "-m 32  -n 32  -k 256  -s 0 -a 0.0  -b 0.0  -T 32 -i fp16 -o fp32 -w 2 -M 2 -N 16 -Q 4 -p 2 -t sgemm_tcu_op -l run2.log"
  # "-m 32  -n 32  -k 256  -s 0 -a 0.0  -b 0.0  -T 32 -i fp8  -o fp32 -w 2 -M 2 -N 16 -Q 4 -p 2 -t sgemm_tcu,sgemm_tcu_sp -l run1.log"
  # "-m 32  -n 32  -k 256  -s 0 -a 0.0  -b 0.0  -T 32 -i fp16 -o fp32 -w 2 -M 2 -N 16 -Q 4 -p 2 -t sgemm_tcu,sgemm_tcu_sp -l run2.log"
  
  #  SMALL SPARSITIES SENSITIVITY DIAGRAM
  # "-m 64  -n 64  -k 512  -s 0 -a 0.0  -b 0.0  -T 32 -i fp8  -o fp32 -w 2 -M 2 -N 16 -Q 4 -p 2 -t sgemm_tcu_op -l run3.log"
  # "-m 64  -n 64  -k 512  -s 2 -a 0.2  -b 0.2  -T 32 -i fp8  -o fp32 -w 2 -M 2 -N 16 -Q 4 -p 2 -t sgemm_tcu_op -l run4.log"
  # "-m 64  -n 64  -k 512  -s 2 -a 0.2  -b 0.6  -T 32 -i fp8  -o fp32 -w 2 -M 2 -N 16 -Q 4 -p 2 -t sgemm_tcu_op -l run5.log"
  # "-m 64  -n 64  -k 512  -s 2 -a 0.2  -b 0.9  -T 32 -i fp8  -o fp32 -w 2 -M 2 -N 16 -Q 4 -p 2 -t sgemm_tcu_op -l run6.log"
  # "-m 64  -n 64  -k 512  -s 2 -a 0.6  -b 0.2  -T 32 -i fp8  -o fp32 -w 2 -M 2 -N 16 -Q 4 -p 2 -t sgemm_tcu_op -l run7.log"
  # "-m 64  -n 64  -k 512  -s 2 -a 0.6  -b 0.6  -T 32 -i fp8  -o fp32 -w 2 -M 2 -N 16 -Q 4 -p 2 -t sgemm_tcu_op -l run8.log"
  # "-m 64  -n 64  -k 512  -s 2 -a 0.6  -b 0.9  -T 32 -i fp8  -o fp32 -w 2 -M 2 -N 16 -Q 4 -p 2 -t sgemm_tcu_op -l run9.log"
  # "-m 64  -n 64  -k 512  -s 2 -a 0.9  -b 0.2  -T 32 -i fp8  -o fp32 -w 2 -M 2 -N 16 -Q 4 -p 2 -t sgemm_tcu_op -l run10.log"
  # "-m 64  -n 64  -k 512  -s 2 -a 0.9  -b 0.6  -T 32 -i fp8  -o fp32 -w 2 -M 2 -N 16 -Q 4 -p 2 -t sgemm_tcu_op -l run11.log"
  # "-m 64  -n 64  -k 512  -s 2 -a 0.9  -b 0.9  -T 32 -i fp8  -o fp32 -w 2 -M 2 -N 16 -Q 4 -p 2 -t sgemm_tcu_op -l run12.log"

  #  SPARSITIES SENSITIVITY DIAGRAM
  # "-m 512  -n 512  -k 512  -s 0 -a 0.0  -b 0.0  -T 32 -i fp8  -o fp32 -w 2 -M 2 -N 16 -Q 4 -p 2 -t sgemm_tcu_op -l run13.log -d 0"

  # "-m 512  -n 512  -k 512  -s 2 -a 0.2  -b 0.2  -T 32 -i fp8  -o fp32 -w 2 -M 2 -N 16 -Q 4 -p 2 -t sgemm_tcu_op -l run14.log -d 0"
  # "-m 512  -n 512  -k 512  -s 2 -a 0.2  -b 0.4  -T 32 -i fp8  -o fp32 -w 2 -M 2 -N 16 -Q 4 -p 2 -t sgemm_tcu_op -l run15.log -d 0"
  # "-m 512  -n 512  -k 512  -s 2 -a 0.2  -b 0.6  -T 32 -i fp8  -o fp32 -w 2 -M 2 -N 16 -Q 4 -p 2 -t sgemm_tcu_op -l run16.log -d 0"
  # "-m 512  -n 512  -k 512  -s 2 -a 0.2  -b 0.8  -T 32 -i fp8  -o fp32 -w 2 -M 2 -N 16 -Q 4 -p 2 -t sgemm_tcu_op -l run17.log -d 0"
  # "-m 512  -n 512  -k 512  -s 2 -a 0.2  -b 0.9  -T 32 -i fp8  -o fp32 -w 2 -M 2 -N 16 -Q 4 -p 2 -t sgemm_tcu_op -l run18.log -d 0"
  # "-m 512  -n 512  -k 512  -s 2 -a 0.2  -b 0.99 -T 32 -i fp8  -o fp32 -w 2 -M 2 -N 16 -Q 4 -p 2 -t sgemm_tcu_op -l run19.log -d 0"

  # "-m 512  -n 512  -k 512  -s 2 -a 0.3  -b 0.2  -T 32 -i fp8  -o fp32 -w 2 -M 2 -N 16 -Q 4 -p 2 -t sgemm_tcu_op -l run20.log -d 0"
  # "-m 512  -n 512  -k 512  -s 2 -a 0.3  -b 0.4  -T 32 -i fp8  -o fp32 -w 2 -M 2 -N 16 -Q 4 -p 2 -t sgemm_tcu_op -l run21.log -d 0"
  # "-m 512  -n 512  -k 512  -s 2 -a 0.3  -b 0.6  -T 32 -i fp8  -o fp32 -w 2 -M 2 -N 16 -Q 4 -p 2 -t sgemm_tcu_op -l run22.log -d 0"
  # "-m 512  -n 512  -k 512  -s 2 -a 0.3  -b 0.8  -T 32 -i fp8  -o fp32 -w 2 -M 2 -N 16 -Q 4 -p 2 -t sgemm_tcu_op -l run23.log -d 0"
  # "-m 512  -n 512  -k 512  -s 2 -a 0.3  -b 0.9  -T 32 -i fp8  -o fp32 -w 2 -M 2 -N 16 -Q 4 -p 2 -t sgemm_tcu_op -l run24.log -d 0"
  # "-m 512  -n 512  -k 512  -s 2 -a 0.3  -b 0.99 -T 32 -i fp8  -o fp32 -w 2 -M 2 -N 16 -Q 4 -p 2 -t sgemm_tcu_op -l run25.log -d 0"

  # "-m 512  -n 512  -k 512  -s 2 -a 0.4  -b 0.2  -T 32 -i fp8  -o fp32 -w 2 -M 2 -N 16 -Q 4 -p 2 -t sgemm_tcu_op -l run26.log -d 0"
  # "-m 512  -n 512  -k 512  -s 2 -a 0.4  -b 0.4  -T 32 -i fp8  -o fp32 -w 2 -M 2 -N 16 -Q 4 -p 2 -t sgemm_tcu_op -l run27.log -d 0"
  # "-m 512  -n 512  -k 512  -s 2 -a 0.4  -b 0.6  -T 32 -i fp8  -o fp32 -w 2 -M 2 -N 16 -Q 4 -p 2 -t sgemm_tcu_op -l run28.log -d 0"
  # "-m 512  -n 512  -k 512  -s 2 -a 0.4  -b 0.8  -T 32 -i fp8  -o fp32 -w 2 -M 2 -N 16 -Q 4 -p 2 -t sgemm_tcu_op -l run29.log -d 0"
  # "-m 512  -n 512  -k 512  -s 2 -a 0.4  -b 0.9  -T 32 -i fp8  -o fp32 -w 2 -M 2 -N 16 -Q 4 -p 2 -t sgemm_tcu_op -l run30.log -d 0"
  # "-m 512  -n 512  -k 512  -s 2 -a 0.4  -b 0.99 -T 32 -i fp8  -o fp32 -w 2 -M 2 -N 16 -Q 4 -p 2 -t sgemm_tcu_op -l run31.log -d 0"

  # "-m 512  -n 512  -k 512  -s 2 -a 0.5  -b 0.2  -T 32 -i fp8  -o fp32 -w 2 -M 2 -N 16 -Q 4 -p 2 -t sgemm_tcu_op -l run32.log -d 0"
  # "-m 512  -n 512  -k 512  -s 2 -a 0.5  -b 0.4  -T 32 -i fp8  -o fp32 -w 2 -M 2 -N 16 -Q 4 -p 2 -t sgemm_tcu_op -l run33.log -d 0"
  # "-m 512  -n 512  -k 512  -s 2 -a 0.5  -b 0.6  -T 32 -i fp8  -o fp32 -w 2 -M 2 -N 16 -Q 4 -p 2 -t sgemm_tcu_op -l run34.log -d 0"
  # "-m 512  -n 512  -k 512  -s 2 -a 0.5  -b 0.8  -T 32 -i fp8  -o fp32 -w 2 -M 2 -N 16 -Q 4 -p 2 -t sgemm_tcu_op -l run35.log -d 0"
  # "-m 512  -n 512  -k 512  -s 2 -a 0.5  -b 0.9  -T 32 -i fp8  -o fp32 -w 2 -M 2 -N 16 -Q 4 -p 2 -t sgemm_tcu_op -l run36.log -d 0"
  # "-m 512  -n 512  -k 512  -s 2 -a 0.5  -b 0.99 -T 32 -i fp8  -o fp32 -w 2 -M 2 -N 16 -Q 4 -p 2 -t sgemm_tcu_op -l run37.log -d 0"

  # "-m 512  -n 512  -k 512  -s 2 -a 0.6  -b 0.2  -T 32 -i fp8  -o fp32 -w 2 -M 2 -N 16 -Q 4 -p 2 -t sgemm_tcu_op -l run38.log -d 0"
  # "-m 512  -n 512  -k 512  -s 2 -a 0.6  -b 0.4  -T 32 -i fp8  -o fp32 -w 2 -M 2 -N 16 -Q 4 -p 2 -t sgemm_tcu_op -l run39.log -d 0"
  # "-m 512  -n 512  -k 512  -s 2 -a 0.6  -b 0.6  -T 32 -i fp8  -o fp32 -w 2 -M 2 -N 16 -Q 4 -p 2 -t sgemm_tcu_op -l run40.log -d 0"
  # "-m 512  -n 512  -k 512  -s 2 -a 0.6  -b 0.8  -T 32 -i fp8  -o fp32 -w 2 -M 2 -N 16 -Q 4 -p 2 -t sgemm_tcu_op -l run41.log -d 0"
  # "-m 512  -n 512  -k 512  -s 2 -a 0.6  -b 0.9  -T 32 -i fp8  -o fp32 -w 2 -M 2 -N 16 -Q 4 -p 2 -t sgemm_tcu_op -l run42.log -d 0"
  # "-m 512  -n 512  -k 512  -s 2 -a 0.6  -b 0.99 -T 32 -i fp8  -o fp32 -w 2 -M 2 -N 16 -Q 4 -p 2 -t sgemm_tcu_op -l run43.log -d 0"

  # "-m 512  -n 512  -k 512  -s 2 -a 0.7  -b 0.2  -T 32 -i fp8  -o fp32 -w 2 -M 2 -N 16 -Q 4 -p 2 -t sgemm_tcu_op -l run44.log -d 0"
  # "-m 512  -n 512  -k 512  -s 2 -a 0.7  -b 0.4  -T 32 -i fp8  -o fp32 -w 2 -M 2 -N 16 -Q 4 -p 2 -t sgemm_tcu_op -l run45.log -d 0"
  # "-m 512  -n 512  -k 512  -s 2 -a 0.7  -b 0.6  -T 32 -i fp8  -o fp32 -w 2 -M 2 -N 16 -Q 4 -p 2 -t sgemm_tcu_op -l run46.log -d 0"
  # "-m 512  -n 512  -k 512  -s 2 -a 0.7  -b 0.8  -T 32 -i fp8  -o fp32 -w 2 -M 2 -N 16 -Q 4 -p 2 -t sgemm_tcu_op -l run47.log -d 0"
  # "-m 512  -n 512  -k 512  -s 2 -a 0.7  -b 0.9  -T 32 -i fp8  -o fp32 -w 2 -M 2 -N 16 -Q 4 -p 2 -t sgemm_tcu_op -l run48.log -d 0"
  # "-m 512  -n 512  -k 512  -s 2 -a 0.7  -b 0.99 -T 32 -i fp8  -o fp32 -w 2 -M 2 -N 16 -Q 4 -p 2 -t sgemm_tcu_op -l run49.log -d 0"

  # "-m 512  -n 512  -k 512  -s 2 -a 0.8  -b 0.2  -T 32 -i fp8  -o fp32 -w 2 -M 2 -N 16 -Q 4 -p 2 -t sgemm_tcu_op -l run50.log -d 0"
  # "-m 512  -n 512  -k 512  -s 2 -a 0.8  -b 0.4  -T 32 -i fp8  -o fp32 -w 2 -M 2 -N 16 -Q 4 -p 2 -t sgemm_tcu_op -l run51.log -d 0"
  # "-m 512  -n 512  -k 512  -s 2 -a 0.8  -b 0.6  -T 32 -i fp8  -o fp32 -w 2 -M 2 -N 16 -Q 4 -p 2 -t sgemm_tcu_op -l run52.log -d 0"
  # "-m 512  -n 512  -k 512  -s 2 -a 0.8  -b 0.8  -T 32 -i fp8  -o fp32 -w 2 -M 2 -N 16 -Q 4 -p 2 -t sgemm_tcu_op -l run53.log -d 0"
  # "-m 512  -n 512  -k 512  -s 2 -a 0.8  -b 0.9  -T 32 -i fp8  -o fp32 -w 2 -M 2 -N 16 -Q 4 -p 2 -t sgemm_tcu_op -l run54.log -d 0"
  # "-m 512  -n 512  -k 512  -s 2 -a 0.8  -b 0.99 -T 32 -i fp8  -o fp32 -w 2 -M 2 -N 16 -Q 4 -p 2 -t sgemm_tcu_op -l run55.log -d 0"

  # "-m 512  -n 512  -k 512  -s 2 -a 0.9  -b 0.2  -T 32 -i fp8  -o fp32 -w 2 -M 2 -N 16 -Q 4 -p 2 -t sgemm_tcu_op -l run56.log -d 0"
  # "-m 512  -n 512  -k 512  -s 2 -a 0.9  -b 0.4  -T 32 -i fp8  -o fp32 -w 2 -M 2 -N 16 -Q 4 -p 2 -t sgemm_tcu_op -l run57.log -d 0"
  # "-m 512  -n 512  -k 512  -s 2 -a 0.9  -b 0.6  -T 32 -i fp8  -o fp32 -w 2 -M 2 -N 16 -Q 4 -p 2 -t sgemm_tcu_op -l run58.log -d 0"
  # "-m 512  -n 512  -k 512  -s 2 -a 0.9  -b 0.8  -T 32 -i fp8  -o fp32 -w 2 -M 2 -N 16 -Q 4 -p 2 -t sgemm_tcu_op -l run59.log -d 0"
  # "-m 512  -n 512  -k 512  -s 2 -a 0.9  -b 0.9  -T 32 -i fp8  -o fp32 -w 2 -M 2 -N 16 -Q 4 -p 2 -t sgemm_tcu_op -l run60.log -d 0"
  # "-m 512  -n 512  -k 512  -s 2 -a 0.9  -b 0.99 -T 32 -i fp8  -o fp32 -w 2 -M 2 -N 16 -Q 4 -p 2 -t sgemm_tcu_op -l run61.log -d 0"

  # "-m 512  -n 512  -k 512  -s 2 -a 0.99 -b 0.2  -T 32 -i fp8  -o fp32 -w 2 -M 2 -N 16 -Q 4 -p 2 -t sgemm_tcu_op -l run62.log -d 0"
  # "-m 512  -n 512  -k 512  -s 2 -a 0.99 -b 0.4  -T 32 -i fp8  -o fp32 -w 2 -M 2 -N 16 -Q 4 -p 2 -t sgemm_tcu_op -l run63.log -d 0"
  # "-m 512  -n 512  -k 512  -s 2 -a 0.99 -b 0.6  -T 32 -i fp8  -o fp32 -w 2 -M 2 -N 16 -Q 4 -p 2 -t sgemm_tcu_op -l run64.log -d 0"
  # "-m 512  -n 512  -k 512  -s 2 -a 0.99 -b 0.8  -T 32 -i fp8  -o fp32 -w 2 -M 2 -N 16 -Q 4 -p 2 -t sgemm_tcu_op -l run65.log -d 0"
  # "-m 512  -n 512  -k 512  -s 2 -a 0.99 -b 0.9  -T 32 -i fp8  -o fp32 -w 2 -M 2 -N 16 -Q 4 -p 2 -t sgemm_tcu_op -l run66.log -d 0"
  # "-m 512  -n 512  -k 512  -s 2 -a 0.99 -b 0.99 -T 32 -i fp8  -o fp32 -w 2 -M 2 -N 16 -Q 4 -p 2 -t sgemm_tcu_op -l run67.log -d 0"
)

usage() {
  cat <<'EOF'
Usage: ./custom/run_all.sh

Runs custom/run.sh over a predefined configuration sweep.

Environment overrides:
  TESTS          Default test list passed to run.sh -t when a config omits -t (default: all)
  OUTPUT_DIR     Results directory (default: custom/run_all)
  CSV_FILE       Manifest CSV path (default: custom/run_all/run_all.csv)
  LOG_DIR        Default log directory relative to build/ when a config omits -l (default: .)
  DO_CLEAN_FLAG  Set to -C to skip make clean in run.sh
  JOBS           Number of configurations to run in parallel (default: 1)
  RUN_GROUP      Log group used when a config omits -l (default: all)
  WORK_BUILD_ROOT Per-worker build-copy root for JOBS > 1 (default: build/run_all_workers)
  MAKE_LOCK_FILE Shared lock file used to serialize build make steps (default: build/run_all.make.lock)

Config options:
  -d <value>     blackbox --debug level passed through to run.sh (default: run.sh default)

Generated logs and stats:
  Logs should use run_<name>_<n>.log, for example run_q_1.log.
  Stats from run_<name>_<n>.log are written under OUTPUT_DIR/run_<name>/.
  Queue-depth stats and plots are written under custom/run_all/run_q/.
EOF
}

csv_escape() {
  local value="$1"
  value="${value//\"/\"\"}"
  printf '"%s"' "${value}"
}

initialize_csv() {
  printf '%s\n' "${CSV_HEADER}" > "${CSV_FILE}"
}

merge_csv_parts() {
  local temp_file="${CSV_FILE}.tmp.${BASHPID}"
  local csv_part

  {
    printf '%s\n' "${CSV_HEADER}"
    {
      for csv_part in "${OUTPUT_DIR}"/run_all.part*.csv; do
        if [[ -f "${csv_part}" ]]; then
          cat "${csv_part}"
        fi
      done
    } | sort -t, -k3,3n
  } > "${temp_file}" && mv "${temp_file}" "${CSV_FILE}"
}

make_log_path() {
  local run_index="$1"

  if [[ -n "${LOG_DIR}" && "${LOG_DIR}" != "." ]]; then
    printf '%s/run_%s_%s.log' "${LOG_DIR}" "${RUN_GROUP}" "${run_index}"
  else
    printf 'run_%s_%s.log' "${RUN_GROUP}" "${run_index}"
  fi
}

normalize_log_path() {
  local log_file="$1"

  case "${log_file}" in
    /*|*/*)
      printf '%s\n' "${log_file}"
      ;;
    *)
      if [[ -n "${LOG_DIR}" && "${LOG_DIR}" != "." ]]; then
        printf '%s/%s\n' "${LOG_DIR}" "${log_file}"
      else
        printf '%s\n' "${log_file}"
      fi
      ;;
  esac
}

set_config_log_file() {
  local replacement="$1"
  local index

  for index in "${!config_args[@]}"; do
    if [[ "${config_args[$index]}" == "-l" ]]; then
      config_args[$((index + 1))]="${replacement}"
      return
    fi
  done

  config_args+=("-l" "${replacement}")
}

resolve_log_path() {
  local log_file="$1"

  case "${log_file}" in
    /*)
      printf '%s\n' "${log_file}"
      ;;
    *)
      printf '%s/%s\n' "${BUILD_DIR}" "${log_file}"
      ;;
  esac
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

print_run_banner() {
  local label="$1"
  local log_file="$2"
  local phase="$3"

  printf '%s\n' '*****************************'
  printf '%s: %s log=%s\n' "${phase}" "${label}" "${log_file}"
  printf '%s\n' '*****************************'
}

prefix_run_output() {
  local label="$1"
  local log_file="$2"
  local line

  while IFS= read -r line; do
    printf '[%s log=%s] %s\n' "${label}" "${log_file}" "${line}"
  done
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
  local log_path stats_dir

  log_path="$(resolve_log_path "${log_file}")"
  stats_dir="$(stats_dir_for_log "${log_file}")"

  if [[ ! -f "${log_path}" ]]; then
    echo "Skipping stats, missing log: ${log_path}" >&2
    return 1
  fi

  echo "stats: ${log_file}"
  mkdir -p "${stats_dir}"
  STATS_DIR="${stats_dir}" PYTHONDONTWRITEBYTECODE=1 python3 "${PLOT_PY}" "${log_path}"
}

run_summary_plots() {
  if [[ ! -f "${QUEUE_PLOT_PY}" ]]; then
    return 0
  fi

  echo "queue-depth plot: ${QUEUE_PLOT_PY}"
  "${PLOT_PYTHON}" "${QUEUE_PLOT_PY}" "${OUTPUT_DIR}/run_q" || return $?
}

remove_log_outputs() {
  local log_file="$1"
  local log_path stat_path

  log_path="$(resolve_log_path "${log_file}")"
  stat_path="$(stat_path_for_log "${log_file}")"

  rm -f "${log_path}" "${stat_path}"
}

stats_dir_for_log() {
  local log_file="$1"
  local base stem suffix group

  base="$(basename "${log_file}")"
  stem="${base%.*}"
  suffix="${stem##*_}"

  if [[ "${stem}" == run_*_* && "${suffix}" =~ ^[0-9]+$ ]]; then
    group="${stem%_*}"
    printf '%s/%s\n' "${OUTPUT_DIR}" "${group}"
  else
    printf '%s\n' "${OUTPUT_DIR}"
  fi
}

stat_path_for_log() {
  local log_file="$1"
  local base stats_dir

  base="$(basename "${log_file}")"
  stats_dir="$(stats_dir_for_log "${log_file}")"
  printf '%s/%s.stat\n' "${stats_dir}" "${base%.*}"
}

status_for_log() {
  local log_file="$1"
  local fallback_status="$2"
  local run_started_at="$3"
  local stat_path

  stat_path="$(stat_path_for_log "${log_file}")"
  if [[ -f "${stat_path}" ]]; then
    if [[ "$(stat -c '%Y' "${stat_path}")" -ge "${run_started_at}" ]] && rg -q '^PASS$' "${stat_path}"; then
      printf '0\n'
      return
    fi
    if [[ "$(stat -c '%Y' "${stat_path}")" -ge "${run_started_at}" ]] && rg -q '^FAIL$' "${stat_path}"; then
      printf '1\n'
      return
    fi
  fi

  printf '%s\n' "${fallback_status}"
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
      -d) config_args+=("$1" "$2"); shift 2 ;;
      -t) config_tests="$2"; config_args+=("$1" "$2"); shift 2 ;;
      -l) log_file="$(normalize_log_path "$2")"; config_args+=("$1" "${log_file}"); shift 2 ;;
      *)
        echo "Unsupported config option: $1" >&2
        exit 1
        ;;
    esac
  done

  validate_tests "${config_tests}"
}

prepare_worker_build() {
  local worker_build_dir="$1"

  if command -v rsync >/dev/null 2>&1; then
    mkdir -p "${worker_build_dir}"
    rsync -a --delete \
      --exclude '/run_all_workers/' \
      --exclude '*.log' \
      --exclude '*.vcd' \
      "${BUILD_DIR}/" "${worker_build_dir}/"
    return
  fi

  rm -rf "${worker_build_dir}"
  mkdir -p "${worker_build_dir}"
  (
    cd "${BUILD_DIR}" &&
    tar \
      --exclude='./run_all_workers' \
      --exclude='*.log' \
      --exclude='*.vcd' \
      -cf - .
  ) | (
    cd "${worker_build_dir}" &&
    tar -xf -
  )
}

run_config() {
  local config="$1"
  local logical_run_index="$2"
  local worker_build_dir="$3"
  local csv_part="$4"
  local status run_status test_log_file run_log_file run_label run_started_at

  read -r -a raw_config_args <<< "${config}"
  load_config "${raw_config_args[@]}"
  if [[ -z "${log_file}" ]]; then
    log_file="$(make_log_path "${logical_run_index}")"
  fi

  run_log_file="${log_file}"
  if [[ "${worker_build_dir}" != "${BUILD_DIR}" ]]; then
    run_log_file="$(resolve_log_path "${log_file}")"
  fi
  set_config_log_file "${run_log_file}"

  if [[ "$(dirname "${log_file}")" != "." ]]; then
    mkdir -p "$(dirname "$(resolve_log_path "${log_file}")")"
  fi

  run_label="run${logical_run_index}"
  print_run_banner "${run_label}" "${log_file}" "START"
  echo "${run_label}: tests=${config_tests} writing=${log_file} m=${m} n=${n} k=${k} sparsity=${sparsity} a=${a_sparsity} b=${b_sparsity} threads=${num_threads} queue=${queue_depth}"

  if test_enabled "${config_tests}" "sgemm_tcu_op"; then
    remove_log_outputs "${log_file}"
  fi
  if test_enabled "${config_tests}" "sgemm_tcu"; then
    remove_log_outputs "$(make_suffixed_log "${log_file}" "_ip")"
  fi
  if test_enabled "${config_tests}" "sgemm_tcu_sp"; then
    remove_log_outputs "$(make_suffixed_log "${log_file}" "_ip_sp")"
  fi

  run_started_at="$(date +%s)"
  MAKE_LOCK_FILE="${MAKE_LOCK_FILE}" BUILD_DIR="${worker_build_dir}" "${RUN_SH}" \
    -t "${config_tests}" \
    "${config_args[@]}" \
    ${DO_CLEAN_FLAG} \
    > >(prefix_run_output "${run_label}" "${log_file}") \
    2> >(prefix_run_output "${run_label}" "${log_file}" >&2)
  run_status=$?
  status="${run_status}"
  print_run_banner "${run_label}" "${log_file}" "END status=${run_status}"

  local CSV_FILE="${csv_part}"
  if test_enabled "${config_tests}" "sgemm_tcu_op"; then
    run_plot_for_log "${log_file}" || status=$?
    append_csv_row "${log_file}" "sgemm_tcu_op" "$(status_for_log "${log_file}" "${run_status}" "${run_started_at}")" "${logical_run_index}" "${m}" "${n}" "${k}" "${sparsity}" "${a_sparsity}" "${b_sparsity}" "${num_threads}" "${itype}" "${otype}" "${warps}" "${block_m}" "${block_n}" "${queue_depth}"
  fi
  if test_enabled "${config_tests}" "sgemm_tcu"; then
    test_log_file="$(make_suffixed_log "${log_file}" "_ip")"
    run_plot_for_log "${test_log_file}" || status=$?
    append_csv_row "${test_log_file}" "sgemm_tcu" "$(status_for_log "${test_log_file}" "${run_status}" "${run_started_at}")" "${logical_run_index}" "${m}" "${n}" "${k}" "${sparsity}" "${a_sparsity}" "${b_sparsity}" "${num_threads}" "${itype}" "${otype}" "${warps}" "${block_m}" "${block_n}" "${queue_depth}"
  fi
  if test_enabled "${config_tests}" "sgemm_tcu_sp"; then
    test_log_file="$(make_suffixed_log "${log_file}" "_ip_sp")"
    run_plot_for_log "${test_log_file}" || status=$?
    append_csv_row "${test_log_file}" "sgemm_tcu_sp" "$(status_for_log "${test_log_file}" "${run_status}" "${run_started_at}")" "${logical_run_index}" "${m}" "${n}" "${k}" "${sparsity}" "${a_sparsity}" "${b_sparsity}" "${num_threads}" "${itype}" "${otype}" "${warps}" "${block_m}" "${block_n}" "${queue_depth}"
  fi

  return "${status}"
}

run_worker() {
  local worker_index="$1"
  local worker_count="$2"
  local worker_build_dir="$3"
  local csv_part="$4"
  local config_index status

  : > "${csv_part}"
  status=0

  config_index=$((worker_index - 1))
  while [[ "${config_index}" -lt "${#CONFIGS[@]}" ]]; do
    run_config "${CONFIGS[$config_index]}" "$((config_index + 1))" "${worker_build_dir}" "${csv_part}" || status=$?
    config_index=$((config_index + worker_count))
  done

  return "${status}"
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

case "${JOBS}" in
  ''|*[!0-9]*)
    echo "JOBS must be a positive integer." >&2
    exit 1
    ;;
esac
if [[ "${JOBS}" -lt 1 ]]; then
  echo "JOBS must be a positive integer." >&2
  exit 1
fi
if [[ "${JOBS}" -gt "${#CONFIGS[@]}" ]]; then
  JOBS="${#CONFIGS[@]}"
fi

mkdir -p "${BUILD_DIR}"
mkdir -p "${OUTPUT_DIR}"
mkdir -p "$(dirname "${CSV_FILE}")"
if [[ -n "${LOG_DIR}" && "${LOG_DIR}" != "." ]]; then
  mkdir -p "$(dirname "$(resolve_log_path "${LOG_DIR}/.keep")")"
fi

initialize_csv

overall_status=0
run_all_parent_pid="${BASHPID}"
trap 'if [[ "${BASHPID}" -eq "${run_all_parent_pid}" ]]; then merge_csv_parts; fi' EXIT

if [[ "${JOBS}" -eq 1 ]]; then
  run_worker 1 1 "${BUILD_DIR}" "${OUTPUT_DIR}/run_all.part1.csv" || overall_status=$?
  merge_csv_parts
else
  mkdir -p "${WORK_BUILD_ROOT}"
  pids=()
  for worker_index in $(seq 1 "${JOBS}"); do
    worker_build_dir="${WORK_BUILD_ROOT}/job${worker_index}"
    csv_part="${OUTPUT_DIR}/run_all.part${worker_index}.csv"
    echo "Preparing worker ${worker_index}/${JOBS}: ${worker_build_dir}"
    prepare_worker_build "${worker_build_dir}" || exit $?
    run_worker "${worker_index}" "${JOBS}" "${worker_build_dir}" "${csv_part}" &
    pids+=("$!")
  done

  for pid in "${pids[@]}"; do
    wait "${pid}" || overall_status=$?
    merge_csv_parts
  done
fi

merge_csv_parts

run_summary_plots || overall_status=$?

echo "Wrote ${CSV_FILE}"
exit "${overall_status}"
