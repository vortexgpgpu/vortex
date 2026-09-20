#!/bin/bash
# simx perf-counter matrix for the wg_dxa gap investigation
# NW=16, NT=16, IW in {1,4}, m=n=k=128
set -u
BUILD=~/dev/vortex_spg/build
OUT=~/dev/vortex_spg/eval_tcu_vs_wg/gap_analysis
STATUS=$OUT/status_perf_matrix.txt
cd "$BUILD"
: > "$STATUS"

run_one() {
  local APP=$1 IW=$2 PERF=$3
  local CFG="-DVX_CFG_NUM_THREADS=16 -DVX_CFG_NUM_WARPS=16 -DVX_CFG_ISSUE_WIDTH=$IW -DVX_CFG_EXT_TCU_ENABLE -DVX_CFG_TCU_WGMMA_ENABLE -DVX_CFG_EXT_DXA_ENABLE -DITYPE=fp16 -DOTYPE=fp32 -DWGMMA_NRC=16"
  local TAG="${APP}_iw${IW}_perf${PERF}"
  local LOG=$OUT/${TAG}.log
  echo "[$(date +%H:%M:%S)] START $TAG" >> "$STATUS"
  make -C tests/regression/$APP clean > /dev/null 2>&1
  if ! CONFIGS="$CFG" make -C tests/regression/$APP > "$LOG" 2>&1; then
    echo "[$(date +%H:%M:%S)] APP-BUILD-FAIL $TAG" >> "$STATUS"; return
  fi
  CONFIGS="$CFG" ./ci/blackbox.sh --driver=simx --app=$APP \
    --args="-m 128 -n 128 -k 128" --perf=$PERF >> "$LOG" 2>&1
  echo "[$(date +%H:%M:%S)] DONE $TAG rc=$? " >> "$STATUS"
}

for IW in 1 4; do
  for PERF in 1 4 7 11; do
    run_one sgemm_tcu        $IW $PERF
    run_one sgemm_tcu_wg     $IW $PERF
    run_one sgemm_tcu_wg_dxa $IW $PERF
  done
  run_one sgemm_tcu_wg_dxa $IW 16
done
echo "MATRIX-COMPLETE" >> "$STATUS"
