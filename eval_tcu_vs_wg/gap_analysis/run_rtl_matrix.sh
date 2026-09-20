#!/bin/bash
set -u
BUILD=~/dev/vortex_spg/build
OUT=~/dev/vortex_spg/eval_tcu_vs_wg/gap_analysis
STATUS=$OUT/status_rtl_matrix.txt
cd "$BUILD"
: > "$STATUS"
run(){
  local APP=$1 IW=$2 PC=$3
  local CFG="-DVX_CFG_NUM_THREADS=16 -DVX_CFG_NUM_WARPS=16 -DVX_CFG_ISSUE_WIDTH=$IW -DVX_CFG_EXT_TCU_ENABLE -DVX_CFG_TCU_WGMMA_ENABLE -DVX_CFG_EXT_DXA_ENABLE -DITYPE=fp16 -DOTYPE=fp32 -DWGMMA_NRC=16"
  local TAG="rtl_${APP}_iw${IW}_perf${PC}"
  echo "[$(date +%H:%M:%S)] START $TAG" >> "$STATUS"
  make -C tests/regression/$APP clean >/dev/null 2>&1
  CONFIGS="$CFG" make -C tests/regression/$APP > $OUT/${TAG}.log 2>&1 || { echo "BUILD-FAIL $TAG" >> "$STATUS"; return; }
  CONFIGS="$CFG" ./ci/blackbox.sh --driver=rtlsim --app=$APP --args="-m 128 -n 128 -k 128" --perf=$PC >> $OUT/${TAG}.log 2>&1
  echo "[$(date +%H:%M:%S)] DONE $TAG rc=$?" >> "$STATUS"
}
for PC in 1 7 11 16; do run sgemm_tcu_wg_dxa 4 $PC; done
for PC in 1 16;      do run sgemm_tcu_wg_dxa 1 $PC; done
echo "RTL-MATRIX-COMPLETE" >> "$STATUS"
