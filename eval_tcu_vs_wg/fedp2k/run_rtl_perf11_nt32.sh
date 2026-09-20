#!/bin/bash
# FEDP2K investigation: NT=32 IW=4 rtlsim counter runs (cached configs from sweep).
set -u
BUILD=~/dev/vortex_spg/build
OUT=~/dev/vortex_spg/eval_tcu_vs_wg/fedp2k
STATUS=$OUT/status_rtl_perf.txt
cd "$BUILD"
: > "$STATUS"
run(){
  local EXTRA=$1 PC=$2 TAG=$3
  local CFG="-DVX_CFG_NUM_THREADS=32 -DVX_CFG_NUM_WARPS=16 -DVX_CFG_ISSUE_WIDTH=4 -DVX_CFG_EXT_TCU_ENABLE -DVX_CFG_TCU_WGMMA_ENABLE -DVX_CFG_EXT_DXA_ENABLE $EXTRA -DITYPE=fp16 -DOTYPE=fp32 -DWGMMA_NRC=16"
  echo "[$(date +%H:%M:%S)] START $TAG" >> "$STATUS"
  make -C tests/regression/sgemm_tcu_wg_dxa clean >/dev/null 2>&1
  CONFIGS="$CFG" make -C tests/regression/sgemm_tcu_wg_dxa > $OUT/${TAG}.log 2>&1 || { echo "BUILD-FAIL $TAG" >> "$STATUS"; return; }
  CONFIGS="$CFG" ./ci/blackbox.sh --driver=rtlsim --app=sgemm_tcu_wg_dxa \
    --args="-m 128 -n 128 -k 128" --perf=$PC >> $OUT/${TAG}.log 2>&1
  echo "[$(date +%H:%M:%S)] DONE $TAG rc=$?" >> "$STATUS"
}
run ""                      11 base_iw4_perf11
run "-DVX_CFG_TCU_FEDP2K"   11 fedp2k_iw4_perf11
run ""                      7  base_iw4_perf7
run "-DVX_CFG_TCU_FEDP2K"   7  fedp2k_iw4_perf7
echo "PERF-RUNS-COMPLETE" >> "$STATUS"
