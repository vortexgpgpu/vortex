#!/bin/bash
set -u
cd ~/dev/vortex_spg/build
OUT=~/dev/vortex_spg/eval_tcu_vs_wg/gap_analysis
STATUS=$OUT/status_wc.txt
: > "$STATUS"
CFG="-DVX_CFG_NUM_THREADS=16 -DVX_CFG_NUM_WARPS=16 -DVX_CFG_ISSUE_WIDTH=4 -DVX_CFG_EXT_TCU_ENABLE -DVX_CFG_TCU_WGMMA_ENABLE -DVX_CFG_EXT_DXA_ENABLE -DITYPE=fp16 -DOTYPE=fp32 -DWGMMA_NRC=16"
run(){
  local APP=$1 ARGS=$2 PC=$3 TAG=$4
  echo "[$(date +%H:%M:%S)] START $TAG" >> "$STATUS"
  make -C tests/regression/$APP clean >/dev/null 2>&1
  CONFIGS="$CFG" make -C tests/regression/$APP > $OUT/wc_$TAG.log 2>&1 || { echo "BUILD-FAIL $TAG" >> "$STATUS"; return; }
  CONFIGS="$CFG" ./ci/blackbox.sh --driver=rtlsim --app=$APP --args="$ARGS" --perf=$PC >> $OUT/wc_$TAG.log 2>&1
  echo "[$(date +%H:%M:%S)] DONE $TAG rc=$?" >> "$STATUS"
}
run sgemm_tcu_wg_dxa "-m 32 -n 32 -k 32"    16 tiny32
run sgemm_tcu_wg_dxa "-m 128 -n 128 -k 128" 16 full128_dxa
run sgemm_tcu_wg_dxa "-m 128 -n 128 -k 128" 1  full128_core
run sgemm_tcu_wg_dxa "-m 128 -n 128 -k 128" 11 full128_tcu
run sgemm_tcu_wg_dxa_mcast "-m 128 -n 128 -k 128" 16 mcast128
echo WC-COMPLETE >> "$STATUS"
