#!/bin/bash
set -u
cd ~/dev/vortex_spg/build
OUT=~/dev/vortex_spg/eval_tcu_vs_wg/gap_analysis
STATUS=$OUT/status_wc_extra.txt
: > "$STATUS"
BASE="-DVX_CFG_NUM_THREADS=16 -DVX_CFG_NUM_WARPS=16 -DVX_CFG_ISSUE_WIDTH=4 -DVX_CFG_EXT_TCU_ENABLE -DVX_CFG_TCU_WGMMA_ENABLE -DVX_CFG_EXT_DXA_ENABLE -DITYPE=fp16 -DOTYPE=fp32 -DWGMMA_NRC=16"
run(){
  local TAG=$1 EXTRA=$2 PC=$3
  echo "[$(date +%H:%M:%S)] START $TAG" >> "$STATUS"
  make -C tests/regression/sgemm_tcu_wg_dxa clean >/dev/null 2>&1
  CONFIGS="$BASE $EXTRA" make -C tests/regression/sgemm_tcu_wg_dxa > $OUT/wcx_$TAG.log 2>&1 || { echo "BUILD-FAIL $TAG" >> "$STATUS"; return; }
  CONFIGS="$BASE $EXTRA" ./ci/blackbox.sh --driver=rtlsim --app=sgemm_tcu_wg_dxa --args="-m 128 -n 128 -k 128" --perf=$PC >> $OUT/wcx_$TAG.log 2>&1
  echo "[$(date +%H:%M:%S)] DONE $TAG rc=$?" >> "$STATUS"
}
run dbufv3   "-DDXA_DOUBLE_BUFFER"   1
run fedp2k   "-DVX_CFG_TCU_FEDP2K"   16
echo WCX-COMPLETE >> "$STATUS"
