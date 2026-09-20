#!/bin/bash
# User-requested 3 simx runs: LKG optimizations + L1 sectoring, 128^3, NW=16, NT=32.
set -u
BUILD=~/dev/vortex_spg/build
OUT=~/dev/vortex_spg/eval_tcu_vs_wg/fedp2k
STATUS=$OUT/status_user3.txt
cd "$BUILD"
: > "$STATUS"
CFGBASE="-DVX_CFG_NUM_THREADS=32 -DVX_CFG_NUM_WARPS=16 -DVX_CFG_ISSUE_WIDTH=4 -DVX_CFG_EXT_TCU_ENABLE -DVX_CFG_TCU_WGMMA_ENABLE -DITYPE=fp16 -DOTYPE=fp32 -DWGMMA_NRC=16"
DSEC="-DVX_CFG_DCACHE_LINE_SIZE=128 -DVX_CFG_DCACHE_SECTOR_SIZE=64"
BUNDLE="-DVX_CFG_EXT_DXA_ENABLE -DCSTORE_LMEM -DDXA_DOUBLE_BUFFER -DVX_CFG_LMEM_LOG_SIZE=15 -DVX_CFG_DXA_MAX_INFLIGHT=32"
run(){
  local APP=$1 EXTRA=$2 TAG=$3
  local CFG="$CFGBASE $EXTRA"
  echo "[$(date +%H:%M:%S)] START $TAG" >> "$STATUS"
  make -C tests/regression/$APP clean >/dev/null 2>&1
  if ! CONFIGS="$CFG" make -C tests/regression/$APP > $OUT/${TAG}.log 2>&1; then
    echo "[$(date +%H:%M:%S)] BUILD-FAIL $TAG" >> "$STATUS"; return
  fi
  CONFIGS="$CFG" ./ci/blackbox.sh --driver=simx --app=$APP \
    --args="-m 128 -n 128 -k 128" --perf=1 >> $OUT/${TAG}.log 2>&1
  echo "[$(date +%H:%M:%S)] DONE $TAG rc=$?" >> "$STATUS"
}
run sgemm_tcu           "$DSEC"                                          u3_wmma_dsec
run sgemm_tcu_wg_dxa    "$BUNDLE $DSEC -DVX_CFG_TCU_FEDP2K"              u3_dxa2k_bundle_dsec
run sgemm_tcu_wg_dxa_mcast "-DDXA_MCAST_A $BUNDLE $DSEC -DVX_CFG_TCU_FEDP2K" u3_mcast2k_bundle_dsec
echo "USER3-COMPLETE" >> "$STATUS"
