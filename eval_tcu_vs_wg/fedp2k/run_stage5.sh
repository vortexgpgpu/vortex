#!/bin/bash
# Stage 5: L1 (dcache) sectoring — 128B line / 64B sector — aimed at the
# C-store epilogue (per-row single-line coverage, halved MSHR/tag pressure,
# dirty-sector-only writebacks). No L2 (user's call: single-core, L1 only).
set -u
BUILD=~/dev/vortex_spg/build
OUT=~/dev/vortex_spg/eval_tcu_vs_wg/fedp2k
STATUS=$OUT/status_stage5.txt
cd "$BUILD"
: > "$STATUS"
while ! grep -q "STAGE4-COMPLETE" $OUT/status_stage4.txt 2>/dev/null; do sleep 60; done
echo "[$(date +%H:%M:%S)] stage4 complete, starting stage5" >> "$STATUS"

CFGBASE="-DVX_CFG_NUM_THREADS=32 -DVX_CFG_NUM_WARPS=16 -DVX_CFG_ISSUE_WIDTH=4 -DVX_CFG_EXT_TCU_ENABLE -DVX_CFG_TCU_WGMMA_ENABLE -DVX_CFG_EXT_DXA_ENABLE -DITYPE=fp16 -DOTYPE=fp32 -DWGMMA_NRC=16"
DSEC="-DVX_CFG_DCACHE_LINE_SIZE=128 -DVX_CFG_DCACHE_SECTOR_SIZE=64"
BUNDLE="-DCSTORE_LMEM -DDXA_DOUBLE_BUFFER -DVX_CFG_LMEM_LOG_SIZE=15 -DVX_CFG_DXA_MAX_INFLIGHT=32"

run(){
  local EXTRA=$1 TAG=$2 K=$3
  local CFG="$CFGBASE $EXTRA"
  echo "[$(date +%H:%M:%S)] START $TAG" >> "$STATUS"
  make -C tests/regression/sgemm_tcu_wg_dxa clean >/dev/null 2>&1
  if ! CONFIGS="$CFG" make -C tests/regression/sgemm_tcu_wg_dxa > $OUT/${TAG}.log 2>&1; then
    echo "[$(date +%H:%M:%S)] BUILD-FAIL $TAG" >> "$STATUS"; return
  fi
  t0=$(date +%s)
  CONFIGS="$CFG" ./ci/blackbox.sh --driver=rtlsim --app=sgemm_tcu_wg_dxa \
    --args="-m 128 -n 128 -k $K" --perf=1 >> $OUT/${TAG}.log 2>&1
  rc=$?; t1=$(date +%s)
  echo "[$(date +%H:%M:%S)] DONE $TAG rc=$rc wall=$((t1-t0))s" >> "$STATUS"
}

# L1 sectoring alone (plain store), with the coalesced store, and in the bundle
run "$DSEC"                                     s5_dsec_k128         128
run "$DSEC -DCSTORE_LMEM"                       s5_dsec_cstore_k128  128
run "$BUNDLE $DSEC"                             s5_dsec_bundle_k128  128
run "$BUNDLE $DSEC -DVX_CFG_TCU_FEDP2K"         s5_dsec_bundle2k_k128 128
run "$BUNDLE $DSEC"                             s5_dsec_bundle_k512  512
run "$BUNDLE $DSEC -DVX_CFG_TCU_FEDP2K"         s5_dsec_bundle2k_k512 512

echo "STAGE5-COMPLETE" >> "$STATUS"
