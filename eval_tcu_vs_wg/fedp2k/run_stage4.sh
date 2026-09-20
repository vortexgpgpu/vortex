#!/bin/bash
# FEDP2K investigation stage 4: final rtlsim validation matrix (NT=32, IW=4).
#  - CSTORE_LMEM epilogue, alone and combined with dbuf+inflight32
#  - K=512 regime where staging/compute margins dominate the fixed tail
#  - mcast-A variant at K=512 (its wave-boundary cost amortizes there)
set -u
BUILD=~/dev/vortex_spg/build
OUT=~/dev/vortex_spg/eval_tcu_vs_wg/fedp2k
STATUS=$OUT/status_stage4.txt
cd "$BUILD"
: > "$STATUS"

CFGBASE="-DVX_CFG_NUM_THREADS=32 -DVX_CFG_NUM_WARPS=16 -DVX_CFG_ISSUE_WIDTH=4 -DVX_CFG_EXT_TCU_ENABLE -DVX_CFG_TCU_WGMMA_ENABLE -DVX_CFG_EXT_DXA_ENABLE -DITYPE=fp16 -DOTYPE=fp32 -DWGMMA_NRC=16"

run(){
  local APP=$1 EXTRA=$2 TAG=$3 K=$4
  local CFG="$CFGBASE $EXTRA"
  echo "[$(date +%H:%M:%S)] START $TAG" >> "$STATUS"
  make -C tests/regression/$APP clean >/dev/null 2>&1
  if ! CONFIGS="$CFG" make -C tests/regression/$APP > $OUT/${TAG}.log 2>&1; then
    echo "[$(date +%H:%M:%S)] BUILD-FAIL $TAG" >> "$STATUS"; return
  fi
  t0=$(date +%s)
  CONFIGS="$CFG" ./ci/blackbox.sh --driver=rtlsim --app=$APP \
    --args="-m 128 -n 128 -k $K" --perf=1 >> $OUT/${TAG}.log 2>&1
  rc=$?; t1=$(date +%s)
  echo "[$(date +%H:%M:%S)] DONE $TAG rc=$rc wall=$((t1-t0))s" >> "$STATUS"
}

APP=sgemm_tcu_wg_dxa
MC=sgemm_tcu_wg_dxa_mcast
BUNDLE="-DCSTORE_LMEM -DDXA_DOUBLE_BUFFER -DVX_CFG_LMEM_LOG_SIZE=15 -DVX_CFG_DXA_MAX_INFLIGHT=32"
MCB="-DDXA_MCAST_A $BUNDLE"

# k128 story
run $APP "-DCSTORE_LMEM"                       cstore_k128       128
run $APP "-DCSTORE_LMEM -DVX_CFG_TCU_FEDP2K"   cstore2k_k128     128
run $APP "$BUNDLE"                             bundle_k128       128
run $APP "$BUNDLE -DVX_CFG_TCU_FEDP2K"         bundle2k_k128     128

# k512 regime
run $APP ""                                    base_k512         512
run $APP "-DVX_CFG_TCU_FEDP2K"                 fedp2k_k512       512
run $APP "$BUNDLE"                             bundle_k512       512
run $APP "$BUNDLE -DVX_CFG_TCU_FEDP2K"         bundle2k_k512     512
run $MC  "$MCB"                                mcbundle_k512     512
run $MC  "$MCB -DVX_CFG_TCU_FEDP2K"            mcbundle2k_k512   512

echo "STAGE4-COMPLETE" >> "$STATUS"
