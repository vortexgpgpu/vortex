#!/bin/bash
# Re-run the rtlsim DXA sweeps on the improved architecture
# (grouped scatter drain fa1765c85 + kernel v3 bd1420351, single-buffer default)
set -u
BUILD=~/dev/vortex_spg/build
LOGDIR=~/dev/vortex_spg/eval_tcu_vs_wg/improved
STATUS=$LOGDIR/status.txt
cd "$BUILD"
: > "$STATUS"

run(){
  local APP=$1 NT=$2 IW=$3 EXTRA=$4 TAG=$5
  local CFG="-DVX_CFG_NUM_THREADS=$NT -DVX_CFG_NUM_WARPS=16 -DVX_CFG_ISSUE_WIDTH=$IW -DVX_CFG_EXT_TCU_ENABLE -DVX_CFG_TCU_WGMMA_ENABLE -DVX_CFG_EXT_DXA_ENABLE $EXTRA -DITYPE=fp16 -DOTYPE=fp32 -DWGMMA_NRC=16"
  local LOG=$LOGDIR/${TAG}.log
  echo "[$(date +%H:%M:%S)] START $TAG" >> "$STATUS"
  make -C tests/regression/$APP clean > /dev/null 2>&1
  if ! CONFIGS="$CFG" make -C tests/regression/$APP > "$LOG" 2>&1; then
    echo "[$(date +%H:%M:%S)] APP-BUILD-FAIL $TAG" >> "$STATUS"; return
  fi
  t0=$(date +%s)
  CONFIGS="$CFG" ./ci/blackbox.sh --driver=rtlsim --app=$APP \
    --args="-m 128 -n 128 -k 128" --perf=1 >> "$LOG" 2>&1
  rc=$?
  t1=$(date +%s)
  echo "[$(date +%H:%M:%S)] DONE $TAG rc=$rc wall=$((t1-t0))s" >> "$STATUS"
}

for NT in 16 8 4; do
  for IW in 4 2 1; do
    run sgemm_tcu_wg_dxa $NT $IW "" "dxa_nt${NT}_iw${IW}"
  done
done
for NT in 16 8; do
  for IW in 4 2 1; do
    run sgemm_tcu_wg_dxa $NT $IW "-DVX_CFG_TCU_FEDP2K" "dxa2k_nt${NT}_iw${IW}"
  done
done
echo "SWEEP-IMPROVED-COMPLETE" >> "$STATUS"
