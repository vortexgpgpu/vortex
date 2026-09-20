#!/bin/bash
# extension sweep: sgemm_tcu_wg_dxa (WGMMA + DXA-fed tile buffers)
# same matrix: M=N=K=128, fp16->fp32, NW=16, NT in {4,8,16}, IW in {1,2,4}, --perf=1
set -u
BUILD=~/dev/vortex_spg/build
LOGDIR=~/dev/vortex_spg/eval_tcu_vs_wg
STATUS=$LOGDIR/status_2k.txt
cd "$BUILD"
: > "$STATUS"

APP=sgemm_tcu_wg_dxa
SUFFIX=_2k
for NT in 8 16; do
  for IW in 1 2 4; do
    CFG="-DVX_CFG_NUM_THREADS=$NT -DVX_CFG_NUM_WARPS=16 -DVX_CFG_ISSUE_WIDTH=$IW -DVX_CFG_EXT_TCU_ENABLE -DVX_CFG_TCU_WGMMA_ENABLE -DVX_CFG_EXT_DXA_ENABLE -DVX_CFG_TCU_FEDP2K -DITYPE=fp16 -DOTYPE=fp32 -DWGMMA_NRC=16"
    TAG="${APP}${SUFFIX}_nt${NT}_iw${IW}"
    LOG=$LOGDIR/${TAG}.log
    echo "[$(date +%H:%M:%S)] START $TAG" >> "$STATUS"
    make -C tests/regression/$APP clean > /dev/null 2>&1
    if ! CONFIGS="$CFG" make -C tests/regression/$APP > "$LOG" 2>&1; then
      echo "[$(date +%H:%M:%S)] APP-BUILD-FAIL $TAG" >> "$STATUS"
      continue
    fi
    t0=$(date +%s)
    CONFIGS="$CFG" ./ci/blackbox.sh --driver=rtlsim --app=$APP \
      --args="-m 128 -n 128 -k 128" --perf=1 >> "$LOG" 2>&1
    rc=$?
    t1=$(date +%s)
    echo "[$(date +%H:%M:%S)] DONE $TAG rc=$rc wall=$((t1-t0))s" >> "$STATUS"
  done
done
echo "SWEEP-COMPLETE" >> "$STATUS"
