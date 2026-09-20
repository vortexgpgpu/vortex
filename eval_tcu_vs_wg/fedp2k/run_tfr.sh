#!/bin/bash
# TCU_OP datapath regression for the exact-product (TFR) path.
#   usage: run_tfr.sh PHASE TYPE [TAG:EXTRA_DEFINES:ARGS ...]
# TYPE selects the datapath backend (TFR = new, DPI = legacy reference).
set -u
PHASE=$1; shift
TYPE=$1; shift
BUILD=~/dev/vortex_spg/build
OUT=~/dev/vortex_spg/eval_tcu_vs_wg/fedp2k/tfr
mkdir -p $OUT
RES=$OUT/results.txt
cd "$BUILD" || exit 1

BASE="-DTCU_OP -DVX_CFG_TCU_TYPE=$TYPE -DVX_CFG_NUM_THREADS=32 -DVX_CFG_NUM_WARPS=16 \
-DVX_CFG_ISSUE_WIDTH=4 -DVX_CFG_EXT_TCU_ENABLE -DVX_CFG_EXT_DXA_ENABLE \
-DITYPE=fp16 -DOTYPE=fp32 -DSGEMM_CONST_SPARSITY=${SPARSITY:-0} \
-DTCU_FEOP_BLOCK_M_OVERRIDE=2 -DTCU_FEOP_BLOCK_N_OVERRIDE=16 \
-DVX_CFG_LMEM_LOG_SIZE=16 -DVX_DBG_DEBUG_LEVEL=0"

for spec in "$@"; do
  IFS=: read -r TAG EXTRA ARGS <<< "$spec"
  [ -z "$ARGS" ] && ARGS="-m 128 -n 128 -k 128 -s 0"
  CFG="$BASE $EXTRA"
  LOG=$OUT/${PHASE}_${TAG}.log
  echo "[$(date +%H:%M:%S)] START $PHASE/$TYPE $TAG" >> $OUT/status.txt
  make -C tests/regression/sgemm_tcu_op clean >/dev/null 2>&1
  if ! CONFIGS="$CFG" make -C tests/regression/sgemm_tcu_op > $LOG 2>&1; then
    echo "$PHASE/$TYPE $TAG BUILD-FAIL(kernel)" | tee -a $RES >> $OUT/status.txt; continue
  fi
  t0=$(date +%s)
  CONFIGS="$CFG" timeout 3600 ./ci/blackbox.sh --driver=rtlsim --app=sgemm_tcu_op \
    --args="$ARGS" --perf=1 >> $LOG 2>&1
  rc=$?; t1=$(date +%s)
  cyc=$(grep -aoE "PERF: instrs=[0-9]+, cycles=[0-9]+" $LOG | tail -1 | grep -oE "cycles=[0-9]+" | cut -d= -f2)
  v=$(grep -aoE "PASSED|FAILED" $LOG | tail -1)
  [ $rc -eq 124 ] && v=TIMEOUT
  [ -z "$v" ] && v="ABORT(rc=$rc)"
  asrt=$(grep -ac "Assertion failed\|\*\*\* " $LOG)
  echo "$PHASE/$TYPE $TAG cycles=${cyc:-NA} $v asserts=$asrt wall=$((t1-t0))s" | tee -a $RES >> $OUT/status.txt
done
echo "[$(date +%H:%M:%S)] PHASE-DONE $PHASE/$TYPE" >> $OUT/status.txt
