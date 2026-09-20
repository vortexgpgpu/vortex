#!/bin/bash
# Fast TCU_OP iteration harness: one plain (non-debug) rtlsim run at a given
# size/IW. Usage: tcuop_try.sh <SIZE> <IW> <tag>
set -u
BUILD=~/dev/vortex_spg/build
OUT=~/dev/vortex_spg/eval_tcu_vs_wg/fedp2k
SIZE=${1:-128}
IW=${2:-4}
TAG=${3:-try}
cd "$BUILD"
CFG="-DTCU_OP -DVX_CFG_TCU_TYPE=DPI -DVX_CFG_NUM_THREADS=32 -DVX_CFG_NUM_WARPS=16 \
-DVX_CFG_ISSUE_WIDTH=$IW -DVX_CFG_EXT_TCU_ENABLE -DVX_CFG_EXT_DXA_ENABLE -DITYPE=fp16 -DOTYPE=fp32 \
-DSGEMM_CONST_M=$SIZE -DSGEMM_CONST_N=$SIZE -DSGEMM_CONST_K=$SIZE -DSGEMM_CONST_SPARSITY=0 \
-DTCU_FEOP_BLOCK_M_OVERRIDE=2 -DTCU_FEOP_BLOCK_N_OVERRIDE=16 \
-DTCU_FEOP_XBAR_QUEUE_DEPTH_OVERRIDE=2 -DVX_CFG_LMEM_LOG_SIZE=16 -DSGEMM_TILE_K_MULT=4"
: > $OUT/${TAG}.log
make -C tests/regression/sgemm_tcu_op clean >/dev/null 2>&1
if ! CONFIGS="$CFG" make -C tests/regression/sgemm_tcu_op >> $OUT/${TAG}.log 2>&1; then
  echo "RESULT $TAG BUILD-FAIL"; exit 1
fi
t0=$(date +%s)
CONFIGS="$CFG" timeout 3600 ./ci/blackbox.sh --driver=rtlsim --app=sgemm_tcu_op \
  --args="-m $SIZE -n $SIZE -k $SIZE -s 0" --perf=1 >> $OUT/${TAG}.log 2>&1
rc=$?; t1=$(date +%s)
cyc=$(grep -aoE "cycles=[0-9]+" $OUT/${TAG}.log | tail -1 | cut -d= -f2)
ins=$(grep -aoE "instrs=[0-9]+" $OUT/${TAG}.log | tail -1 | cut -d= -f2)
v=$(grep -aoE "PASSED|FAILED" $OUT/${TAG}.log | tail -1)
[ $rc -eq 124 ] && v=TIMEOUT
[ -z "$v" ] && v="DEADLOCK/ABORT(rc=$rc)"
echo "RESULT $TAG size=$SIZE iw=$IW rc=$rc wall=$((t1-t0))s cycles=${cyc:-NA} instrs=${ins:-NA} $v"
