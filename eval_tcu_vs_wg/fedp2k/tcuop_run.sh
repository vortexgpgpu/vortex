#!/bin/bash
# General TCU_OP rtlsim harness (plain / non-debug build).
# Usage: tcuop_run.sh <M> <N> <K> <IW> <TILE_K_MULT> <tag> [extra CONFIGS]
set -u
BUILD=~/dev/vortex_spg/build
OUT=~/dev/vortex_spg/eval_tcu_vs_wg/fedp2k
M=$1; N=$2; K=$3; IW=$4; KM=$5; TAG=$6; EXTRA=${7:-}
cd "$BUILD"
CFG="-DTCU_OP -DVX_CFG_TCU_TYPE=DPI -DVX_CFG_NUM_THREADS=32 -DVX_CFG_NUM_WARPS=16 \
-DVX_CFG_ISSUE_WIDTH=$IW -DVX_CFG_EXT_TCU_ENABLE -DVX_CFG_EXT_DXA_ENABLE -DITYPE=fp16 -DOTYPE=fp32 \
-DSGEMM_CONST_M=$M -DSGEMM_CONST_N=$N -DSGEMM_CONST_K=$K -DSGEMM_CONST_SPARSITY=0 \
-DTCU_FEOP_BLOCK_M_OVERRIDE=2 -DTCU_FEOP_BLOCK_N_OVERRIDE=16 \
-DTCU_FEOP_XBAR_QUEUE_DEPTH_OVERRIDE=2 -DVX_CFG_LMEM_LOG_SIZE=16 -DSGEMM_TILE_K_MULT=$KM $EXTRA"
: > $OUT/${TAG}.log
make -C tests/regression/sgemm_tcu_op clean >/dev/null 2>&1
if ! CONFIGS="$CFG" make -C tests/regression/sgemm_tcu_op >> $OUT/${TAG}.log 2>&1; then
  echo "RESULT $TAG BUILD-FAIL"
  grep -aiE "error|assert" $OUT/${TAG}.log | grep -av "^ " | tail -5
  exit 1
fi
t0=$(date +%s)
CONFIGS="$CFG" timeout 5400 ./ci/blackbox.sh --driver=rtlsim --app=sgemm_tcu_op \
  --args="-m $M -n $N -k $K -s 0" --perf=1 >> $OUT/${TAG}.log 2>&1
rc=$?; t1=$(date +%s)
# Pull from the PERF summary line specifically -- a plain "cycles=" grep also
# matches the engine's "Full-queue stall cycles=0" traces.
perfline=$(grep -aoE "PERF: instrs=[0-9]+, cycles=[0-9]+" $OUT/${TAG}.log | tail -1)
cyc=$(echo "$perfline" | grep -oE "cycles=[0-9]+" | cut -d= -f2)
ins=$(echo "$perfline" | grep -oE "instrs=[0-9]+" | cut -d= -f2)
kbc=$(grep -aoE "Kernel body cycles: [0-9]+" $OUT/${TAG}.log | tail -1 | grep -oE "[0-9]+")
v=$(grep -aoE "PASSED|FAILED" $OUT/${TAG}.log | tail -1)
[ $rc -eq 124 ] && v=TIMEOUT
[ -z "$v" ] && v="DEADLOCK/ABORT(rc=$rc)"
echo "RESULT $TAG ${M}x${N}x${K} iw=$IW kmult=$KM rc=$rc wall=$((t1-t0))s cycles=${cyc:-NA} instrs=${ins:-NA} kbody=${kbc:-NA} $v"
