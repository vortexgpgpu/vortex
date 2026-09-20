#!/bin/bash
# Fresh --debug=3 rtlsim trace of TCU_OP for Perfetto analysis.
# Config = ci/testcases/tensor_op.yaml tcu_op-dxa-stress-128 (128^3, IW=4,
# dense, kmult=4 -> the 81,087-cycle headline point), minus
# -DVX_DBG_DEBUG_LEVEL=0: DEBUG=3 sets the level itself, and a 0 there would
# silence the trace.
#
# --debug=3 enables every DBG_TRACE_* category (pipeline, mem, cache, tcu, ...)
# and prints TRACE levels 1..3. Level 3 adds cache bank/MSHR events. Pipeline
# stage traces (the exporter's instruction slices) exist ONLY in --debug
# builds; the earlier non-debug log had engine lines but no pipeline.
set -u
BUILD=~/dev/vortex_spg/build
OUT=~/dev/vortex_spg/eval_tcu_vs_wg/fedp2k/perfetto_tcuop
cd "$BUILD" || exit 1

CFG="-DTCU_OP -DVX_CFG_TCU_TYPE=DPI -DVX_CFG_NUM_THREADS=32 -DVX_CFG_NUM_WARPS=16 \
-DVX_CFG_ISSUE_WIDTH=4 -DVX_CFG_EXT_TCU_ENABLE -DVX_CFG_EXT_DXA_ENABLE \
-DITYPE=fp16 -DOTYPE=fp32 -DSGEMM_CONST_M=128 -DSGEMM_CONST_N=128 -DSGEMM_CONST_K=128 \
-DSGEMM_CONST_SPARSITY=0 -DTCU_FEOP_BLOCK_M_OVERRIDE=2 -DTCU_FEOP_BLOCK_N_OVERRIDE=16 \
-DTCU_FEOP_XBAR_QUEUE_DEPTH_OVERRIDE=2 -DVX_CFG_LMEM_LOG_SIZE=16 -DSGEMM_TILE_K_MULT=4 -DDBG_TRACE_DXA"

echo "[$(date +%H:%M:%S)] START trace" > $OUT/status_dxa.txt
make -C tests/regression/sgemm_tcu_op clean >/dev/null 2>&1
if ! CONFIGS="$CFG" make -C tests/regression/sgemm_tcu_op > $OUT/build_dxa.log 2>&1; then
  echo "[$(date +%H:%M:%S)] BUILD-FAIL" >> $OUT/status_dxa.txt; exit 1
fi
t0=$(date +%s)
CONFIGS="$CFG" timeout 10800 ./ci/blackbox.sh --driver=rtlsim --app=sgemm_tcu_op \
  --args="-m 128 -n 128 -k 128 -s 0" --perf=1 --debug=3 --log=$OUT/trace_iw4_dxa.log \
  > $OUT/blackbox_dxa.log 2>&1
rc=$?; t1=$(date +%s)
v=$(grep -aoE "PASSED|FAILED" $OUT/trace_iw4_dxa.log | tail -1)
[ $rc -eq 124 ] && v=TIMEOUT
c=$(grep -aoE "PERF: instrs=[0-9]+, cycles=[0-9]+" $OUT/trace_iw4_dxa.log | tail -1)
echo "[$(date +%H:%M:%S)] DONE rc=$rc wall=$((t1-t0))s ${v:-ABORT} ${c:-no-perf} size=$(du -h $OUT/trace_iw4_dxa.log | cut -f1)" >> $OUT/status_dxa.txt
