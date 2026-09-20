#!/bin/bash
# TCU_OP model, comparable config: NW=16, NT=32, IW=4, 128x128x128, fp16->fp32, dense.
set -u
BUILD=~/dev/vortex_spg/build
OUT=~/dev/vortex_spg/eval_tcu_vs_wg/fedp2k
STATUS=$OUT/status_tcuop.txt
cd "$BUILD"
: > "$STATUS"
CFG="-DTCU_OP -DVX_CFG_TCU_TYPE=DPI -DVX_CFG_NUM_THREADS=32 -DVX_CFG_NUM_WARPS=16 -DVX_CFG_ISSUE_WIDTH=4 -DVX_CFG_EXT_TCU_ENABLE -DVX_CFG_EXT_DXA_ENABLE -DITYPE=fp16 -DOTYPE=fp32 -DSGEMM_CONST_M=128 -DSGEMM_CONST_N=128 -DSGEMM_CONST_K=128 -DSGEMM_CONST_SPARSITY=0 -DTCU_FEOP_BLOCK_M_OVERRIDE=2 -DTCU_FEOP_BLOCK_N_OVERRIDE=16 -DTCU_FEOP_XBAR_QUEUE_DEPTH_OVERRIDE=2 -DVX_CFG_LMEM_LOG_SIZE=16 -DSGEMM_TILE_K_MULT=4"
echo "[$(date +%H:%M:%S)] START tcuop_128" >> "$STATUS"
make -C tests/regression/sgemm_tcu_op clean >/dev/null 2>&1
if ! CONFIGS="$CFG" make -C tests/regression/sgemm_tcu_op > $OUT/tcuop_128.log 2>&1; then
  echo "[$(date +%H:%M:%S)] BUILD-FAIL tcuop_128" >> "$STATUS"; exit 1
fi
t0=$(date +%s)
CONFIGS="$CFG" ./ci/blackbox.sh --driver=rtlsim --app=sgemm_tcu_op \
  --args="-m 128 -n 128 -k 128 -s 0" --perf=1 >> $OUT/tcuop_128.log 2>&1
rc=$?; t1=$(date +%s)
echo "[$(date +%H:%M:%S)] DONE tcuop_128 rc=$rc wall=$((t1-t0))s" >> "$STATUS"
echo "TCUOP-COMPLETE" >> "$STATUS"
