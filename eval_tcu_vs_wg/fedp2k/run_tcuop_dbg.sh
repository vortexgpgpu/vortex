#!/bin/bash
# TCU_OP hang diagnosis: minimal single-tile config (32x32x64) with debug traces.
set -u
BUILD=~/dev/vortex_spg/build
OUT=~/dev/vortex_spg/eval_tcu_vs_wg/fedp2k
STATUS=$OUT/status_tcuop_dbg.txt
cd "$BUILD"
: > "$STATUS"
CFG="-DTCU_OP -DVX_CFG_TCU_TYPE=DPI -DVX_CFG_NUM_THREADS=32 -DVX_CFG_NUM_WARPS=16 -DVX_CFG_ISSUE_WIDTH=4 -DVX_CFG_EXT_TCU_ENABLE -DVX_CFG_EXT_DXA_ENABLE -DITYPE=fp16 -DOTYPE=fp32 -DSGEMM_CONST_M=128 -DSGEMM_CONST_N=128 -DSGEMM_CONST_K=128 -DSGEMM_CONST_SPARSITY=0 -DTCU_FEOP_BLOCK_M_OVERRIDE=2 -DTCU_FEOP_BLOCK_N_OVERRIDE=16 -DTCU_FEOP_XBAR_QUEUE_DEPTH_OVERRIDE=2 -DVX_CFG_LMEM_LOG_SIZE=16 -DSGEMM_TILE_K_MULT=4"
echo "[$(date +%H:%M:%S)] START tcuop_dbg" >> "$STATUS"
make -C tests/regression/sgemm_tcu_op clean >/dev/null 2>&1
if ! CONFIGS="$CFG" make -C tests/regression/sgemm_tcu_op > $OUT/tcuop_dbg.log 2>&1; then
  echo "[$(date +%H:%M:%S)] BUILD-FAIL tcuop_dbg" >> "$STATUS"; exit 1
fi
echo "[$(date +%H:%M:%S)] APP-BUILT, launching rtlsim (debug)" >> "$STATUS"
t0=$(date +%s)
# timeout guard: a single 32x32x64 tile should simulate in well under 20 min
# even with debug tracing; anything longer is the hang reproducing.
CONFIGS="$CFG" timeout 1500 ./ci/blackbox.sh --driver=rtlsim --app=sgemm_tcu_op \
  --args="-m 64 -n 128 -k 128 -s 0" --debug=2 >> $OUT/tcuop_dbg.log 2>&1
rc=$?; t1=$(date +%s)
echo "[$(date +%H:%M:%S)] DONE tcuop_dbg rc=$rc wall=$((t1-t0))s" >> "$STATUS"
echo "TCUOP-DBG-COMPLETE" >> "$STATUS"
