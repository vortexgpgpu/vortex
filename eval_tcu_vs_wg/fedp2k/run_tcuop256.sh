#!/bin/bash
# TCU_OP at 256x256x256, rtlsim, IW in {1,2,4} -- to slot into the 256^3
# WGMMA-vs-WMMA comparison table. NW=16, NT=32, fp16->fp32, dense (s=0).
# TCU_OP has no simx model (no MMA_OP in sim/simx), so rtlsim only.
#
# Known blocker P0.2: non-debug rtlsim builds deadlock where --debug=2 passes.
# Each IW is therefore attempted twice: plain first (cheap, fails fast if the
# deadlock reproduces), then --debug=2 as the only path known to complete.
set -u
BUILD=~/dev/vortex_spg/build
OUT=~/dev/vortex_spg/eval_tcu_vs_wg/fedp2k
STATUS=$OUT/status_tcuop256.txt
RES=$OUT/tcuop256_results.txt
cd "$BUILD"
: > "$STATUS"
: > "$RES"

SZ="-m 256 -n 256 -k 256 -s 0"
BASE="-DTCU_OP -DVX_CFG_TCU_TYPE=DPI -DVX_CFG_NUM_THREADS=32 -DVX_CFG_NUM_WARPS=16 \
-DVX_CFG_EXT_TCU_ENABLE -DVX_CFG_EXT_DXA_ENABLE -DITYPE=fp16 -DOTYPE=fp32 \
-DSGEMM_CONST_M=256 -DSGEMM_CONST_N=256 -DSGEMM_CONST_K=256 -DSGEMM_CONST_SPARSITY=0 \
-DTCU_FEOP_BLOCK_M_OVERRIDE=2 -DTCU_FEOP_BLOCK_N_OVERRIDE=16 \
-DTCU_FEOP_XBAR_QUEUE_DEPTH_OVERRIDE=2 -DVX_CFG_LMEM_LOG_SIZE=16 -DSGEMM_TILE_K_MULT=4"

# $1=IW $2=tag $3=dbgflag ("" or "--debug=2") $4=timeout
run(){
  local IW=$1 TAG=$2 DBG=$3 TMO=$4
  local CFG="$BASE -DVX_CFG_ISSUE_WIDTH=$IW"
  echo "[$(date +%H:%M:%S)] START $TAG" >> "$STATUS"
  make -C tests/regression/sgemm_tcu_op clean >/dev/null 2>&1
  if ! CONFIGS="$CFG" make -C tests/regression/sgemm_tcu_op > $OUT/${TAG}.log 2>&1; then
    echo "[$(date +%H:%M:%S)] BUILD-FAIL $TAG" >> "$STATUS"
    echo "$TAG BUILD-FAIL" >> "$RES"; return 1
  fi
  local t0=$(date +%s)
  if [ -z "$DBG" ]; then
    CONFIGS="$CFG" timeout $TMO ./ci/blackbox.sh --driver=rtlsim --app=sgemm_tcu_op \
      --args="$SZ" --perf=1 >> $OUT/${TAG}.log 2>&1
  else
    CONFIGS="$CFG" timeout $TMO ./ci/blackbox.sh --driver=rtlsim --app=sgemm_tcu_op \
      --args="$SZ" $DBG >> $OUT/${TAG}.log 2>&1
  fi
  local rc=$? t1=$(date +%s)
  # --debug=2 sends the run to run.log inside the app build dir
  local RUNLOG=tests/regression/sgemm_tcu_op/run.log
  local SRC=$OUT/${TAG}.log
  [ -n "$DBG" ] && [ -f "$RUNLOG" ] && { cp "$RUNLOG" $OUT/${TAG}_run.log; SRC=$OUT/${TAG}_run.log; }
  local cyc ins verdict
  cyc=$(grep -aoE "cycles=[0-9]+" "$SRC" | tail -1 | cut -d= -f2)
  ins=$(grep -aoE "instrs=[0-9]+" "$SRC" | tail -1 | cut -d= -f2)
  verdict=$(grep -aoE "PASSED|FAILED" "$SRC" | tail -1)
  [ $rc -eq 124 ] && verdict="TIMEOUT"
  [ -z "$verdict" ] && verdict="DEADLOCK/ABORT(rc=$rc)"
  echo "$TAG rc=$rc wall=$((t1-t0))s cycles=${cyc:-NA} instrs=${ins:-NA} $verdict" >> "$RES"
  echo "[$(date +%H:%M:%S)] DONE $TAG rc=$rc wall=$((t1-t0))s cycles=${cyc:-NA} $verdict" >> "$STATUS"
  [ "$verdict" = "PASSED" ] && return 0 || return 1
}

# Pass 1: plain (non-debug) builds -- the configuration the WGMMA table used.
for IW in 1 2 4; do run $IW "t256_op_iw${IW}_rtl" "" 5400; done
echo "PLAIN-PASS-DONE" >> "$STATUS"

# Pass 2: --debug=2 fallback, only for the IWs that did not pass above.
for IW in 1 2 4; do
  if grep -q "t256_op_iw${IW}_rtl .* PASSED" "$RES"; then
    echo "[$(date +%H:%M:%S)] SKIP dbg IW=$IW (plain passed)" >> "$STATUS"
  else
    run $IW "t256_op_iw${IW}_dbg" "--debug=2" 10800
  fi
done

echo "TCUOP256-COMPLETE" >> "$STATUS"
