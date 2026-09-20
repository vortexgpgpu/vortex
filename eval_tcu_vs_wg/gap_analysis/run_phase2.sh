#!/bin/bash
# Phase 2: debug traces + variant experiments (simx)
set -u
BUILD=~/dev/vortex_spg/build
OUT=~/dev/vortex_spg/eval_tcu_vs_wg/gap_analysis
STATUS=$OUT/status_phase2.txt
cd "$BUILD"
: > "$STATUS"

CFG_BASE="-DVX_CFG_NUM_THREADS=16 -DVX_CFG_NUM_WARPS=16 -DVX_CFG_EXT_TCU_ENABLE -DVX_CFG_TCU_WGMMA_ENABLE -DVX_CFG_EXT_DXA_ENABLE -DITYPE=fp16 -DOTYPE=fp32 -DWGMMA_NRC=16"

dbg_run() {
  local APP=$1 IW=$2 TAG=$3
  local CFG="$CFG_BASE -DVX_CFG_ISSUE_WIDTH=$IW"
  echo "[$(date +%H:%M:%S)] START-DBG $TAG" >> "$STATUS"
  make -C tests/regression/$APP clean > /dev/null 2>&1
  CONFIGS="$CFG" make -C tests/regression/$APP > $OUT/${TAG}.buildlog 2>&1 || { echo "BUILD-FAIL $TAG" >> "$STATUS"; return; }
  CONFIGS="$CFG" ./ci/blackbox.sh --driver=simx --app=$APP \
    --args="-m 128 -n 128 -k 128" --debug=3 --log=$OUT/${TAG}.trace > $OUT/${TAG}.stdout 2>&1
  echo "[$(date +%H:%M:%S)] DONE-DBG $TAG rc=$? size=$(du -sh $OUT/${TAG}.trace 2>/dev/null | cut -f1)" >> "$STATUS"
}

var_run() {
  local APP=$1 IW=$2 EXTRA=$3 ARGS=$4 TAG=$5
  local CFG="$CFG_BASE -DVX_CFG_ISSUE_WIDTH=$IW $EXTRA"
  echo "[$(date +%H:%M:%S)] START-VAR $TAG" >> "$STATUS"
  make -C tests/regression/$APP clean > /dev/null 2>&1
  CONFIGS="$CFG" make -C tests/regression/$APP > $OUT/${TAG}.log 2>&1 || { echo "BUILD-FAIL $TAG" >> "$STATUS"; return; }
  CONFIGS="$CFG" ./ci/blackbox.sh --driver=simx --app=$APP \
    --args="$ARGS" --perf=1 >> $OUT/${TAG}.log 2>&1
  echo "[$(date +%H:%M:%S)] DONE-VAR $TAG rc=$?" >> "$STATUS"
}

dbg_run sgemm_tcu_wg_dxa 4 trace_wg_dxa_iw4
dbg_run sgemm_tcu        4 trace_sgemm_tcu_iw4
dbg_run sgemm_tcu_wg_dxa 1 trace_wg_dxa_iw1

var_run sgemm_tcu_wg_dxa 4 "-DSW_LOAD_B"          "-m 128 -n 128 -k 128" var_swloadB_iw4
var_run sgemm_tcu_wg_dxa 4 "-DSW_LOAD_A"          "-m 128 -n 128 -k 128" var_swloadA_iw4
var_run sgemm_tcu_wg_dxa 4 "-DDXA_DOUBLE_BUFFER"  "-m 128 -n 128 -k 128" var_dbuf_iw4
var_run sgemm_tcu_wg_dxa 4 ""                     "-m 128 -n 128 -k 512" var_k512_dxa_iw4
var_run sgemm_tcu        4 ""                     "-m 128 -n 128 -k 512" var_k512_base_iw4
echo "PHASE2-COMPLETE" >> "$STATUS"
