#!/bin/bash
# Isolate WHICH block knob caused the WGMMA SS regression at IW=4.
#
# run_blocks_symmetric.sh moved three knobs together:
#     NUM_LSU_BLOCKS = NUM_ALU_BLOCKS = NUM_FPU_BLOCKS = IW
# and SS went 63,175 -> 66,621 (+5.45%) while RS was flat (-0.23%) and WMMA
# improved 26.7%. I attributed the SS regression to LSU arbitration latency on
# the strength of load_lat rising 180.13 -> 407.14. That is consistent with the
# LSU story but does NOT isolate it: a regression from the ALU or FPU split
# would look the same in total cycles, and a unit split into per-issue-slot
# blocks changes DISPATCH PARTITIONING as well as width -- a warp bound to issue
# slot i can only use block i, so static partitioning can create imbalance that
# a single shared block does not have. Bandwidth up, utilisation down.
#
# One knob at a time, everything else at 1, all at IW=4:
#   ab_ss_lsu4   LSU=4 ALU=1 FPU=1
#   ab_ss_alu4   LSU=1 ALU=4 FPU=1
#   ab_ss_fpu4   LSU=1 ALU=1 FPU=4
# Reference points already measured (same app/size/driver/--perf=1):
#   SS IW=4 all-blocks=1  63,175   <- baseline
#   SS IW=4 all-blocks=4  66,621   <- +5.45%
#
# The three deltas should roughly account for the combined +3,446 cycles. If
# none of them reproduces it, the effect is an interaction between knobs and the
# single-knob framing (mine included) is wrong.
#
# WMMA control at LSU=4 only: if WMMA's 26.7% gain is really the load pool, it
# should survive with ALU/FPU left at 1. If it does not, my WMMA explanation is
# also wrong and both results need re-reading.
set -u
BUILD=~/dev/vortex_spg/build
OUT=~/dev/vortex_spg/eval_tcu_vs_wg/fedp2k
STATUS=$OUT/status_ss_ablation.txt
RES=$OUT/ss_ablation_results.txt
cd "$BUILD" || exit 1
: > "$STATUS"
: > "$RES"

COMMON="-DVX_CFG_NUM_THREADS=32 -DVX_CFG_NUM_WARPS=16 -DITYPE=fp16 -DOTYPE=fp32"
WG="-DVX_CFG_TCU_WGMMA_ENABLE -DVX_CFG_EXT_DXA_ENABLE -DVX_CFG_TCU_FEDP2K -DWGMMA_NRC=16"
WGSS="$WG -DWGMMA_SS"

# $1=app $2=extra $3=lsu $4=alu $5=fpu $6=tag
run(){
  local APP=$1 EXTRA=$2 L=$3 A=$4 F=$5 TAG=$6
  local CFG="$COMMON -DVX_CFG_ISSUE_WIDTH=4 $EXTRA \
-DVX_CFG_NUM_LSU_BLOCKS=$L -DVX_CFG_NUM_ALU_BLOCKS=$A -DVX_CFG_NUM_FPU_BLOCKS=$F"
  echo "[$(date +%H:%M:%S)] START $TAG (LSU=$L ALU=$A FPU=$F)" >> "$STATUS"
  local t0=$(date +%s)
  make -C tests/regression/$APP clean >/dev/null 2>&1
  if ! CONFIGS="$CFG" make -C tests/regression/$APP > $OUT/${TAG}.log 2>&1; then
    echo "[$(date +%H:%M:%S)] BUILD-FAIL $TAG" >> "$STATUS"
    echo "$TAG LSU=$L ALU=$A FPU=$F BUILD-FAIL" >> "$RES"; return
  fi
  CONFIGS="$CFG" timeout 7200 ./ci/blackbox.sh --driver=rtlsim --app=$APP \
    --args="-m 128 -n 128 -k 128" --perf=1 >> $OUT/${TAG}.log 2>&1
  local rc=$? t1=$(date +%s)
  local cyc ins verdict llat
  cyc=$(grep -aoE "cycles=[0-9]+" $OUT/${TAG}.log | tail -1 | cut -d= -f2)
  ins=$(grep -aoE "instrs=[0-9]+" $OUT/${TAG}.log | tail -1 | cut -d= -f2)
  llat=$(grep -aoE "load_lat=[0-9.]+" $OUT/${TAG}.log | tail -1 | cut -d= -f2)
  verdict=$(grep -aoE "PASSED|FAILED" $OUT/${TAG}.log | tail -1)
  [ $rc -eq 124 ] && verdict=TIMEOUT
  [ -z "$verdict" ] && verdict="ABORT(rc=$rc)"
  echo "$TAG LSU=$L ALU=$A FPU=$F rc=$rc wall=$((t1-t0))s cycles=${cyc:-NA} instrs=${ins:-NA} load_lat=${llat:-NA} $verdict" >> "$RES"
  echo "[$(date +%H:%M:%S)] DONE $TAG rc=$rc wall=$((t1-t0))s cycles=${cyc:-NA} load_lat=${llat:-NA} $verdict" >> "$STATUS"
}

# SS single-knob ablation
run sgemm_tcu_wg_dxa "-DVX_CFG_EXT_TCU_ENABLE $WGSS" 4 1 1 "ab_ss_lsu4"
run sgemm_tcu_wg_dxa "-DVX_CFG_EXT_TCU_ENABLE $WGSS" 1 4 1 "ab_ss_alu4"
run sgemm_tcu_wg_dxa "-DVX_CFG_EXT_TCU_ENABLE $WGSS" 1 1 4 "ab_ss_fpu4"

# WMMA control: is the 26.7% gain really the LSU pool?
run sgemm_tcu "-DVX_CFG_EXT_TCU_ENABLE" 4 1 1 "ab_wmma_lsu4"

echo "[$(date +%H:%M:%S)] SS-ABLATION-COMPLETE" >> "$STATUS"
