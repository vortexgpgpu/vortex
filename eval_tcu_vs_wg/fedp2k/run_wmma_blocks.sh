#!/bin/bash
# WMMA (sgemm_tcu) at 128x128x128, NW=16 NT=32, rtlsim, scaling the *backend*
# block counts with ISSUE_WIDTH instead of leaving them at 1:
#     VX_CFG_NUM_LSU_BLOCKS = VX_CFG_NUM_ALU_BLOCKS = VX_CFG_NUM_FPU_BLOCKS = IW
#
# WHY: the IW sweep showed WMMA flat across IW (97,797 / 99,990 / 99,927) while
# WGMMA scaled (98,370 / 73,427 / 63,501). The counters said WMMA is bound on
# scoreboard stalls behind ~1.09M global loads at ~67-70 cycle latency, with
# tcu=0% and lsu=0%. VX_config.toml scales NUM_TCU_BLOCKS with IW (line 260) but
# pins NUM_LSU_BLOCKS/NUM_ALU_BLOCKS/NUM_FPU_BLOCKS at 1 (lines 93/85/110), so
# widening IW multiplied the one resource WMMA had spare. This run multiplies the
# backend instead.
#
# WHAT THIS DOES AND DOES NOT WIDEN (verified via gen_config.py --format cflags):
#   LSU_PENDING_SIZE = 32 PER BLOCK  -> total outstanding loads 32/64/128  SCALES
#   DCACHE_NUM_REQS  = NUM_LSU_BLOCKS * DCACHE_CHANNELS                   SCALES
#   DCACHE_MSHR_SIZE = 16            -> constant                          DOES NOT
#   DCACHE_NUM_BANKS = 1             -> constant                          DOES NOT
# So it raises core-side memory-level parallelism but not cache-side miss
# capacity. A null result would point at the 16-entry MSHR / single bank.
#
# Baselines to compare against (already in iw_sweep_results.txt, same app, same
# size, same driver, --perf=1, blocks left at 1):
#   sw_wmma_iw1_rtl  97797 cycles  21999 instrs
#   sw_wmma_iw2_rtl  99990 cycles  21999 instrs
#   sw_wmma_iw4_rtl  99927 cycles  21999 instrs
#
# The IW=1 case here is config-identical to the IW=1 baseline (blocks=1 either
# way). It is included as a harness control: if it does not reproduce 97,797
# then something other than the block counts differs and the comparison is void.
set -u
BUILD=~/dev/vortex_spg/build
OUT=~/dev/vortex_spg/eval_tcu_vs_wg/fedp2k
STATUS=$OUT/status_wmma_blocks.txt
RES=$OUT/wmma_blocks_results.txt
cd "$BUILD" || exit 1
: > "$STATUS"
: > "$RES"

# Identical to run_iw_sweep.sh COMMON so the comparison is apples-to-apples.
COMMON="-DVX_CFG_NUM_THREADS=32 -DVX_CFG_NUM_WARPS=16 -DITYPE=fp16 -DOTYPE=fp32"

# $1=IW  $2=blocks  $3=tag
run(){
  local IW=$1 NB=$2 TAG=$3
  local CFG="$COMMON -DVX_CFG_ISSUE_WIDTH=$IW -DVX_CFG_EXT_TCU_ENABLE \
-DVX_CFG_NUM_LSU_BLOCKS=$NB -DVX_CFG_NUM_ALU_BLOCKS=$NB -DVX_CFG_NUM_FPU_BLOCKS=$NB"
  echo "[$(date +%H:%M:%S)] START $TAG (IW=$IW blocks=$NB)" >> "$STATUS"
  local t0=$(date +%s)
  make -C tests/regression/sgemm_tcu clean >/dev/null 2>&1
  if ! CONFIGS="$CFG" make -C tests/regression/sgemm_tcu > $OUT/${TAG}.log 2>&1; then
    echo "[$(date +%H:%M:%S)] BUILD-FAIL $TAG" >> "$STATUS"
    echo "$TAG BUILD-FAIL" >> "$RES"; return
  fi
  CONFIGS="$CFG" timeout 5400 ./ci/blackbox.sh --driver=rtlsim --app=sgemm_tcu \
    --args="-m 128 -n 128 -k 128" --perf=1 >> $OUT/${TAG}.log 2>&1
  local rc=$? t1=$(date +%s)
  local cyc ins verdict
  cyc=$(grep -aoE "cycles=[0-9]+" $OUT/${TAG}.log | tail -1 | cut -d= -f2)
  ins=$(grep -aoE "instrs=[0-9]+" $OUT/${TAG}.log | tail -1 | cut -d= -f2)
  verdict=$(grep -aoE "PASSED|FAILED" $OUT/${TAG}.log | tail -1)
  [ $rc -eq 124 ] && verdict=TIMEOUT
  echo "$TAG IW=$IW blocks=$NB rc=$rc wall=$((t1-t0))s cycles=${cyc:-NA} instrs=${ins:-NA} ${verdict:-NA}" >> "$RES"
  echo "[$(date +%H:%M:%S)] DONE $TAG rc=$rc wall=$((t1-t0))s cycles=${cyc:-NA} ${verdict:-NA}" >> "$STATUS"
}

run 1 1 "wb_wmma_iw1_nb1"   # harness control, must reproduce 97797
run 2 2 "wb_wmma_iw2_nb2"
run 4 4 "wb_wmma_iw4_nb4"

echo "[$(date +%H:%M:%S)] WMMA-BLOCKS-COMPLETE" >> "$STATUS"
