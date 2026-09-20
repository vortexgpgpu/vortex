#!/bin/bash
# Restore symmetry to the 128x128x128 comparison: re-run WGMMA RS, WGMMA SS and
# TCU_OP with the backend block counts scaled with ISSUE_WIDTH, the same way
# run_wmma_blocks.sh did for WMMA.
#
#     VX_CFG_NUM_LSU_BLOCKS = VX_CFG_NUM_ALU_BLOCKS = VX_CFG_NUM_FPU_BLOCKS = IW
#
# WHY: scaling those blocks took WMMA from flat (97,797/99,990/99,927) to
# 97,797/81,033/73,233 -- a 1.36x gain at IW=4 -- by lifting the outstanding-load
# pool from 32 to 32*IW. Measured in-flight loads went 2.91/2.89/2.87 (blocks=1)
# to 2.91/5.79/13.11 (blocks=IW). That leaves the cross-variant table asymmetric
# IN WMMA'S FAVOUR: WMMA now gets IW LSU/ALU/FPU blocks while WGMMA and TCU_OP
# were all measured at blocks=1. Quoting that gap as an ISA result would be wrong.
# This run gives every variant the same backend so the remaining delta is
# attributable to the instruction set and the kernel, not the provisioning.
#
# WGMMA RS/SS baselines (blocks=1) are already comparable: iw_sweep_results.txt
# and ss_sweep_results.txt were produced with --perf=1, same app, same size,
# same driver:
#   RS  98,370 / 73,427 / 63,501      SS  99,065 / 73,654 / 63,175
#
# TCU_OP IS DIFFERENT. Its 81,060 / 81,159 / 81,087 baselines came from the CI
# cases in ci/testcases/tensor_op.yaml, which do NOT pass --perf=1. PERF_ENABLE
# instantiates counter hardware, so comparing a --perf=1 run against those would
# confound the block change with the counter change. This script therefore
# regenerates the TCU_OP blocks=1 baseline under --perf=1 as well, and the
# TCU_OP comparison is made strictly within this script's own runs.
#
# Every variant's IW=1/blocks=1 case is config-identical to its existing
# baseline and acts as a harness control: if it does not reproduce, the
# comparison for that variant is void.
#
# NOTE on ss_sweep_results.txt: its rows read mode=RS-LEAKED. That label is a bug
# in the old script's detection (it grepped the compile flag, not the compiled
# path); those runs are genuinely SS. Not re-litigated here.
set -u
BUILD=~/dev/vortex_spg/build
OUT=~/dev/vortex_spg/eval_tcu_vs_wg/fedp2k
STATUS=$OUT/status_blocks_sym.txt
RES=$OUT/blocks_sym_results.txt
cd "$BUILD" || exit 1
: > "$STATUS"
: > "$RES"

COMMON="-DVX_CFG_NUM_THREADS=32 -DVX_CFG_NUM_WARPS=16 -DITYPE=fp16 -DOTYPE=fp32"

WG="-DVX_CFG_TCU_WGMMA_ENABLE -DVX_CFG_EXT_DXA_ENABLE -DVX_CFG_TCU_FEDP2K -DWGMMA_NRC=16"
WGSS="$WG -DWGMMA_SS"

# TCU_OP: mirrors ci/testcases/tensor_op.yaml tcu_op-dxa-stress-128 (and
# tcu_op-iw1 / tcu_op-iw2, which differ only in ISSUE_WIDTH). kmult=4 matches all
# three 128^3 baselines.
OPCFG="-DTCU_OP -DVX_CFG_TCU_TYPE=DPI -DVX_CFG_EXT_TCU_ENABLE -DVX_CFG_EXT_DXA_ENABLE \
-DSGEMM_CONST_M=128 -DSGEMM_CONST_N=128 -DSGEMM_CONST_K=128 -DSGEMM_CONST_SPARSITY=0 \
-DTCU_FEOP_BLOCK_M_OVERRIDE=2 -DTCU_FEOP_BLOCK_N_OVERRIDE=16 \
-DTCU_FEOP_XBAR_QUEUE_DEPTH_OVERRIDE=2 -DVX_CFG_LMEM_LOG_SIZE=16 \
-DSGEMM_TILE_K_MULT=4 -DVX_DBG_DEBUG_LEVEL=0"

# $1=app $2=extra_cfg $3=IW $4=blocks $5=args $6=tag
run(){
  local APP=$1 EXTRA=$2 IW=$3 NB=$4 ARGS=$5 TAG=$6
  local CFG="$COMMON -DVX_CFG_ISSUE_WIDTH=$IW $EXTRA \
-DVX_CFG_NUM_LSU_BLOCKS=$NB -DVX_CFG_NUM_ALU_BLOCKS=$NB -DVX_CFG_NUM_FPU_BLOCKS=$NB"
  echo "[$(date +%H:%M:%S)] START $TAG (IW=$IW blocks=$NB)" >> "$STATUS"
  local t0=$(date +%s)
  make -C tests/regression/$APP clean >/dev/null 2>&1
  if ! CONFIGS="$CFG" make -C tests/regression/$APP > $OUT/${TAG}.log 2>&1; then
    echo "[$(date +%H:%M:%S)] BUILD-FAIL $TAG" >> "$STATUS"
    echo "$TAG IW=$IW blocks=$NB BUILD-FAIL" >> "$RES"; return
  fi
  CONFIGS="$CFG" timeout 7200 ./ci/blackbox.sh --driver=rtlsim --app=$APP \
    --args="$ARGS" --perf=1 >> $OUT/${TAG}.log 2>&1
  local rc=$? t1=$(date +%s)
  local cyc ins verdict
  cyc=$(grep -aoE "cycles=[0-9]+" $OUT/${TAG}.log | tail -1 | cut -d= -f2)
  ins=$(grep -aoE "instrs=[0-9]+" $OUT/${TAG}.log | tail -1 | cut -d= -f2)
  verdict=$(grep -aoE "PASSED|FAILED" $OUT/${TAG}.log | tail -1)
  [ $rc -eq 124 ] && verdict=TIMEOUT
  [ -z "$verdict" ] && verdict="ABORT(rc=$rc)"
  echo "$TAG IW=$IW blocks=$NB rc=$rc wall=$((t1-t0))s cycles=${cyc:-NA} instrs=${ins:-NA} $verdict" >> "$RES"
  echo "[$(date +%H:%M:%S)] DONE $TAG rc=$rc wall=$((t1-t0))s cycles=${cyc:-NA} $verdict" >> "$STATUS"
}

SZ="-m 128 -n 128 -k 128"
OPSZ="-m 128 -n 128 -k 128 -s 0"

# --- WGMMA RS: blocks=IW (IW=1 doubles as the harness control -> expect 98,370)
for IW in 1 2 4; do
  run sgemm_tcu_wg_dxa "-DVX_CFG_EXT_TCU_ENABLE $WG"   $IW $IW "$SZ" "bs_rs_iw${IW}_nb${IW}"
done

# --- WGMMA SS: blocks=IW (IW=1 control -> expect 99,065)
for IW in 1 2 4; do
  run sgemm_tcu_wg_dxa "-DVX_CFG_EXT_TCU_ENABLE $WGSS" $IW $IW "$SZ" "bs_ss_iw${IW}_nb${IW}"
done

# --- TCU_OP: BOTH arms under --perf=1, since its published baseline was not.
for IW in 1 2 4; do
  run sgemm_tcu_op "$OPCFG" $IW 1   "$OPSZ" "bs_op_iw${IW}_nb1"
done
for IW in 1 2 4; do
  run sgemm_tcu_op "$OPCFG" $IW $IW "$OPSZ" "bs_op_iw${IW}_nb${IW}"
done

echo "[$(date +%H:%M:%S)] BLOCKS-SYM-COMPLETE" >> "$STATUS"
