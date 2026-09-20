#!/bin/bash
# TCU_OP vs baseline TCU evaluation: MNK=128^3, fp16, IW=4, NW=16, NT=32.
#   usage: run_eval.sh TAG
# One app at a time -- concurrent builds share the same tree and silently
# produce mixed binaries. Every app's binary and the driver library are deleted
# first, so a stale artifact can never be mistaken for a result.
set -u
TAG=${1:-ev}
BUILD=~/dev/vortex_spg/build
OUT=~/dev/vortex_spg/eval_tcu_vs_wg/fedp2k/eval
mkdir -p $OUT
RES=$OUT/${TAG}_results.txt
cd "$BUILD" || exit 1

COMMON="-DVX_CFG_NUM_THREADS=32 -DVX_CFG_NUM_WARPS=16 -DVX_CFG_ISSUE_WIDTH=4 \
-DVX_CFG_EXT_TCU_ENABLE -DVX_CFG_TCU_TYPE=TFR -DITYPE=fp16 -DOTYPE=fp32 \
-DVX_CFG_LMEM_LOG_SIZE=16 -DVX_DBG_DEBUG_LEVEL=0"

ARGS="-m 128 -n 128 -k 128"

run_one () {
  local name=$1 app=$2 extra=$3
  local log=$OUT/${TAG}_${name}.log
  echo "[$(date +%H:%M:%S)] START $name ($app)" | tee -a $OUT/${TAG}_status.txt

  # Delete the binary and the driver library before rebuilding.
  rm -f tests/regression/$app/$app tests/regression/$app/*.vxbin \
        tests/regression/$app/*.elf sw/runtime/librtlsim.so 2>/dev/null
  make -C tests/regression/$app clean >/dev/null 2>&1

  local cfg="$COMMON $extra"
  if ! CONFIGS="$cfg" make -C tests/regression/$app > $log 2>&1; then
    echo "$name BUILD-FAIL" | tee -a $RES; return
  fi
  local t0=$(date +%s)
  CONFIGS="$cfg" timeout 3600 ./ci/blackbox.sh --driver=rtlsim --app=$app \
    --args="$ARGS" --perf=1 >> $log 2>&1
  local rc=$? t1=$(date +%s)

  local cyc ins v
  cyc=$(grep -aoE "cycles=[0-9]+" $log | tail -1 | cut -d= -f2)
  ins=$(grep -aoE "instrs=[0-9]+" $log | tail -1 | cut -d= -f2)
  v=$(grep -aoE "PASSED|FAILED" $log | tail -1)
  [ $rc -eq 124 ] && v=TIMEOUT
  [ -z "$v" ] && v="ABORT(rc=$rc)"
  echo "$name cycles=${cyc:-NA} instrs=${ins:-NA} $v wall=$((t1-t0))s" | tee -a $RES
}

# A: baseline TCU (mma), B: baseline TCU + WGMMA + DXA, C: TCU_OP outer product
run_one A_sgemm_tcu        sgemm_tcu        ""
run_one B_sgemm_tcu_wg_dxa sgemm_tcu_wg_dxa ""
run_one C_sgemm_tcu_op     sgemm_tcu_op     "-DTCU_OP -DVX_CFG_EXT_DXA_ENABLE -DTCU_FEOP_BLOCK_M_OVERRIDE=2 -DTCU_FEOP_BLOCK_N_OVERRIDE=16 -DSGEMM_CONST_M=128 -DSGEMM_CONST_N=128 -DSGEMM_CONST_K=128"

echo "[$(date +%H:%M:%S)] EVAL-DONE $TAG" | tee -a $OUT/${TAG}_status.txt
