#!/bin/bash
# 256x256x256 sweep: WGMMA+DXA+FEDP2K (RS and SS) vs WMMA baseline.
# NW=16, NT=32, IW in {1,2,4}, on simx then rtlsim. 18 runs total.
set -u
BUILD=~/dev/vortex_spg/build
OUT=~/dev/vortex_spg/eval_tcu_vs_wg/fedp2k
STATUS=$OUT/status_sweep256.txt
RES=$OUT/sweep256_results.txt
cd "$BUILD"
: > "$STATUS"
: > "$RES"

SZ="-m 256 -n 256 -k 256"
COMMON="-DVX_CFG_NUM_THREADS=32 -DVX_CFG_NUM_WARPS=16 -DITYPE=fp16 -DOTYPE=fp32"
WG="-DVX_CFG_TCU_WGMMA_ENABLE -DVX_CFG_EXT_DXA_ENABLE -DVX_CFG_TCU_FEDP2K -DWGMMA_NRC=16"

# $1=app $2=extra $3=IW $4=driver $5=tag $6=timeout
run(){
  local APP=$1 EXTRA=$2 IW=$3 DRV=$4 TAG=$5 TMO=$6
  local CFG="$COMMON -DVX_CFG_ISSUE_WIDTH=$IW -DVX_CFG_EXT_TCU_ENABLE $EXTRA"
  echo "[$(date +%H:%M:%S)] START $TAG" >> "$STATUS"
  make -C tests/regression/$APP clean >/dev/null 2>&1
  if ! CONFIGS="$CFG" make -C tests/regression/$APP > $OUT/${TAG}.log 2>&1; then
    echo "[$(date +%H:%M:%S)] BUILD-FAIL $TAG" >> "$STATUS"
    echo "$TAG BUILD-FAIL" >> "$RES"; return
  fi
  local mode="n/a"
  case "$EXTRA" in
    *WGMMA_SS*) mode=SS ;;
    *WGMMA_ENABLE*) mode=RS ;;
  esac
  CONFIGS="$CFG" timeout $TMO ./ci/blackbox.sh --driver=$DRV --app=$APP \
    --args="$SZ" --perf=1 >> $OUT/${TAG}.log 2>&1
  local rc=$?
  local cyc ins verdict
  cyc=$(grep -aoE "cycles=[0-9]+" $OUT/${TAG}.log | tail -1 | cut -d= -f2)
  ins=$(grep -aoE "instrs=[0-9]+" $OUT/${TAG}.log | tail -1 | cut -d= -f2)
  verdict=$(grep -aoE "PASSED|FAILED" $OUT/${TAG}.log | tail -1)
  [ $rc -eq 124 ] && verdict="TIMEOUT"
  echo "$TAG mode=$mode rc=$rc cycles=${cyc:-NA} instrs=${ins:-NA} ${verdict:-NA}" >> "$RES"
  echo "[$(date +%H:%M:%S)] DONE $TAG mode=$mode cycles=${cyc:-NA} ${verdict:-NA}" >> "$STATUS"
}

# ---- simx first (fast): full 9-row simx column ----
for IW in 1 2 4; do run sgemm_tcu_wg_dxa "$WG"              $IW simx "s256_rs_iw${IW}_simx" 3600; done
for IW in 1 2 4; do run sgemm_tcu_wg_dxa "$WG -DWGMMA_SS"   $IW simx "s256_ss_iw${IW}_simx" 3600; done
for IW in 1 2 4; do run sgemm_tcu        ""                  $IW simx "s256_wmma_iw${IW}_simx" 3600; done
echo "SIMX-DONE" >> "$STATUS"

# ---- rtlsim (slow: ~8x the 128^3 cycle counts) ----
for IW in 1 2 4; do run sgemm_tcu_wg_dxa "$WG"              $IW rtlsim "s256_rs_iw${IW}_rtl" 10800; done
for IW in 1 2 4; do run sgemm_tcu_wg_dxa "$WG -DWGMMA_SS"   $IW rtlsim "s256_ss_iw${IW}_rtl" 10800; done
for IW in 1 2 4; do run sgemm_tcu        ""                  $IW rtlsim "s256_wmma_iw${IW}_rtl" 10800; done

echo "SWEEP256-COMPLETE" >> "$STATUS"
