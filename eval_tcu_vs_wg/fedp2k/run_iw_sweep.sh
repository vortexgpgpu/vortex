#!/bin/bash
# IW sweep: sgemm_tcu_wg_dxa+FEDP2K (RS) vs sgemm_tcu (WMMA baseline)
# 128x128x128, NW=16, NT=32, IW in {1,2,4}, on simx and rtlsim.
set -u
BUILD=~/dev/vortex_spg/build
OUT=~/dev/vortex_spg/eval_tcu_vs_wg/fedp2k
STATUS=$OUT/status_iw_sweep.txt
RES=$OUT/iw_sweep_results.txt
cd "$BUILD"
: > "$STATUS"
: > "$RES"

COMMON="-DVX_CFG_NUM_THREADS=32 -DVX_CFG_NUM_WARPS=16 -DITYPE=fp16 -DOTYPE=fp32"

# $1=app  $2=extra cfg  $3=IW  $4=driver  $5=tag
run(){
  local APP=$1 EXTRA=$2 IW=$3 DRV=$4 TAG=$5
  local CFG="$COMMON -DVX_CFG_ISSUE_WIDTH=$IW -DVX_CFG_EXT_TCU_ENABLE $EXTRA"
  echo "[$(date +%H:%M:%S)] START $TAG" >> "$STATUS"
  make -C tests/regression/$APP clean >/dev/null 2>&1
  if ! CONFIGS="$CFG" make -C tests/regression/$APP > $OUT/${TAG}.log 2>&1; then
    echo "[$(date +%H:%M:%S)] BUILD-FAIL $TAG" >> "$STATUS"
    echo "$TAG BUILD-FAIL" >> "$RES"; return
  fi
  CONFIGS="$CFG" timeout 3000 ./ci/blackbox.sh --driver=$DRV --app=$APP \
    --args="-m 128 -n 128 -k 128" --perf=1 >> $OUT/${TAG}.log 2>&1
  local rc=$?
  local cyc ins verdict
  cyc=$(grep -aoE "cycles=[0-9]+" $OUT/${TAG}.log | tail -1 | cut -d= -f2)
  ins=$(grep -aoE "instrs=[0-9]+" $OUT/${TAG}.log | tail -1 | cut -d= -f2)
  verdict=$(grep -aoE "PASSED|FAILED" $OUT/${TAG}.log | tail -1)
  echo "$TAG rc=$rc cycles=${cyc:-NA} instrs=${ins:-NA} ${verdict:-NA}" >> "$RES"
  echo "[$(date +%H:%M:%S)] DONE $TAG rc=$rc cycles=${cyc:-NA} ${verdict:-NA}" >> "$STATUS"
}

WG="-DVX_CFG_TCU_WGMMA_ENABLE -DVX_CFG_EXT_DXA_ENABLE -DVX_CFG_TCU_FEDP2K -DWGMMA_NRC=16"

for IW in 1 2 4; do
  run sgemm_tcu_wg_dxa "$WG" $IW simx   "sw_fedp2k_iw${IW}_simx"
  run sgemm_tcu        ""    $IW simx   "sw_wmma_iw${IW}_simx"
done
for IW in 1 2 4; do
  run sgemm_tcu_wg_dxa "$WG" $IW rtlsim "sw_fedp2k_iw${IW}_rtl"
  run sgemm_tcu        ""    $IW rtlsim "sw_wmma_iw${IW}_rtl"
done

echo "IW-SWEEP-COMPLETE" >> "$STATUS"
