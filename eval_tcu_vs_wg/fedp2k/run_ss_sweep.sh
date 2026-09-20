#!/bin/bash
# SS-mode sweep: sgemm_tcu_wg_dxa+FEDP2K with A from shared memory (not registers).
# 128x128x128, NW=16, NT=32, IW in {1,2,4}, on simx and rtlsim.
set -u
BUILD=~/dev/vortex_spg/build
OUT=~/dev/vortex_spg/eval_tcu_vs_wg/fedp2k
STATUS=$OUT/status_ss_sweep.txt
RES=$OUT/ss_sweep_results.txt
cd "$BUILD"
: > "$STATUS"
: > "$RES"

COMMON="-DVX_CFG_NUM_THREADS=32 -DVX_CFG_NUM_WARPS=16 -DITYPE=fp16 -DOTYPE=fp32"
WG="-DVX_CFG_TCU_WGMMA_ENABLE -DVX_CFG_EXT_DXA_ENABLE -DVX_CFG_TCU_FEDP2K -DWGMMA_NRC=16 -DWGMMA_SS"

run(){
  local IW=$1 DRV=$2 TAG=$3
  local CFG="$COMMON -DVX_CFG_ISSUE_WIDTH=$IW -DVX_CFG_EXT_TCU_ENABLE $WG"
  echo "[$(date +%H:%M:%S)] START $TAG" >> "$STATUS"
  make -C tests/regression/sgemm_tcu_wg_dxa clean >/dev/null 2>&1
  if ! CONFIGS="$CFG" make -C tests/regression/sgemm_tcu_wg_dxa > $OUT/${TAG}.log 2>&1; then
    echo "[$(date +%H:%M:%S)] BUILD-FAIL $TAG" >> "$STATUS"
    echo "$TAG BUILD-FAIL" >> "$RES"; return
  fi
  # confirm the SS path really got selected (RS must be absent)
  local mode="SS"
  grep -q -- "-DWGMMA_SS" $OUT/${TAG}.log || mode="RS"
  CONFIGS="$CFG" timeout 3000 ./ci/blackbox.sh --driver=$DRV --app=sgemm_tcu_wg_dxa \
    --args="-m 128 -n 128 -k 128" --perf=1 >> $OUT/${TAG}.log 2>&1
  local rc=$?
  local cyc ins verdict
  cyc=$(grep -aoE "cycles=[0-9]+" $OUT/${TAG}.log | tail -1 | cut -d= -f2)
  ins=$(grep -aoE "instrs=[0-9]+" $OUT/${TAG}.log | tail -1 | cut -d= -f2)
  verdict=$(grep -aoE "PASSED|FAILED" $OUT/${TAG}.log | tail -1)
  echo "$TAG mode=$mode rc=$rc cycles=${cyc:-NA} instrs=${ins:-NA} ${verdict:-NA}" >> "$RES"
  echo "[$(date +%H:%M:%S)] DONE $TAG mode=$mode rc=$rc cycles=${cyc:-NA} ${verdict:-NA}" >> "$STATUS"
}

for IW in 1 2 4; do run $IW simx   "ss_fedp2k_iw${IW}_simx"; done
for IW in 1 2 4; do run $IW rtlsim "ss_fedp2k_iw${IW}_rtl";  done

echo "SS-SWEEP-COMPLETE" >> "$STATUS"
