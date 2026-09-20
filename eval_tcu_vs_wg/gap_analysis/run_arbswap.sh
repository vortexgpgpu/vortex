#!/bin/bash
set -u
cd ~/dev/vortex_spg/build
OUT=~/dev/vortex_spg/eval_tcu_vs_wg/gap_analysis
STATUS=$OUT/status_arbswap.txt
: > "$STATUS"
CFG="-DVX_CFG_NUM_THREADS=16 -DVX_CFG_NUM_WARPS=16 -DVX_CFG_ISSUE_WIDTH=4 -DVX_CFG_EXT_TCU_ENABLE -DVX_CFG_TCU_WGMMA_ENABLE -DVX_CFG_EXT_DXA_ENABLE -DITYPE=fp16 -DOTYPE=fp32 -DWGMMA_NRC=16"
make -C tests/regression/sgemm_tcu_wg_dxa clean >/dev/null 2>&1
CONFIGS="$CFG" make -C tests/regression/sgemm_tcu_wg_dxa >/dev/null 2>&1
for PC in 1 11; do
  echo "[$(date +%H:%M:%S)] START arbswap perf$PC" >> "$STATUS"
  CONFIGS="$CFG" ./ci/blackbox.sh --driver=rtlsim --app=sgemm_tcu_wg_dxa --args="-m 128 -n 128 -k 128" --perf=$PC > $OUT/arbswap_iw4_perf$PC.log 2>&1
  echo "[$(date +%H:%M:%S)] DONE arbswap perf$PC rc=$?" >> "$STATUS"
done
echo ARBSWAP-COMPLETE >> "$STATUS"
