#!/bin/bash
set -u
cd ~/dev/vortex_spg/build
OUT=~/dev/vortex_spg/eval_tcu_vs_wg/fix_validation2.txt
: > $OUT
run() { local APP=$1 NT=$2 EXTRA=$3 DRV=$4 TAG=$5
  local CFG="-DVX_CFG_NUM_THREADS=$NT -DVX_CFG_NUM_WARPS=16 -DVX_CFG_ISSUE_WIDTH=2 -DVX_CFG_EXT_TCU_ENABLE -DVX_CFG_TCU_WGMMA_ENABLE -DVX_CFG_EXT_DXA_ENABLE $EXTRA -DITYPE=fp16 -DOTYPE=fp32 -DWGMMA_NRC=16"
  make -C tests/regression/$APP clean > /dev/null 2>&1
  if ! CONFIGS="$CFG" make -C tests/regression/$APP > /dev/null 2>&1; then echo "$TAG: APP-BUILD-FAIL" >> $OUT; return; fi
  local R=$(CONFIGS="$CFG" ./ci/blackbox.sh --driver=$DRV --app=$APP --args="-m 128 -n 128 -k 128" 2>&1 | grep -oE 'PASSED!|Found [0-9]+ / [0-9]+ errors|Assertion failed[^,]*' | head -1)
  echo "$TAG: ${R:-FAIL}" >> $OUT
}
# the newly-supported geometry
run sgemm_tcu_wg     16 "-DVX_CFG_TCU_FEDP2K" rtlsim "wg+2k nt16 rtlsim"
run sgemm_tcu_wg_dxa 16 "-DVX_CFG_TCU_FEDP2K" rtlsim "wg_dxa+2k nt16 rtlsim"
# re-verify previously-passing shapes against the bbuf change
run sgemm_tcu_wg     8  "-DVX_CFG_TCU_FEDP2K" rtlsim "wg+2k nt8 rtlsim (recheck)"
run sgemm_tcu_wg_dxa 32 "-DVX_CFG_TCU_FEDP2K" rtlsim "wg_dxa+2k nt32 rtlsim (recheck)"
run sgemm_tcu_wg_dxa 16 ""                    rtlsim "wg_dxa-no2k nt16 rtlsim (regression)"
run sgemm_tcu_wg_sp_dxa 8 "-DVX_CFG_TCU_SPARSE_ENABLE" rtlsim "wg_sp_dxa nt8 rtlsim (sparse regression)"
echo "VALIDATION-COMPLETE" >> $OUT
