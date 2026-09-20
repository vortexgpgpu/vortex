#!/bin/bash
set -u
cd ~/dev/vortex_spg/build
OUT=~/dev/vortex_spg/eval_tcu_vs_wg/dbuf_results.txt
: > $OUT
for MODE in single dbuf; do
  EXTRA=""; [ "$MODE" = dbuf ] && EXTRA="-DDXA_DOUBLE_BUFFER"
  for SH in "8 2" "16 4"; do
    set -- $SH; NT=$1; IW=$2
    CFG="-DVX_CFG_NUM_THREADS=$NT -DVX_CFG_NUM_WARPS=16 -DVX_CFG_ISSUE_WIDTH=$IW -DVX_CFG_EXT_TCU_ENABLE -DVX_CFG_TCU_WGMMA_ENABLE -DVX_CFG_EXT_DXA_ENABLE $EXTRA -DITYPE=fp16 -DOTYPE=fp32 -DWGMMA_NRC=16"
    make -C tests/regression/sgemm_tcu_wg_dxa clean > /dev/null 2>&1
    if ! CONFIGS="$CFG" make -C tests/regression/sgemm_tcu_wg_dxa > /dev/null 2>&1; then
      echo "$MODE nt$NT iw$IW: BUILD-FAIL" >> $OUT; continue
    fi
    L=$(CONFIGS="$CFG" timeout 600 ./ci/blackbox.sh --driver=rtlsim --app=sgemm_tcu_wg_dxa --args="-m 128 -n 128 -k 128" --perf=1 2>&1)
    R=$(echo "$L" | grep -m1 -oE 'PASSED!|Found [0-9]+ / [0-9]+ errors')
    P=$(echo "$L" | grep -E '^PERF: instrs' | tail -1)
    M=$(echo "$L" | grep -E '^PERF: memory' | tail -1)
    echo "$MODE nt$NT iw$IW: ${R:-FAIL} | $P | $M" >> $OUT
  done
done
echo "MEASURE-COMPLETE" >> $OUT
