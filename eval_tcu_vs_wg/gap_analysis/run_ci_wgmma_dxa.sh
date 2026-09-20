#!/bin/bash
set -u
cd ~/dev/vortex_spg/build
OUT=~/dev/vortex_spg/eval_tcu_vs_wg/gap_analysis
STATUS=$OUT/status_ci_wgmma_dxa.txt
: > "$STATUS"
MARK="(tensor_wg or dxa) and not model_parity and not perf_gate"
for X in 32 64; do
  echo "[$(date +%H:%M:%S)] START xlen$X" >> "$STATUS"
  VX_XLEN=$X python3 -m pytest ci -m "$MARK" --strict-markers -v \
    > $OUT/ci_wgmma_dxa_xlen$X.log 2>&1
  echo "[$(date +%H:%M:%S)] DONE xlen$X rc=$?" >> "$STATUS"
done
echo CI-WGMMA-DXA-COMPLETE >> "$STATUS"
