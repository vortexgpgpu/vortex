#!/bin/bash
# Stage 5: TCU_OP at 256x256x256, rtlsim, plain (non-debug) builds, IW 1/2/4.
# Produces the three rows that slot into the 256^3 WGMMA-vs-WMMA table.
# TCU_OP has no simx model, so rtlsim only.
#
# VX_DBG_DEBUG_LEVEL=0 suppresses the TCU_OP trace flood (37MB at 128^3, 138MB
# at 128x128x512, so ~300MB/point here). TRACE is observation-only and stage 3b
# run B1 confirms the cycle count is identical with it off.
set -u
R=~/dev/vortex_spg/eval_tcu_vs_wg/fedp2k
S=$R/status_stage5.txt
: > $S
# wait for stage 3b to release the build tree
while ! grep -q "STAGE3B-COMPLETE" $R/status_stage3b.txt 2>/dev/null; do sleep 15; done
for IW in 1 2 4; do
  echo "[$(date +%H:%M:%S)] 256^3 IW=$IW" >> $S
  $R/tcuop_run.sh 256 256 256 $IW 4 s5_op_256_iw${IW} "-DVX_DBG_DEBUG_LEVEL=0" >> $S 2>&1
done
echo "[$(date +%H:%M:%S)] STAGE5-COMPLETE" >> $S
