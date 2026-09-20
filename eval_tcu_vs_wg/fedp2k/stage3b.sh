#!/bin/bash
# Stage 3 validation, chained after stage23 completes.
#  B1: 128^3 with traces suppressed (VX_DBG_DEBUG_LEVEL=0). MUST reproduce run
#      A's 81087 cycles exactly -- TRACE is observation-only, so this proves the
#      suppression is behaviour-neutral and lets stage 5 run without emitting
#      ~300MB logs per point.
#  B2: 128^3 with a non-zero C. Validates P1.1: with C == 0 (the app default)
#      dropping C is indistinguishable from accumulating it, so the existing
#      test cannot detect the null-C latching defect at all.
set -u
R=~/dev/vortex_spg/eval_tcu_vs_wg/fedp2k
S=$R/status_stage3b.txt
: > $S
# wait for stage23 to release the build tree
while ! grep -q "STAGE23-COMPLETE" $R/status_stage23.txt 2>/dev/null; do sleep 10; done
echo "[$(date +%H:%M:%S)] B1: 128^3 traces off (expect cycles=81087)" >> $S
$R/tcuop_run.sh 128 128 128 4 4 s3b_noTrace_128 "-DVX_DBG_DEBUG_LEVEL=0" >> $S 2>&1
echo "[$(date +%H:%M:%S)] B2: 128^3 non-zero C (P1.1)" >> $S
$R/tcuop_run.sh 128 128 128 4 4 s3b_nonzeroC_128 "-DVX_DBG_DEBUG_LEVEL=0 -DSGEMM_NONZERO_C" >> $S 2>&1
echo "[$(date +%H:%M:%S)] STAGE3B-COMPLETE" >> $S
