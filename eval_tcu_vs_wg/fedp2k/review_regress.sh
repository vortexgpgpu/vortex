#!/bin/bash
# Post-review TCU_OP regression: problem sizes 64^3/128^3/256^3 (single engine
# and 4-engine cooperative, IW=4; single engine IW=1), then the tensor_op CI
# suite (stage4b) and the sparse suite (stage6). rtlsim, non-debug, asserts on.
set -u
R=~/dev/vortex_spg/eval_tcu_vs_wg/fedp2k
RUN=$R/perfetto_tcuop/run_cskip.sh
PH=${1:-rev1}

sz(){ echo "-DSGEMM_CONST_M=$1 -DSGEMM_CONST_N=$1 -DSGEMM_CONST_K=$1"; }
ar(){ echo "-m $1 -n $1 -k $1 -s 0"; }
COOP="-DSGEMM_TCU_COOP -DSGEMM_TCU_ENGINES=4 -DSGEMM_NO_KERNEL_METRICS"

bash $RUN $PH \
  "single64:$(sz 64):$(ar 64)" \
  "single128:$(sz 128):$(ar 128)" \
  "single256:$(sz 256):$(ar 256)" \
  "coop64:$(sz 64) $COOP -DSGEMM_TILE_K_MULT=2:$(ar 64)" \
  "coop128:$(sz 128) $COOP -DSGEMM_TILE_K_MULT=4:$(ar 128)" \
  "coop256:$(sz 256) $COOP -DSGEMM_TILE_K_MULT=4:$(ar 256)" \
  "single128_iw1:$(sz 128) -DVX_CFG_ISSUE_WIDTH=1:$(ar 128)"

bash $R/stage4b_ci.sh
bash $R/stage6_sparse.sh
echo "[$(date +%H:%M:%S)] REVIEW-REGRESS-DONE" >> $R/perfetto_tcuop/cskip/status.txt
