#!/bin/bash
# Quick TCU_OP functional subset for RTL iterations: dense single/coop,
# S1/S2 sparse, nonzero C. rtlsim, non-debug, asserts on.
set -u
R=~/dev/vortex_spg/eval_tcu_vs_wg/fedp2k
RUN=$R/perfetto_tcuop/run_cskip.sh
PH=${1:?phase}
sz(){ echo "-DSGEMM_CONST_M=$1 -DSGEMM_CONST_N=$1 -DSGEMM_CONST_K=$1"; }
ar(){ echo "-m $1 -n $1 -k $1 -s 0"; }
COOP="-DSGEMM_TCU_COOP -DSGEMM_TCU_ENGINES=4 -DSGEMM_NO_KERNEL_METRICS"

bash $RUN $PH \
  "single64:$(sz 64):$(ar 64)" \
  "single128:$(sz 128):$(ar 128)" \
  "coop128:$(sz 128) $COOP -DSGEMM_TILE_K_MULT=4:$(ar 128)" \
  "nonzeroc128:$(sz 128) -DSGEMM_NONZERO_C:$(ar 128)"

# Sparse cases through stage6's runner (its own base config).
S=$R/status_quick_sparse_$PH.txt; : > $S
eval "$(sed -n '/^sp(){/,/^}/p' $R/stage6_sparse.sh)"
sp 128 128 128 1 "-b 0.5 -p n"        q_${PH}_s1_128
sp 128 128 128 2 "-a 0.5 -b 0.5 -p u" q_${PH}_s2_128
cat $S >> $R/perfetto_tcuop/cskip/status.txt
echo "[$(date +%H:%M:%S)] QUICK-DONE $PH" >> $R/perfetto_tcuop/cskip/status.txt
