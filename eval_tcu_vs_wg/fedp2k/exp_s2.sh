#!/bin/bash
set -u
R=~/dev/vortex_spg/eval_tcu_vs_wg/fedp2k
PH=${1:?phase}
S=$R/status_quick_sparse_$PH.txt; : > $S
eval "$(sed -n '/^sp(){/,/^}/p' $R/stage6_sparse.sh)"
sp 128 128 128 2 "-a 0.5 -b 0.5 -p u" q_${PH}_s2_128
grep RESULT $S
