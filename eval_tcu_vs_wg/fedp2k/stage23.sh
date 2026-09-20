#!/bin/bash
# Stage 2/3 verification, all fixes in tree (P0.2 tag field, P1.1 null-C latch,
# P1.2/P1.3 descriptor guards, P1.4 LMEM-residency assert).
#   A: 128^3  IW=4 kmult=4 -- reconfirm the P0.2 baseline with P1 applied
#   B: 128x128x512 IW=4 kmult=2 -- P0.3, K-chunked continuation at the DEFAULT
#      tile_K (8 chunks/tile: init on first, flush on last, neither between).
#      This is the mode the kmult=4 workaround existed to avoid.
#   C: 128x128x512 IW=4 kmult=4 -- same K, 4 chunks, cross-checks chunk count.
set -u
R=~/dev/vortex_spg/eval_tcu_vs_wg/fedp2k
S=$R/status_stage23.txt
: > $S
run(){ $R/tcuop_run.sh "$@" >> $S 2>&1; }
echo "[$(date +%H:%M:%S)] A: 128^3 kmult=4" >> $S
run 128 128 128 4 4 s23_a_128_k4
echo "[$(date +%H:%M:%S)] B: 128x128x512 kmult=2 (P0.3, 8 chunks)" >> $S
run 128 128 512 4 2 s23_b_512_k2
echo "[$(date +%H:%M:%S)] C: 128x128x512 kmult=4 (P0.3, 4 chunks)" >> $S
run 128 128 512 4 4 s23_c_512_k4
echo "[$(date +%H:%M:%S)] STAGE23-COMPLETE" >> $S
