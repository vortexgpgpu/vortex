#!/bin/bash
# fast_sweep.sh <first seed> <last seed> [bin]: the B4/s4x endpoint test (STAGED 0, SPEC 2, CH_SHIFT 2, ID_W 6) under
# HBM-like fast answers with every read slot held (64 reads in flight against 32 slots), R's ready delivered late,
# token delays at a fold's cadence, a long idle start and AR backpressure bursts: per seed one of eight profiles, 8 runs
# in parallel. Run from the test's build directory; a failing run's lines go to sweep/<seed>.log, the tally to
# sweep/tally.txt.
B=$(readlink -f ${3:-./foldsim_mem}); mkdir -p sweep; ulimit -f 1048576
one() {
  s=$1; case $((s % 8)) in
    0) a="bank_s4x_rd +reads=64 +hbmhi=9 +tokmin=32 +rrlag=20";;
    1) a="bank_s4x_rd +reads=64 +hbmhi=9 +rrlag=40 +tokdelay=8";;
    2) a="bank_s4x +reads=64 +hbmlo=5 +hbmhi=12 +rrlag=30 +tokdelay=4 +ready=0.9";;
    3) a="bank_full +reads=64 +hbmhi=9 +rrlag=20 +tokdelay=3";;
    4) a="bank_s4x_1b +reads=64 +hbmhi=9 +tokmin=32 +rrlag=20 +arstall=500";;
    5) a="bank_s4x_rows +reads=64 +hbmhi=3 +rrlag=10 +ready=1.0";;
    6) a="bank_s4x_rd +reads=64 +hbmlo=3 +hbmhi=20 +tokmin=30 +tokdelay=30 +rrlag=60 +arstall=2000";;
    7) a="bank_s4x +reads=64 +hbmhi=9 +tokmin=40 +tokdelay=8 +rrlag=40 +idle=40000";;
  esac
  out=$(timeout 900 $B +seed=$s $a 2>&1); rc=$?
  if [ $rc -ne 0 ] || ! echo "$out" | grep -q "^PASSED"; then
    { echo "seed $s args $a rc $rc"; echo "$out" | grep -E "FAIL|never completes" | head -20; } > sweep/$s.log
    echo "$s FAIL $a" >> sweep/tally.txt
  else echo "$s ok" >> sweep/tally.txt; fi
}
export -f one; export B
seq $1 $2 | xargs -P 8 -I{} bash -c 'one {}'
echo "done $1-$2 $(date '+%T') fails $(grep -c FAIL sweep/tally.txt)" >> sweep/tally.txt
