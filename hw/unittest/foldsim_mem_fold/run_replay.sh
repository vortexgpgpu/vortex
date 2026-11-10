#!/bin/bash
# run_replay.sh <out dir> <snapshot dir> <ser_trace.txt> <fold_trace.txt> [cycles]: builds tb_replay.sv
# (foldsim_mem_ser + foldsim_mem_fold of this tree, the snapshot's threaded endpoint) and replays a recorded gate run.
set -e
O=$1; S=$2; TR=$3; CF=$4; N=${5:-2000000}
T=$(cd "$(dirname "$0")/../../.." && pwd); R=${FOLDSIM_SHELL:?the Foldsim shell (share/foldsim/shell)}/rtl
mkdir -p $O
${VERILATOR:-verilator} --binary --timing -O2 -Wno-fatal -Wno-lint -Wno-style -I$R --top-module tb -Mdir $O/obj \
  $R/foldsim_mem_ser.sv ${FOLD_SV:-$R/foldsim_mem_fold.sv} $S/foldsim_mem_threaded.sv \
  $T/hw/unittest/foldsim_mem_fold/tb_replay.sv > $O/build.log 2>&1 || { tail -20 $O/build.log; exit 1; }
$O/obj/Vtb +verilator+rand+reset+0 +trace=$TR +cfg=$CF +cycles=$N > $O/run.log 2>&1
grep -E "REPLAY port" $O/run.log | awk '{print $4, $6}' | sort | uniq -c
