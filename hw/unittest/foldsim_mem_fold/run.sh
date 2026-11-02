#!/bin/bash
# run.sh <out dir> <snapshot dir> [plusargs sets...]: builds tb_fold.sv (foldsim_mem_fold, with the threaded endpoint
# of a portfold_gen.sh snapshot, against foldsim_mem_par) with Verilator and runs it once per plusargs set ("+n=200
# +mode=random" ...). RTL from this tree, or FOLD_SV=<file> for a mutant of foldsim_mem_fold.sv.
set -e
O=$1; S=$2; shift 2
T=$(cd "$(dirname "$0")/../../.." && pwd); R=$T/hw/rtl/afu/foldsim
mkdir -p $O
${VERILATOR:-verilator} --binary --timing -O2 -Wno-fatal -Wno-lint -Wno-style -I$R --top-module tb -Mdir $O/obj \
  $R/foldsim_token.sv $R/foldsim_mem_bmap.sv $R/foldsim_mem_bank.sv $R/foldsim_mem.sv $R/foldsim_mem_par.sv \
  ${FOLD_SV:-$R/foldsim_mem_fold.sv} $S/foldsim_mem_threaded.sv $T/hw/unittest/foldsim_mem_fold/tb_fold.sv \
  > $O/build.log 2>&1 || { tail -20 $O/build.log; exit 1; }
i=0
for a in "$@"; do
  i=$((i + 1))
  $O/obj/Vtb +verilator+rand+reset+2 $a > $O/run$i.log 2>&1
  echo "[$a] $(grep -E "PASS|FAIL|MISMATCH|TIMEOUT|differs" $O/run$i.log | tr "\n" " ")"
done
