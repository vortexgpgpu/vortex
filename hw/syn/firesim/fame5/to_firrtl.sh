#!/bin/bash
# usage: yosys2firrtl.sh <in.v> <top> <out.fir> <keep module> [<top rename>] [<yosys cmds before hierarchy>]
# Verilog (sv2v output) -> FIRRTL, flattened except for <keep module>.
#
# Only the boundary of the unit to be threaded has to survive: FAME-5 rewrites
# that unit's insides regardless, and everything around it is one model. The
# kept module must be instantiated without parameter overrides: Yosys's FIRRTL
# writer cannot name an instance of a parameterized ($paramod) module, and
# identical instances are what threading needs in the first place.
set -euo pipefail
IN=$1 TOP=$2 OUT=$3 KEEP=$4 NEWTOP=${5:-$2} PRE=${6:-}
Y=${YOSYS:-yosys}
# Intermediate files live beside the output (kept: the Yosys log and the
# unrepaired FIRRTL are what one reads when a later step rejects the result).
W=$(dirname "$(realpath -m "$OUT")")/$(basename "$OUT" .fir).work
rm -rf "$W" && mkdir -p "$W"
REN=""
[ "$NEWTOP" != "$TOP" ] && REN="rename $TOP $NEWTOP"
$Y -q -l "$W/y.log" -p "
  read_verilog -sv $IN
  $PRE
  hierarchy -check -top $TOP
  setattr -set hdlname \"blackbox\" t:\$paramod*
  setattr -mod -set keep_hierarchy 1 $KEEP
  proc -norom; flatten; opt -nosdff -nodffe
  memory -nomap; memory_nordff; opt -full; opt -nosdff -nodffe
  dffunmap; opt_clean -purge
  splitnets -driver; opt_clean -purge
  splitnets $TOP; opt_clean -purge
  $REN
  write_firrtl $W/raw.fir" || { tail -5 "$W/y.log"; exit 1; }
# The writer types clocks as UInt<1> and converts each use with asClock();
# (Above: the non-threaded top is split to single bits -- Yosys reuses bits of a
# wire to drive other bits of the same wire, a loop at FIRRTL's word level.)
# every clock here is a port named clk, which Chisel sees as a Clock.
python3 "$(dirname "$0")/fix_firrtl.py" "$W/raw.fir" "$W/fixed0.fir"
python3 "$(dirname "$0")/gen_rams.py" "$W/fixed0.fir" "$W/fixed.fir"
sed -e 's/^\(    input clk: \)UInt<1>/\1Clock/' -e 's/asClock(clk)/clk/g' "$W/fixed.fir" > "$OUT"
echo "yosys2firrtl: $(grep -c '^  module ' "$OUT") modules, $(grep -c ' inst ' "$OUT") instances -> $OUT"
