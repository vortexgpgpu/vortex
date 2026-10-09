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
SPLITKEPT="splitnets -ports -driver A:keep_hierarchy; opt_clean -purge"
[ "${FAME5_SPLIT_KEPT_PORTS:-1}" = 0 ] && SPLITKEPT=""
# Kept modules' ports are split at driver boundaries, as the top's are: a unit
# that computes some bits of a wide output from other bits of it (an LSU
# request built from its own address field) would otherwise read its own
# output port, which FIRRTL's word-level loop check rejects.
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
  $SPLITKEPT
  splitnets $TOP; opt_clean -purge
  $REN
  write_rtlil $W/design.il" || { tail -5 "$W/y.log"; exit 1; }
python3 "$(dirname "$0")/name_kept.py" "$W/design.il"
# FIRRTL memories carry one enable and one uniform mask per write port, so a
# handful of RTL arrays - read-modify-write DCR files, shift queues, per-field
# struct writes - collect into \$mem cells write_firrtl rejects ('Complex
# write enable'). Everything it can express stays a memory (threading turns
# those into dense deep LUTRAM); only the rejected ones are memory_map'ed to
# the FF logic every other flow infers for them anyway. The writer names each
# offender, so export simply iterates until it goes through.
YOSYS=$Y python3 "$(dirname "$0")/export_firrtl.py" "$W/design.il" "$W/raw0.fir" "$W/export.log" || { tail -5 "$W/export.log"; exit 1; }
# Flattened + bit-split identifiers are most of the file's bytes; a full-GPU
# export otherwise exceeds the FIRRTL reader's 2 GiB input limit.
python3 "$(dirname "$0")/shorten_fir.py" "$W/raw0.fir" "$W/raw.fir"
rm -f "$W/raw0.fir"
# The writer types clocks as UInt<1> and converts each use with asClock();
# (Above: the non-threaded top is split to single bits -- Yosys reuses bits of a
# wire to drive other bits of the same wire, a loop at FIRRTL's word level.)
# every clock here is a port named clk, which Chisel sees as a Clock.
python3 "$(dirname "$0")/fix_firrtl.py" "$W/raw.fir" "$W/fixed0.fir"
python3 "$(dirname "$0")/gen_rams.py" "$W/fixed0.fir" "$W/fixed.fir"
sed -e 's/^\(    input clk: \)UInt<1>/\1Clock/' -e 's/asClock(clk)/clk/g' "$W/fixed.fir" > "$OUT"
echo "yosys2firrtl: $(grep -c '^  module ' "$OUT") modules, $(grep -c ' inst ' "$OUT") instances -> $OUT"
