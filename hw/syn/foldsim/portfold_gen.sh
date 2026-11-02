#!/bin/bash
# portfold_gen.sh <out dir> <threads> [ID_W DATA_W CH_SHIFT SPEC]: the memory endpoint (foldsim_mem with its DRAM bank
# model) threaded <threads> ways with Foldsim's FAME-5 pass, for the folded memory ports (st32_1031/lk2/img4dbg/
# PORTFOLD_IFACE.md): circt-verilog (the endpoint with the given parameters) -> circt-opt (flatten) -> foldsim-opt
# foldsim-fame5{threads, thread-index, clock-enable} -> firtool. Writes <out>/foldsim_mem_threaded.sv and <out>/REV (the
# tools' revisions, this tree's commit and the commands). Run once into a frozen snapshot; the kernel build reads the
# snapshot and never runs the compiler. Tools from CIRCT_BIN (circt-verilog, circt-opt, firtool) and FOLDSIM_OPT.
set -eo pipefail
O=$1; N=$2; IDW=${3:-9}; DW=${4:-512}; CHS=${5:-5}; SPEC=${6:-2}
R=$(cd "$(dirname "$0")/../../rtl/afu/foldsim" && pwd); T=$(cd "$(dirname "$0")/../../.." && pwd)
: "${CIRCT_BIN:?}" "${FOLDSIM_OPT:?}"
mkdir -p $O/src; cd $O
for f in foldsim_token.sv foldsim_mem_bmap.sv foldsim_mem_bank.sv foldsim_mem.sv; do cp $R/$f src/; done
python3 - src/foldsim_mem.sv $IDW $DW $CHS $SPEC <<'PY'
import re, sys
p = sys.argv[1]; s = open(p).read()
for k, v in zip(('ID_W', 'DATA_W', 'CH_SHIFT', 'SPEC'), sys.argv[2:]):
    s = re.sub(r'(parameter\s+%s\s*=\s*)[^,\n)]+' % k, r'\g<1>%s' % v, s, count=1)
open(p, 'w').write(s)
PY
C1="$CIRCT_BIN/circt-verilog --ir-hw --top=foldsim_mem --allow-use-before-declare --detect-memories --max-unroll-trip-count=100000 src/foldsim_token.sv src/foldsim_mem_bmap.sv src/foldsim_mem_bank.sv src/foldsim_mem.sv -o design.mlir"
C2="$CIRCT_BIN/circt-opt design.mlir --symbol-dce --hw-flatten-modules --symbol-dce --canonicalize --cse --seq-retime-firmem-reads=report=memories.rpt --canonicalize --cse -o flat.mlir"
C3="$FOLDSIM_OPT flat.mlir '--pass-pipeline=builtin.module(foldsim-fame5{module=foldsim_mem threads=$N thread-index=true clock-enable=true})' -o threaded.mlir"
C4="$CIRCT_BIN/firtool --format=mlir threaded.mlir --add-vivado-ram-address-conflict-synthesis-bug-workaround -o foldsim_mem_threaded.sv"
eval "$C1"; [ $(grep -cE 'llhd\.(process|combinational|coroutine)|func\.func' design.mlir) = 0 ] || { echo "portfold_gen: unlowered ops in the import"; exit 1; }
eval "$C2"; eval "$C3"; eval "$C4"
{ echo "foldsim_mem threaded $N ways (ID_W $IDW DATA_W $DW CH_SHIFT $CHS SPEC $SPEC), $(date '+%F %T')"
  echo "vortex $(git -C $T rev-parse --short HEAD)$(git -C $T diff --quiet -- hw/rtl/afu/foldsim || echo +dirty)"
  echo "circt-verilog: $($CIRCT_BIN/circt-verilog --version 2>&1 | head -1)"
  echo "foldsim-opt: $FOLDSIM_OPT ($(cat $(dirname $FOLDSIM_OPT)/REV 2>/dev/null | cut -c1-80))"
  echo "firtool: $($CIRCT_BIN/firtool --version 2>&1 | head -1)"
  printf '%s\n' "$C1" "$C2" "$C3" "$C4"; } > REV
echo "portfold_gen: $O/foldsim_mem_threaded.sv"
