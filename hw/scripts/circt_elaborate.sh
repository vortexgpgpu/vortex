#!/bin/bash

# Copyright © 2019-2023
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Elaborates SystemVerilog sources with CIRCT and writes the design as one
# file, the netlist every synthesis flow reads:
#
#   - memories are detected, and a RAM that asks for block RAM with its
#     synthesis attribute (ram_style = "block", ...) reads synchronously, so
#     that it maps to a block RAM or an SRAM macro (seq-retime-firmem-reads).
#     A request that cannot be honored stops the build with the reason. -m
#     decides for RAMs without a hint: off (the default) leaves them, auto
#     converts those it can. A report lists every asynchronous-read RAM;
#   - black-box modules (vendor IP, SRAM macros) stay instances of
#     themselves, with their parameter values, for the tool to bind. -x names
#     a file of their declarations (ports only), read for their interfaces;
#     -b names modules declared in the sources themselves;
#   - a conditional write of one element of an array at a dynamic index
#     becomes a select per element, as the statement means;
#   - a RAM whose read port is registered states its read-under-write
#     behavior to Vivado (rw_addr_collision), which otherwise leaves a read
#     colliding with a write undefined in block RAM; other tools ignore it;
#   - -f sv (the default) writes SystemVerilog, for Vivado and Quartus; -f
#     v2005 writes Verilog-2005 for tools with a Verilog frontend (yosys): no
#     packed arrays or structs, no local variables, no size casts; -f mlir
#     writes the HW/Comb/Seq netlist itself, for tools built on CIRCT (the
#     logic screen), and exports nothing.
#
# The sources are a filelist (-l, as gen_sources.sh writes it: +define+,
# +incdir+ and files in compile order), or include directories whose sources
# are all read (-I). CIRCT_VERILOG, CIRCT_OPT and FIRTOOL name the tools
# (hw/syn/common.mk).

CIRCT_VERILOG="${CIRCT_VERILOG:-circt-verilog}"
CIRCT_OPT="${CIRCT_OPT:-circt-opt}"
FIRTOOL="${FIRTOOL:-firtool}"

set -eo pipefail

top=""
filelist=""
format=sv
mem_mode=off
report=""
includes=()
defines=()
blackboxes=()
stubs=()
output_file=out.sv

usage() { echo "$0 usage:" && grep " .)\ #" $0; exit 0; }
[ $# -eq 0 ] && usage
while getopts "t:l:o:I:D:b:x:f:m:r:h" arg; do
    case $arg in
    t) # top module
        top=${OPTARG}
        ;;
    l) # filelist (+define+, +incdir+, files)
        filelist=${OPTARG}
        ;;
    o) # output file
        output_file=${OPTARG}
        ;;
    I) # include directory, whose sources are read too
        includes+=("${OPTARG}")
        ;;
    D) # macro definition
        defines+=("-D${OPTARG}")
        ;;
    b) # black-box modules declared in the sources, comma-separated
        IFS=',' read -r -a names <<< "${OPTARG}"
        for name in "${names[@]}"; do
            [ -n "$name" ] && blackboxes+=("--blackbox=$name")
        done
        ;;
    x) # file of black-box declarations (vendor IP)
        stubs+=("${OPTARG}")
        ;;
    f) # output format: sv, v2005 or mlir
        format=${OPTARG}
        ;;
    m) # RAMs without a hint: off or auto
        mem_mode=${OPTARG}
        ;;
    r) # memory report (default: next to the output)
        report=${OPTARG}
        ;;
    h | *)
      usage
      exit 0
      ;;
  esac
done

if [ -z "$top" ]; then
    echo "circt_elaborate.sh: error: no top module (-t)" >&2
    exit 1
fi
case "$format" in
    sv|v2005|mlir) ;;
    *) echo "circt_elaborate.sh: error: format must be sv, v2005 or mlir, not '$format'" >&2
       exit 1 ;;
esac
[ -z "$report" ] && report="${output_file%.*}.memories.rpt"

args=()
files=()
if [ -n "$filelist" ]; then
    while IFS= read -r line; do
        case "$line" in
            "") ;;
            +define+*) args+=("-D${line#+define+}") ;;
            +incdir+*) args+=("-I${line#+incdir+}") ;;
            *) files+=("$line") ;;
        esac
    done < "$filelist"
fi
for dir in "${includes[@]}"; do
    args+=("-I$dir")
    while IFS= read -r file; do
        files+=("$file")
    done < <(find "$dir" -maxdepth 1 -type f \( -name '*.v' -o -name '*.sv' \) | sort)
done
for stub in "${stubs[@]}"; do
    files+=("$stub")
    while IFS= read -r name; do
        blackboxes+=("--blackbox=$name")
    done < <(sed -nE 's/^[[:space:]]*module[[:space:]]+([A-Za-z_][A-Za-z0-9_$]*).*/\1/p' "$stub")
done

stem="${output_file%.*}"
# The netlist is the output itself in mlir format; keep the import beside it.
[ "$format" = mlir ] && stem="$stem.import"
"$CIRCT_VERILOG" --ir-hw --top="$top" --detect-memories --allow-use-before-declare \
    --allow-hierarchical-const --ignore-uninstantiated-modules "${blackboxes[@]}" \
    "${defines[@]}" "${args[@]}" "${files[@]}" -o "$stem.mlir"

# Memories are detected on import; what remains of a conditional write at a
# dynamic index (`if (c) x[i] = v`) is expanded into per-element selects.
passes=(--hw-expand-array-writes "--seq-retime-firmem-reads=mode=$mem_mode report=$report")
if [ "$format" = v2005 ]; then
    passes+=(--seq-lower-aggregate-regs --hw-aggregate-ports-to-int --hw-convert-bitcasts
             --hw-aggregate-to-comb)
    lowering=noAlwaysComb,disallowLocalVariables,disallowPackedArrays,disallowPackedStructAssignments,disallowSizeCasts,svVarDeclStyle=reg,locationInfoStyle=none,omitVersionComment
else
    lowering=locationInfoStyle=none,omitVersionComment
fi
# A module body is a graph region, where CSE cannot replace a duplicate whose
# users come first; sorting the ops first lets it merge every duplicate.
if [ "$format" = mlir ]; then
    "$CIRCT_OPT" "$stem.mlir" "${passes[@]}" --canonicalize --topological-sort --cse \
        -o "$output_file"
    exit 0
fi
"$CIRCT_OPT" "$stem.mlir" "${passes[@]}" --canonicalize --topological-sort --cse \
    -o "$stem.lowered.mlir"
"$FIRTOOL" --format=mlir "$stem.lowered.mlir" --disable-all-randomization \
    --add-vivado-ram-address-conflict-synthesis-bug-workaround \
    --lowering-options="$lowering" -o "$output_file"
