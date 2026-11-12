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

# Elaborates SystemVerilog sources with CIRCT into one HW/Comb/Seq netlist
# (MLIR), the model Beacon is built on. No synthesis tool reads it:
# Vivado, Quartus, Design Compiler and Yosys read the sources themselves.
# Asynchronous RAM reads are made synchronous before this, in the sources
# (gen_sources.sh -F); here:
#
#   - memories are detected, and a RAM that asks for block RAM with its
#     synthesis attribute (ram_style = "block", ...) reads synchronously, so
#     that it maps to a block RAM or an SRAM macro (seq-retime-firmem-reads).
#     A request that cannot be honored stops the build with the reason. -m
#     decides for RAMs without a hint: off (the default) leaves them, auto
#     converts those it can. A report lists every asynchronous-read RAM;
#   - black-box modules (vendor IP, SRAM macros) stay instances of
#     themselves, with their parameter values. -x names a file of their
#     declarations (ports only), read for their interfaces; -b names modules
#     declared in the sources themselves;
#   - a conditional write of one element of an array at a dynamic index
#     becomes a select per element, as the statement means.
#
# The sources are a filelist (-l, as gen_sources.sh writes it: +define+,
# +incdir+ and files in compile order), or include directories whose sources
# are all read (-I). CIRCT_VERILOG and CIRCT_OPT name the tools
# (hw/syn/common.mk).

CIRCT_VERILOG="${CIRCT_VERILOG:-circt-verilog}"
CIRCT_OPT="${CIRCT_OPT:-circt-opt}"

set -eo pipefail

top=""
filelist=""
mem_mode=off
report=""
includes=()
defines=()
blackboxes=()
stubs=()
output_file=out.mlir

usage() { echo "$0 usage:" && grep " .)\ #" $0; exit 0; }
[ $# -eq 0 ] && usage
while getopts "t:l:o:I:D:b:x:m:r:h" arg; do
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
    m) # RAMs without a hint: off or auto
        mem_mode=${OPTARG}
        ;;
    r) # memory report (default: next to the output)
        report=${OPTARG}
        ;;
    h)
      usage
      exit 0
      ;;
    *)
      usage
      exit 1
      ;;
  esac
done

if [ -z "$top" ]; then
    echo "circt_elaborate.sh: error: no top module (-t)" >&2
    exit 1
fi
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

# Keep the import beside the netlist, and every op's source location: the tools
# built on the netlist name logic that has no name of its own by its file and
# line.
import="${output_file%.*}.import.mlir"
"$CIRCT_VERILOG" --ir-hw --top="$top" --detect-memories --allow-use-before-declare \
    --allow-hierarchical-const --ignore-uninstantiated-modules "${blackboxes[@]}" \
    "${defines[@]}" "${args[@]}" "${files[@]}" --mlir-print-debuginfo -o "$import"

# Memories are detected on import; what remains of a conditional write at a
# dynamic index (`if (c) x[i] = v`) is expanded into per-element selects. A
# module body is a graph region, where CSE cannot replace a duplicate whose
# users come first; sorting the ops first lets it merge every duplicate.
"$CIRCT_OPT" "$import" --hw-expand-array-writes \
    "--seq-retime-firmem-reads=mode=$mem_mode report=$report" \
    --canonicalize --topological-sort --cse --mlir-print-debuginfo -o "$output_file"
