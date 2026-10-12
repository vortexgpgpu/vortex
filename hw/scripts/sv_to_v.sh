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
# Verilog-2005 file, for synthesis tools with a Verilog frontend (yosys):
# no packed arrays or structs (ports, registers and logic become plain
# vectors), no local variables, and black-box modules left as instances of
# themselves with their parameter values, for the tool to bind.
# CIRCT_VERILOG, CIRCT_OPT and FIRTOOL name the tools (hw/syn/common.mk).

CIRCT_VERILOG="${CIRCT_VERILOG:-circt-verilog}"
CIRCT_OPT="${CIRCT_OPT:-circt-opt}"
FIRTOOL="${FIRTOOL:-firtool}"

set -eo pipefail

top=""
source=""
includes=()
defines=()
blackboxes=()
output_file=out.v

usage() { echo "$0 usage:" && grep " .)\ #" $0; exit 0; }
[ $# -eq 0 ] && usage
while getopts "t:s:o:I:D:b:h" arg; do
    case $arg in
    t) # top module
        top=${OPTARG}
        ;;
    s) # source
        source=${OPTARG}
        ;;
    o) # output-file
        output_file=${OPTARG}
        ;;
    I) # include directory, whose sources are read too
        includes+=("${OPTARG}")
        ;;
    D) # macro definition
        defines+=("-D${OPTARG}")
        ;;
    b) # black-box modules, comma-separated
        IFS=',' read -r -a names <<< "${OPTARG}"
        for name in "${names[@]}"; do
            [ -n "$name" ] && blackboxes+=("--blackbox=$name")
        done
        ;;
    h | *)
      usage
      exit 0
      ;;
  esac
done

if [ -z "$top" ]; then
    echo "sv_to_v.sh: error: no top module (-t)" >&2
    exit 1
fi

inc_args=()
files=()
[ -n "$source" ] && files+=("$source")
for dir in "${includes[@]}"; do
    inc_args+=("-I$dir")
    while IFS= read -r file; do
        files+=("$file")
    done < <(find "$dir" -maxdepth 1 -type f \( -name '*.v' -o -name '*.sv' \) | sort)
done

ir="${output_file%.*}.mlir"
lowered="${output_file%.*}.lowered.mlir"
"$CIRCT_VERILOG" --ir-hw --top="$top" --detect-memories --allow-use-before-declare \
    --ignore-uninstantiated-modules "${blackboxes[@]}" "${defines[@]}" "${inc_args[@]}" \
    "${files[@]}" -o "$ir"
"$CIRCT_OPT" "$ir" --seq-lower-aggregate-regs --hw-aggregate-ports-to-int \
    --hw-convert-bitcasts --hw-aggregate-to-comb --canonicalize --cse -o "$lowered"
"$FIRTOOL" --format=mlir "$lowered" --disable-all-randomization \
    --lowering-options=noAlwaysComb,disallowLocalVariables,disallowPackedArrays,disallowPackedStructAssignments,svVarDeclStyle=reg,locationInfoStyle=none,omitVersionComment \
    -o "$output_file"
