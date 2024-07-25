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

# this script uses sv2v and yosys tools to run.
# sv2v: https://github.com/zachjs/sv2v
# yosys: http://www.clifford.at/yosys/

# exit when any command fails
set -e

source=""
top_level=""
dir_list=()
inc_args=""
macro_args=""
no_warnings=1
process="elaborate,netlist,techmap,verilog"

declare -a excluded_warnings=("Resizing cell port")

is_excluded_warning() {
    local warning_text="$1"
    for exclusion in "${excluded_warnings[@]}"; do
        if [[ "$warning_text" == *"$exclusion"* ]]; then
            return $no_warnings
        fi
    done
    return 1
}

checkErrors()
{
    log_file="$1"
    if grep -q "Error: " "$log_file"; then
        echo "Error: found errors during synthesis!"
        exit 1
    fi

    count=0
    while IFS= read -r line; do
        if [[ "$line" == *"Warning:"* ]]; then
            warning_text="${line#Warning: }"
            if ! is_excluded_warning "$warning_text"; then
                count=$(expr $count + 1)
            fi
        fi
    done < $log_file

    if [ "$count" -ne 0 ]; then
        echo "Error: found $count unexpected warnings during synthesis!"
        exit $count
    fi
}

usage() { echo "$0 usage:" && grep " .)\ #" $0; exit 0; }
[ $# -eq 0 ] && usage
while getopts "s:t:I:D:P:Wh" arg; do
    case $arg in
    s) # source
        source=${OPTARG}
        ;;
    t) # top-level
        top_level=${OPTARG}
        ;;
    I) # include directory
        dir_list+=(${OPTARG})
        inc_args="$inc_args -I${OPTARG}"
        ;;
    D) # macro definition
        macro_args="$macro_args -D${OPTARG}"
        ;;
    P) # process
        process=${OPTARG}
        ;;
    W) # allow warnings
        no_warnings=0
        ;;
    h | *)
      usage
      exit 0
      ;;
  esac
done

{
    # read design sources
    for dir in "${dir_list[@]}"
    do
        for file in $(find $dir -maxdepth 1 -name '*.v' -o -name '*.sv' -type f)
        do
            echo "read_verilog -defer -nolatches $macro_args $inc_args -sv $file"
        done
    done
    if [ -n "$source" ]; then
        echo "read_verilog -defer -nolatches $macro_args $inc_args -sv $source"
    fi

    # elaborate
    if echo "$process" | grep -q "elaborate"; then
        echo "hierarchy -top $top_level"
    fi

    # synthesize design
    if echo "$process" | grep -q "synthesis"; then
        echo "synth -top $top_level"
    fi

    # convert to netlist
    if echo "$process" | grep -q "netlist"; then
        echo "proc; opt"
    fi

    # convert to gate logic
    if echo "$process" | grep -q "techmap"; then
        echo "techmap; opt"
    fi

    # write synthesized design
    if echo "$process" | grep -q "verilog"; then
        echo "write_verilog synth.v"
    fi

    # Generate a summary report
    echo "stat"
} > synth.ys

yosys -l yosys.log -s synth.ys

checkErrors yosys.log
