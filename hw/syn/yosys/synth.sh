#!/bin/bash

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

usage() { echo "$0 usage:" && grep " .)\ #" $0; exit 0; }
[ $# -eq 0 ] && usage
while getopts "s:t:I:D:h" arg; do
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
            echo "read_verilog $macro_args $inc_args -sv $file"
        done
    done
    if [ -n "$source" ]; then
        echo "read_verilog $macro_args $inc_args -sv $source"
    fi

    # generic synthesis
    echo "synth -top $top_level"

    # mapping to mycells.lib
    echo "dfflibmap -liberty mycells.lib"
    echo "abc -liberty mycells.lib"
    echo "clean"

    # write synthesized design
    echo "write_verilog synth.v"
} > synth.ys

yosys -l yosys.log synth.ys