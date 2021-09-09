#!/bin/bash

# this script uses sv2v and yosys tools to run.
# sv2v: https://github.com/zachjs/sv2v
# yosys: http://www.clifford.at/yosys/

# exit when any command fails
set -e

source=""
top_level=""
dir_list=()
defines=""

usage() { echo "$0 usage:" && grep " .)\ #" $0; exit 0; }
[ $# -eq 0 ] && usage
while getopts "hs:t:I:D:" arg; do
    case $arg in
    s) # source
        source=${OPTARG}
        ;;
    t) # top-level
        top_level=${OPTARG}
        ;;
    I) # include directory
        dir_list+=(${OPTARG})
        ;;
    D) # macro definition
        defines="$defines -D${OPTARG}"
        ;;
    h | *) 
      usage
      exit 0
      ;;
  esac
done

echo "top_level=$top_level, source=$source, defines=$defines"

# process include paths
inc_list=""
for dir in "${dir_list[@]}" 
do
    echo "include: $dir" >> synth.log
	inc_list="$inc_list -I$dir"
done

# process source files
file_list=""
for dir in "${dir_list[@]}" 
do
    for file in $(find $dir -maxdepth 1 -name '*.v' -o -name '*.sv' -type f) 
    do
        echo "file: $file" >> synth.log
        file_list="$file_list $file"
    done
done

# system-verilog to verilog conversion
sv2v $defines -w output.v $inc_list $file_list

{
    echo "read_verilog -sv output.v"
    echo "hierarchy -check -top $top_level"

    # insertation of global reset
	echo "add -global_input reset 1"
	echo "proc -global_arst reset"

    echo "synth -run coarse; opt -fine"
	echo "tee -o brams.log memory_bram -rules scripts/brams.txt;;"
    echo "write_verilog -noexpr -noattr synth.v"
} > synth.ys

yosys -l yosys.log synth.ys