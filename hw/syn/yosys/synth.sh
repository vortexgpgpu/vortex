#!/bin/bash

dir_list='../../rtl/libs ../../rtl/cache ../../rtl/interfaces ../../rtl'

inc_list=""
for dir in $dir_list; do
	inc_list="$inc_list -I$dir"
done

echo "inc_list=$inc_list"

{
    # read design sources
    for dir in $dir_list; do
        for file in $(find $dir -name '*.v' -o -name '*.sv' -type f) 
        do
            echo "read_verilog -sv $inc_list $file"
        done
    done

    echo "hierarchy -check -top Vortex"

    # insertation of global reset
	echo "add -global_input reset 1"
	echo "proc -global_arst reset"

    echo "synth -run coarse; opt -fine"
	echo "tee -o brams.log memory_bram -rules scripts/brams.txt;;"
    echo "write_verilog -noexpr -noattr synth.v"
} > synth.ys

yosys -l synth.log synth.ys