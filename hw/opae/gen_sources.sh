#!/bin/bash

dir_list='../rtl/libs ../rtl/cache ../rtl/interfaces ../rtl ../rtl/fp_cores/fpnew/src/common_cells/include ../rtl/fp_cores/fpnew/src/common_cells/src ../rtl/fp_cores/fpnew/src/fpu_div_sqrt_mvp/hdl ../rtl/fp_cores/fpnew/src'

inc_list=""
for dir in $dir_list; do
	inc_list="$inc_list -I$dir"
done

# read design sources
for dir in $dir_list; do
    echo "+incdir+$dir"
    for file in $(find $dir -maxdepth 1 -name '*.v' -o -name '*.sv' -type f) 
    do
        echo $file
    done
done