#!/bin/bash

dir_list='../rtl/libs ../rtl/cache ../rtl/interfaces ../rtl ../rtl/fp_cores/fpnew/src/common_cells/include ../rtl/fp_cores ../rtl/fp_cores/altera'
exclude_list='VX_fpnew.v' 

# read design sources
for dir in $dir_list; do
    echo "+incdir+$dir"
    for file in $(find $dir -maxdepth 1 -name '*.v' -o -name '*.sv' -type f); do
        exclude=0
        for fe in $exclude_list; do
            if [[ $file =~ $fe ]]; then
                exclude=1
            fi
        done
        if [[ $exclude == 0 ]]; then
            echo $file
        fi
    done
done