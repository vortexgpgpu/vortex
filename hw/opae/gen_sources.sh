#!/bin/bash

rtl_dir="../rtl"

dir_list="$rtl_dir/libs $rtl_dir/cache $rtl_dir/interfaces $rtl_dir $rtl_dir/fp_cores/fpnew/src/common_cells/include $rtl_dir/fp_cores $rtl_dir/fp_cores/altera $rtl_dir/afu"

exclude_list="VX_fpnew.v"

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