#!/bin/bash

rtl_dir="../rtl"
exclude_list="VX_fpu_fpnew.v"
file_list=""

add_dirs()
{
    for dir in $*; do
        echo "+incdir+$dir"
        for file in $(find $dir -maxdepth 1 -name '*.v' -o -name '*.sv' -type f); do
            exclude=0
            for fe in $exclude_list; do
                if [[ $file =~ $fe ]]; then
                    exclude=1
                fi
            done
            if [[ $exclude == 0 ]]; then
                file_list="$file_list $file"
            fi
        done
    done
}

add_files()
{
    for file in $*; do
        file_list="$file_list $file"
    done
}

add_dirs $rtl_dir/fp_cores/altera/arria10
#add_dirs $rtl_dir/fp_cores/altera/stratix10

add_dirs $rtl_dir/libs $rtl_dir/interfaces $rtl_dir/fp_cores $rtl_dir/cache $rtl_dir $rtl_dir/afu

# dump file list
for file in $file_list; do
    echo $file
done