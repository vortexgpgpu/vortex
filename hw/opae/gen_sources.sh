#!/bin/bash

dir_list='../rtl/libs ../rtl/cache ../rtl/interfaces ../rtl'

inc_list=""
for dir in $dir_list; do
	inc_list="$inc_list -I$dir"
done

echo "inc_list=$inc_list"

{
    # read design sources
    for dir in $dir_list; do
        echo "+incdir+$dir"
        for file in $(find $dir -maxdepth 1 -name '*.v' -o -name '*.sv' -type f) 
        do
            echo $file
        done
    done
} > sources.txt