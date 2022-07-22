#!/bin/bash

exclude_list="VX_fpu_fpnew.sv"
macros=()
includes=()

# parse command arguments
while getopts D:I:h flag
do
  case "${flag}" in
    D) macros+=( ${OPTARG} );;
    I) includes+=( ${OPTARG} );;
    h) echo "Usage: [-D macro] [-I include] [-h help]"
       exit 0
    ;;
  \?)
    echo "Invalid option: -$OPTARG" 1>&2
    exit 1
    ;;
  esac
done

# dump macros
for value in ${macros[@]}; do
    echo "+define+$value"
done

# dump include directories
for dir in ${includes[@]}; do
    echo "+incdir+$dir"
done

# dump source files
for dir in ${includes[@]}; do
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