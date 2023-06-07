#!/bin/bash

defines=()
includes=()
externs=()

output_file=""
global_file=""
copy_folder=""
prepropressor=0

defines_str=""
includes_str=""

function absolute_path() {
    if [ -d "$1" ]; then
        (cd "$1"; pwd)
    elif [ -f "$1" ]; then
        if [[ $1 = /* ]]; then
            echo "$1"
        elif [[ $1 == */* ]]; then
            echo "$(cd "${1%/*}"; pwd)/${1##*/}"
        else
            echo "$(pwd)/$1"
        fi
    fi
}

# parse command arguments
while getopts D:I:J:O:G:C:Ph flag
do
  case "${flag}" in
    D) defines+=( ${OPTARG} )
       defines_str+="-D${OPTARG} "
       ;;
    I) includes+=( ${OPTARG} )
       includes_str+="-I${OPTARG} "
       ;;
    J) externs+=( ${OPTARG} );;
    O) output_file=( ${OPTARG} );;
    G) global_file=( ${OPTARG} );;
    C) copy_folder=( ${OPTARG} );;
    P) prepropressor=1;;
    h) echo "Usage: [-D<macro>] [-I<include-path>] [-J<extern-path>] [-O<output-file>] [-C<dest-folder>: copy to] [-G<global_header>] [-P: macro prepropressing] [-h help]"
       exit 0
    ;;
  \?)
    echo "Invalid option: -$OPTARG" 1>&2
    exit 1
    ;;
  esac
done

if [ "$global_file" != "" ]; then
    {
        # dump defines into a global header
        for value in ${defines[@]}; do
            arrNV=(${value//=/ })
            if (( ${#arrNV[@]} > 1 ));
            then
                echo "\`define ${arrNV[0]} ${arrNV[1]}"
            else
                echo "\`define $value"
            fi        
        done
    } > $global_file
fi

if [ "$copy_folder" != "" ]; then
    # copy source files
    mkdir -p $copy_folder
    for dir in ${includes[@]}; do
        for file in $(find $dir -maxdepth 1 -name '*.v' -o -name '*.sv' -o -name '*.vh' -o -name '*.svh' -o -name '*.hex' -type f); do            
            if [ $prepropressor != 0 ]; then
                verilator $defines_str $includes_str -E -P $(absolute_path $file) > $copy_folder/$(basename -- $file)
            else
                cp $(absolute_path $file) $copy_folder
            fi
        done
    done
fi

if [ "$output_file" != "" ]; then
    {
        if [ "$global_file" == "" ]; then
            # dump defines
            for value in ${defines[@]}; do
                echo "+define+$value"
            done
        fi

        if [ "$copy_folder" == "" ]; then
            # dump include directories
            for dir in ${includes[@]}; do
                echo "+incdir+$dir"
            done
            for dir in ${externs[@]}; do
                echo "+incdir+$dir"
            done

            # dump source files
            for dir in ${includes[@]}; do
                for file in $(find $dir -maxdepth 1 -name '*.v' -o -name '*.sv' -type f); do
                    echo $(absolute_path $file)
                done
            done
            for dir in ${externs[@]}; do
                for file in $(find $dir -maxdepth 1 -name '*.v' -o -name '*.sv' -type f); do
                    echo $(absolute_path $file)
                done
            done

            externs
        else
            # dump include directories
            echo "+incdir+$copy_folder"
            for dir in ${externs[@]}; do
                echo "+incdir+$dir"
            done

            # dump source files
            for file in $(find $copy_folder -maxdepth 1 -name '*.v' -o -name '*.sv' -type f); do
                echo $(absolute_path $file)
            done
            for dir in ${externs[@]}; do
                for file in $(find $dir -maxdepth 1 -name '*.v' -o -name '*.sv' -type f); do
                    echo $(absolute_path $file)
                done
            done
        fi
    } > $output_file
fi
