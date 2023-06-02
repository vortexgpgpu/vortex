#!/bin/bash

defines=()
includes=()
excludes=()

output_file=""
global_file=""
dest_folder=""
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

function check_not_excluded() {
    for fe in ${excludes[@]}; do
        if [[ $1 =~ $fe ]]; then
            return 1
        fi
    done
    return 0
}

# parse command arguments
while getopts D:I:E:O:G:F:Ph flag
do
  case "${flag}" in
    D) defines+=( ${OPTARG} )
       defines_str+="-D${OPTARG} "
       ;;
    I) includes+=( ${OPTARG} )
       includes_str+="-I${OPTARG} "
       ;;
    E) excludes+=( ${OPTARG} );;
    O) output_file=( ${OPTARG} );;
    G) global_file=( ${OPTARG} );;
    F) dest_folder=( ${OPTARG} );;
    P) prepropressor=1;;
    h) echo "Usage: [-D<macro>] [-I<include>] [-E<exclude>] [-O<output_file>] [-F<destination_folder>] [-G<global_header>] [-P<prepropressor>] [-h help]"
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

if [ "$dest_folder" != "" ]; then
    # copy source files
    mkdir -p $dest_folder
    for dir in ${includes[@]}; do
        for file in $(find $dir -maxdepth 1 -name '*.v' -o -name '*.sv' -o -name '*.vh' -o -name '*.svh' -o -name '*.hex' -type f); do
            if check_not_excluded $file; then
                if [ $prepropressor != 0 ]; then
                    verilator $defines_str $includes_str -E -P $(absolute_path $file) > $dest_folder/$(basename -- $file)
                else
                    cp $(absolute_path $file) $dest_folder
                fi
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

        if [ "$dest_folder" == "" ]; then
            # dump include directories
            for dir in ${includes[@]}; do
                echo "+incdir+$dir"
            done

            # dump source files
            for dir in ${includes[@]}; do
                for file in $(find $dir -maxdepth 1 -name '*.v' -o -name '*.sv' -type f); do
                    if check_not_excluded $file; then
                        echo $(absolute_path $file)
                    fi
                done
            done
        else
            # dump destination directory
            echo "+incdir+$dest_folder"

            # dump source files
            for file in $(find $dest_folder -maxdepth 1 -name '*.v' -o -name '*.sv' -type f); do
                if check_not_excluded $file; then
                    echo $(absolute_path $file)
                fi
            done
        fi
    } > $output_file
fi
