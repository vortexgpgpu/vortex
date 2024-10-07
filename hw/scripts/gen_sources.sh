#!/bin/bash

# Copyright © 2019-2023
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

defines=()
includes=()
externs=()

output_file=""
define_header=""
top_module=""
copy_folder=""
preprocessor=0

defines_str=""
params_str=""
includes_str=""

# Helper function to append options
add_option() {
    if [ -n "$1" ]; then
        echo "$1 $2"
    else
        echo "$2"
    fi
}

# Parse command arguments
while getopts D:G:T:I:J:O:H:C:Ph flag
do
  case "${flag}" in
    D)  defines+=( ${OPTARG} )
        defines_str=$(add_option "$defines_str" "-D${OPTARG}")
        ;;
    G)  params_str=$(add_option "$params_str" "-G${OPTARG}")
        ;;
    T)  top_module="${OPTARG}"
        ;;
    I)  includes+=( ${OPTARG} )
        includes_str=$(add_option "$includes_str" "-I${OPTARG}")
        ;;
    J)  externs+=( ${OPTARG} )
        includes_str=$(add_option "$includes_str" "-I${OPTARG}")
        ;;
    O)  output_file="${OPTARG}"
        ;;
    H)  define_header="${OPTARG}"
        ;;
    C)  copy_folder="${OPTARG}"
        ;;
    P)  preprocessor=1
        ;;
    h)  echo "Usage: [-D<macro>] [-G<param>=<value>] [-T<top-module>] [-I<include-path>] [-J<external-path>] [-O<output-file>] [-C<dest-folder>: copy to] [-H<define_header>] [-P: macro preprocessing] [-h help]"
        exit 0
        ;;
    \?) echo "Invalid option: -$OPTARG" 1>&2
        exit 1
        ;;
  esac
done

if [ "$define_header" != "" ]; then
    directory=$(dirname "$define_header")
    mkdir -p "$directory"
    {
        # dump defines into a header file
        for value in ${defines[@]}; do
            arrNV=(${value//=/ })
            if (( ${#arrNV[@]} > 1 )); then
                echo "\`define ${arrNV[0]} ${arrNV[1]}"
            else
                echo "\`define $value"
            fi
        done
    } > "$define_header"
fi

if [ "$copy_folder" != "" ]; then
    # copy source files
    mkdir -p "$copy_folder"
    for dir in ${includes[@]}; do
        find "$dir" -maxdepth 1 -type f | while read -r file; do
            file_ext="${file##*.}"
            file_name=$(basename -- "$file")
            if [ $preprocessor != 0 ] && { [ "$file_ext" == "v" ] || [ "$file_ext" == "sv" ]; }; then
                if [[ -n "$params_str" && $file_name == "$top_module."* ]]; then
                    temp_file=$(mktemp)
                    $script_dir/repl_params.py $params_str -T$top_module "$file" > "$temp_file"
                    verilator $defines_str $includes_str -E -P "$temp_file" > "$copy_folder/$file_name"
                else
                    verilator $defines_str $includes_str -E -P "$file" > "$copy_folder/$file_name"
                fi
            else
                cp "$file" "$copy_folder"
            fi
        done
    done
fi

if [ "$output_file" != "" ]; then
    {
        if [ "$define_header" == "" ]; then
            # dump defines
            for value in ${defines[@]}; do
                echo "+define+$value"
            done
        fi

        for dir in ${externs[@]}; do
            echo "+incdir+$(realpath "$dir")"
        done

        for dir in ${externs[@]}; do
            find "$(realpath $dir)" -maxdepth 1 -type f -name "*_pkg.sv" -print
        done
        for dir in ${externs[@]}; do
            find "$(realpath $dir)" -maxdepth 1 -type f \( -name "*.v" -o -name "*.sv" \) ! -name "*_pkg.sv" -print
        done

        if [ "$copy_folder" != "" ]; then
            # dump include directories
            echo "+incdir+$(realpath "$copy_folder")"

            # dump source files
            find "$(realpath "$copy_folder")" -maxdepth 1 -type f -name "*_pkg.sv" -print
            find "$(realpath "$copy_folder")" -maxdepth 1 -type f \( -name "*.v" -o -name "*.sv" \) ! -name "*_pkg.sv" -print
        else
            # dump include directories
            for dir in ${includes[@]}; do
                echo "+incdir+$(realpath "$dir")"
            done

            # dump source files
            for dir in ${includes[@]}; do
                find "$(realpath "$dir")" -maxdepth 1 -type f -name "*_pkg.sv" -print
            done
            for dir in ${includes[@]}; do
                find "$(realpath "$dir")" -maxdepth 1 -type f \( -name "*.v" -o -name "*.sv" \) ! -name "*_pkg.sv" -print
            done
        fi
    } > "$output_file"
fi