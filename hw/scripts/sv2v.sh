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

# this script uses sv2v and yosys tools to run.
# sv2v: https://github.com/zachjs/sv2v
# yosys: http://www.clifford.at/yosys/

# exit when any command fails
set -e

source=""
includes=()
macro_args=""
output_file=out.v

usage() { echo "$0 usage:" && grep " .)\ #" $0; exit 0; }
[ $# -eq 0 ] && usage
while getopts "t:s:o:I:D:h" arg; do
    case $arg in
    t) # source
        top=--top=${OPTARG}
        ;;
    s) # source
        source=${OPTARG}
        ;;
    o) # output-file
        output_file=${OPTARG}
        ;;
    I) # include directory
        includes+=(${OPTARG})
        ;;
    D) # macro definition
        macro_args="$macro_args -D${OPTARG}"
        ;;
    h | *) 
      usage
      exit 0
      ;;
  esac
done

# process include paths
inc_args=""
for dir in "${includes[@]}" 
do
	inc_args="$inc_args -I$dir"
done

# process source files
file_args=$source
for dir in "${includes[@]}" 
do
    for file in $(find $dir -maxdepth 1 -name '*.v' -o -name '*.sv' -type f) 
    do
        echo "file: $file"
        file_args="$file_args $file"
    done
done

# system-verilog to verilog conversion
echo "sv2v $top $macro_args $inc_args $file_args -v -w $output_file"
sv2v $top $macro_args $inc_args $file_args -v -w $output_file