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

# Fail loudly: abort on the first failing command and propagate failures from
# every stage of a pipeline.
set -eo pipefail

# Resolve verilator path at runtime, allowing it to be overridden by environment variable.
VERILATOR="${VERILATOR:-verilator}"

defines=()
includes=()
externs=()
source_files=()

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
    h)  echo "Usage: [-D<macro>] [-G<param>=<value>] [-T<top-module>] [-I<include-path>] [-J<external-path>] [-O<output-file>] [-C<dest-folder>: copy to] [-H<define_header>] [-P: macro preprocessing] [-h help] [files...]"
        exit 0
        ;;
    \?) echo "Invalid option: -$OPTARG" 1>&2
        exit 1
        ;;
  esac
done

# Remaining args are explicit source files
shift $((OPTIND - 1))
for arg in "$@"; do
  source_files+=( "$arg" )
done

# Optional header with `define`s
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

# Helper to copy/preprocess a single file
copy_one_file() {
  local file="$1"
  [ -f "$file" ] || return 0

  local file_ext="${file##*.}"
  local file_name
  file_name=$(basename -- "$file")

  if [ "$preprocessor" != 0 ] && { [ "$file_ext" = "v" ] || [ "$file_ext" = "sv" ]; }; then
    local dest="$copy_folder/$file_name"
    local input="$file"
    local temp_file=""
    if [[ -n "$params_str" && "$file_name" == "$top_module."* ]]; then
      temp_file=$(mktemp)
      "$SCRIPT_DIR"/repl_params.py $params_str -T"$top_module" "$file" > "$temp_file"
      input="$temp_file"
    fi
    # Trust verilator's exit status, not the output size: a missing tool,
    # unresolved include, or preprocessor error returns non-zero and is caught
    # here (this is the failure that otherwise resurfaced far downstream as
    # sv2v "Could not find top module"). An *empty but successful* output is
    # legitimate — config-gated modules (e.g. VX_decompressor.sv when RVC is
    # disabled) preprocess to nothing — so emptiness alone must not fail.
    if ! "$VERILATOR" $defines_str $includes_str -E -P "$input" > "$dest"; then
      if [ -n "$temp_file" ]; then rm -f "$temp_file"; fi
      echo "gen_sources.sh: error: verilator failed to preprocess '$file'" >&2
      echo "  using VERILATOR='$VERILATOR' — ensure it is reachable" \
           "(set VERILATOR / VERILATOR_PATH, see hw/syn/common.mk) and that" \
           "the include paths and defines resolve." >&2
      exit 1
    fi
    if [ -n "$temp_file" ]; then rm -f "$temp_file"; fi
  else
    cp "$file" "$copy_folder"
  fi
  # Return success explicitly: under `set -e` a non-zero status from the last
  # command above (e.g. a skipped conditional) would otherwise abort the run.
  return 0
}

# Optional copy phase
if [ -n "$copy_folder" ]; then
  mkdir -p "$copy_folder"

  # Files from include dirs. Use process substitution (not `find | while`)
  # so copy_one_file runs in this shell — a failure there must abort the whole
  # script, but an `exit` inside a piped `while` would only kill the subshell.
  for dir in "${includes[@]}"; do
    while IFS= read -r file; do
      copy_one_file "$file"
    done < <(find "$dir" -maxdepth 1 -type f)
  done

  # Files passed explicitly on the command line
  for file in "${source_files[@]}"; do
    copy_one_file "$file"
  done

  # Files from extern dirs (-J). When preprocessing, run them through the
  # same path as everything else so their `include directives get inlined.
  # This makes the packaged output self-contained — downstream synthesis
  # no longer needs the original third-party source tree on disk (e.g.
  # cvfpu's common_cells/*.svh headers, which IP packaging does not carry
  # along since they are reachable only via +incdir+).
  if [ "$preprocessor" != 0 ]; then
    for dir in "${externs[@]}"; do
      while IFS= read -r file; do
        copy_one_file "$file"
      done < <(find "$dir" -maxdepth 1 -type f \( -name "*.v" -o -name "*.sv" \))
    done
  fi

  # Strip .master/.slave modport qualifiers from preprocessed copies
  # (Vivado doesn't support modport qualifiers in interface port declarations)
  if [ "$preprocessor" != 0 ]; then
    for f in "$copy_folder"/*.sv; do
      [ -f "$f" ] || continue
      sed -i 's/\.\(master\|slave\)\b//g' "$f"
    done
  fi
fi

# Optional filelist generation
if [ "$output_file" != "" ]; then
    {
        # If we didn't generate a header, push +define+ into the filelist
        if [ -z "$define_header" ]; then
            for value in ${defines[@]}; do
                echo "+define+$value"
            done
        fi

        # Extern (-J) files. When preprocessing into a copy folder they
        # have already been inlined+copied there by the copy phase above,
        # and get emitted below alongside the rest of the copy-folder
        # sources — so the packaged output stays self-contained. Otherwise
        # list them in place via +incdir+ search paths.
        if ! { [ -n "$copy_folder" ] && [ "$preprocessor" != 0 ]; }; then
            # extern search paths
            for dir in ${externs[@]}; do
                echo "+incdir+$(realpath "$dir")"
            done

            # extern *_pkg.sv and .v/.sv files
            for dir in ${externs[@]}; do
                find "$(realpath $dir)" -maxdepth 1 -type f -name "*_pkg.sv" -print
            done
            for dir in ${externs[@]}; do
                find "$(realpath $dir)" -maxdepth 1 -type f \( -name "*.v" -o -name "*.sv" \) ! -name "*_pkg.sv" -print
            done
        fi

        if [ "$copy_folder" != "" ]; then
            # All files have been copied; just point to the copy folder
            echo "+incdir+$(realpath "$copy_folder")"
            find "$(realpath "$copy_folder")" -maxdepth 1 -type f -name "*_pkg.sv" -print
            find "$(realpath "$copy_folder")" -maxdepth 1 -type f -name "*_if.sv" -print
            find "$(realpath "$copy_folder")" -maxdepth 1 -type f \( -name "*.v" -o -name "*.sv" \) ! -name "*_pkg.sv" ! -name "*_if.sv" -print
        else
            # Use original include dirs
            for dir in ${includes[@]}; do
                echo "+incdir+$(realpath "$dir")"
            done

            # *_pkg.sv, then *_if.sv (interfaces), then remaining .v/.sv from include dirs
            for dir in ${includes[@]}; do
                find "$(realpath "$dir")" -maxdepth 1 -type f -name "*_pkg.sv" -print
            done
            for dir in ${includes[@]}; do
                find "$(realpath "$dir")" -maxdepth 1 -type f -name "*_if.sv" -print
            done
            for dir in ${includes[@]}; do
                find "$(realpath "$dir")" -maxdepth 1 -type f \( -name "*.v" -o -name "*.sv" \) ! -name "*_pkg.sv" ! -name "*_if.sv" -print
            done

            # add source files
            for file in "${source_files[@]}"; do
                echo "$(realpath "$file")"
            done
        fi
    } > "$output_file"
fi