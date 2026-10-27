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

# Prints how many AXI memory masters the AFU exposes for a configuration.
#
# Usage: afu_mem_ports.sh [-c <cache-dir>] <verilator> <verilator flags: defines, includes, packages>
#
# The count follows the GPU's memory ports, which only the elaborated RTL
# package knows; a platform top must nevertheless spell its ports out
# literally. Elaborating a one-line probe against the same flags as the build
# hands the build the exact value VX_afu_wrap checks it against.
#
# With -c, the result is cached under <cache-dir>, keyed on the flags and on
# the contents of every package and configuration header they name, so a
# Makefile can resolve it on every parse without re-running Verilator.

set -eo pipefail

cache_dir=""
if [ "$1" = "-c" ]; then
    cache_dir=$2
    shift 2
fi

VERILATOR=$1
shift

if [ -n "$cache_dir" ]; then
    key=$({
        printf '%s\n' "$@"
        for arg in "$@"; do
            case "$arg" in
                -I*) for h in VX_config.vh VX_types.vh vortex_afu.vh; do
                         if [ -f "${arg#-I}/$h" ]; then cat "${arg#-I}/$h"; fi
                     done ;;
                *)   if [ -f "$arg" ]; then cat "$arg"; fi ;;
            esac
        done
    } | sha1sum | cut -d' ' -f1)
    cache_file="$cache_dir/afu_mem_ports.$key"
    if [ -s "$cache_file" ]; then
        cat "$cache_file"
        exit 0
    fi
fi

tmp_dir=$(mktemp -d)
trap 'rm -rf "$tmp_dir"' EXIT

cat > "$tmp_dir/VX_afu_mem_ports_probe.sv" <<'EOF_PROBE'
`include "VX_define.vh"
`include "vortex_afu.vh"
module VX_afu_mem_ports_probe import VX_gpu_pkg::*; ();
    $info("AFU_MEM_PORTS=%0d", `AFU_MEM_PORTS_OF(VX_MEM_PORTS));
endmodule
EOF_PROBE

if ! out=$("$VERILATOR" --lint-only -Wno-fatal "$@" --top-module VX_afu_mem_ports_probe \
                        "$tmp_dir/VX_afu_mem_ports_probe.sv" 2>&1); then
    echo "$out" >&2
    echo "afu_mem_ports.sh: the probe did not elaborate" >&2
    exit 1
fi

count=$(echo "$out" | grep -o 'AFU_MEM_PORTS=[0-9][0-9]*' | head -1 | cut -d= -f2 || true)
if [ -z "$count" ]; then
    echo "$out" >&2
    echo "afu_mem_ports.sh: the probe reported no port count" >&2
    exit 1
fi

if [ -n "$cache_dir" ]; then
    mkdir -p "$cache_dir"
    echo "$count" > "$cache_file.$$"
    mv "$cache_file.$$" "$cache_file"
fi
echo "$count"
