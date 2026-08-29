#!/usr/bin/env python3

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

import json
import sys
import os
import argparse

# -----------------------------------------------------------------------------
# Robustness Helper
# -----------------------------------------------------------------------------
def get_connection_width(cell, port_candidates):
    """
    Determines the width of a port by looking at what is physically connected.
    Tries a list of potential port names (e.g., ['wdata', 'rdata']).
    """
    connections = cell.get("connections", {})

    for port in port_candidates:
        # Yosys JSON keys might be escaped (e.g., "\wdata" or "wdata")
        candidates = [port, f"\\{port}"]

        for c in candidates:
            if c in connections:
                # The connection value is a list of bits. Length = Width.
                conn_bits = connections[c]
                if len(conn_bits) > 0:
                    return len(conn_bits)
    return 0

def is_target_module(cell_type, target_modules):
    """
    Checks if the cell type matches our Blackbox RAMs.
    Handles Yosys prefixing (e.g., '$paramod$...\VX_sp_ram_asic').
    """
    # 1. Exact match
    if cell_type in target_modules:
        return True

    # 2. Suffix match
    normalized_type = cell_type.replace('\\', '/')

    for t in target_modules:
        if normalized_type.endswith(f"/{t}"):
            return True

    return False

def get_arg_list(arg_list):
    """
    Helper to flatten mixed space/comma separated args.
    Input: ['modA,modB', 'modC'] -> Output: ['modA', 'modB', 'modC']
    """
    if not arg_list: return []
    result = []
    for item in arg_list:
        # Split by comma and strip whitespace
        for sub_item in item.split(','):
            if sub_item.strip():
                result.append(sub_item.strip())
    return result

# -----------------------------------------------------------------------------
# Hierarchical Calculation
# -----------------------------------------------------------------------------
def get_module_area(mod_name, modules_dict, target_modules, w_ports, a_ports, args, memo, verbose=False):
    # Memoization to handle multiple instantiations of the same core/cluster
    if mod_name in memo:
        return memo[mod_name]

    if mod_name not in modules_dict:
        return 0.0

    total_area = 0.0
    cells = modules_dict[mod_name].get("cells", {})

    for cell_name, cell_data in cells.items():
        cell_type = cell_data.get("type")

        if is_target_module(cell_type, target_modules):
            # --- HIT: Found a Blackbox RAM Instance ---

            # 1. Infer Width (DATAW) from data ports
            w = get_connection_width(cell_data, w_ports)

            # 2. Infer Depth (SIZE) from address ports (Size = 2 ^ Addr_Bits)
            addr_bits = get_connection_width(cell_data, a_ports)
            d = 1 << addr_bits if addr_bits > 0 else 0

            # 3. Calculate
            if w > 0 and d > 0:
                inst_area = (w * d * args.bit_area) + args.overhead
                total_area += inst_area
                if verbose:
                    print(f"  [RAM MATCH] {cell_name} ({cell_type})\n              -> inferred: {w}x{d} = {inst_area:.2f}")
            else:
                if verbose:
                    print(f"  [RAM FAIL]  {cell_name} ({cell_type})\n              -> Could not infer dims (w={w}, addr_bits={addr_bits})")

        else:
            # --- MISS: It's a sub-module, recurse down ---
            total_area += get_module_area(cell_type, modules_dict, target_modules, w_ports, a_ports, args, memo, verbose)

    memo[mod_name] = total_area
    return total_area

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Robust SRAM Area Estimator via Port Inference")
    parser.add_argument("json_file")
    parser.add_argument("--top", required=True)
    parser.add_argument("-m", "--modules", nargs='+', required=True,
                        help="Blackbox Module names (comma or space separated)")
    parser.add_argument("--width-ports", nargs='+', default=["wdata", "rdata", "din", "dout"],
                        help="Data ports to infer width (comma or space separated)")
    parser.add_argument("--addr-ports", nargs='+', default=["addr", "waddr", "raddr", "address"],
                        help="Address ports to infer depth (comma or space separated)")
    parser.add_argument("--bit-area", type=float, default=0.1)
    parser.add_argument("--overhead", type=float, default=100.0)
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    if not os.path.exists(args.json_file):
        print(f"Error: File {args.json_file} not found.")
        sys.exit(1)

    with open(args.json_file, 'r') as f:
        data = json.load(f)

    modules_dict = data.get("modules", {})

    # Clean up input lists (handle commas)
    target_modules = set(get_arg_list(args.modules))
    w_ports = get_arg_list(args.width_ports)
    a_ports = get_arg_list(args.addr_ports)

    # Robust Top Module Lookup
    real_top = args.top
    if real_top not in modules_dict:
        for m in modules_dict.keys():
            if m == args.top or m.endswith(f"\\{args.top}"):
                real_top = m
                break

    if real_top not in modules_dict:
        print(f"Error: Top module '{args.top}' not found in JSON.")
        sys.exit(1)

    print("\n" + "="*80)
    print(f"SRAM AREA ESTIMATION (Connectivity Inference)")
    print(f"  Top Module  : {real_top}")
    print(f"  Target RAMs : {list(target_modules)}")
    print(f"  Scanning    : Data Ports={w_ports} | Addr Ports={a_ports}")
    print("-" * 80)

    total_area = get_module_area(real_top, modules_dict, target_modules, w_ports, a_ports, args, {}, args.verbose)

    print("-" * 80)
    print(f"TOTAL SRAM ESTIMATED AREA: {total_area:.4f} um^2")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()