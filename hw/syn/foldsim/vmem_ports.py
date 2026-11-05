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

"""What foldsim_kernel.vh says of the simulator's virtual memories.

foldsim --virtual-memories gives foldsim_sim.vh FOLDSIM_VMEM_COUNT; each
memory's foldsim_vmem controller gets its own device-memory master, numbered
after the memory ports. This prints FOLDSIM_VMEM_REPEAT(f,s): `f(port, k) for
each memory k joined by `s, spelled out as the Makefile spells out
FOLDSIM_MEM_REPEAT, and FOLDSIM_VMEM_CACHE_ROWS when given. Each memory k's
host cache takes FOLDSIM_VMEM_<k>_CACHE_ROWS where an earlier header (or the
command line) defines it, else FOLDSIM_VMEM_CACHE_ROWS (0: none); the
foldsim_afu.sv of a kernel.vh without FOLDSIM_VMEM_PER_MEMORY_ROWS takes the
global one for every memory. Nothing for a simulator without them.
With --masters, it prints the kernel's device-memory masters instead.

Grouped memories (foldsim --vmem-group: FOLDSIM_VMEM_GROUPS in
foldsim_sim.vh, each memory's FOLDSIM_VMEM_<k>_GROUP): the controllers are the
groups, so FOLDSIM_VMEM_REPEAT and the masters below count groups, `f(port, g)`;
FOLDSIM_VMEM_MEMBERS(f,s) lists every memory k of group g as member i,
`f(g, i, k)`, and FOLDSIM_VMEM_G<g>_REPEAT(f,s) the members of group g. Group
g's host cache: FOLDSIM_VMEM_G<g>_CACHE_ROWS where the compiler gave it, else
--cache-rows.

Layout v7 (FOLDSIM_VMEM_LAYOUT_V7): each controller's row bytes in device
memory, known-zero bits on or off and their block (FOLDSIM_VMEM_<k>_ROW_BYTES,
_KNOWN_ZERO, _KZ_SHIFT; groups G<g>) are the compiler's where foldsim_sim.vh
gives them, else the v6 layout's (DW/8 rounded up to a power of two of at
least 64 bytes; a bit per row).

Masters for the controllers (--vmem-masters M, Makefile FOLDSIM_VMEM_MASTERS):
by default one per controller, as many as fit beside the memory ports (at
most MAX_MASTERS in all). With fewer masters M than controllers N, controller k
shares master k mod M with the others there (foldsim_vmem_share): the header
then says FOLDSIM_VMEM_MASTERS M and FOLDSIM_VMEM_MASTER_REPEAT(f,s), `f(port,
j) for each master j, and FOLDSIM_VMEM_REPEAT gives each controller its
master's port. With M = N the header is as without the option. Where no
master is left beside the memory ports (or with M = 0), controller k shares
memory port k mod P's master with that port's endpoint: FOLDSIM_VMEM_ON_PORTS.
"""

import argparse
import re
import sys

# The kernel's device-memory masters, as the emulator binds them.
MAX_MASTERS = 32


def vmem_count(sim_vh):
    with open(sim_vh) as f:
        m = re.search(r'^`define FOLDSIM_VMEM_COUNT (\d+)\s*$', f.read(), re.M)
    return int(m.group(1)) if m else 0


def vmem_groups(sim_vh):
    """Each group's memories, in member order, from foldsim_sim.vh; None
    without groups."""
    with open(sim_vh) as f:
        text = f.read()
    m = re.search(r'^`define FOLDSIM_VMEM_GROUPS (\d+)\s*$', text, re.M)
    if not m:
        return None
    groups = [[] for _ in range(int(m.group(1)))]
    for k, g in sorted((int(k), int(g)) for k, g in
                       re.findall(r'^`define FOLDSIM_VMEM_(\d+)_GROUP (\d+)\s*$', text, re.M)):
        groups[g].append(k)
    return groups


def vmem_widths(sim_vh):
    """Each memory's and each group's DW from foldsim_sim.vh: {'<k>' or 'G<g>': DW}."""
    with open(sim_vh) as f:
        return {k: int(w) for k, w in re.findall(r'^`define FOLDSIM_VMEM_(G?\d+)_DW (\d+)\s*$', f.read(), re.M)}


def layout_defaults(name, dw):
    """The v6 layout's row bytes and a known-zero bit per row, where the
    compiler gave none."""
    nb = (dw + 7) // 8
    rb = max(64, 1 << (nb - 1).bit_length())
    return ''.join(f'`ifndef FOLDSIM_VMEM_{name}_{key}\n`define FOLDSIM_VMEM_{name}_{key} {v}\n`endif\n'
                   for key, v in (('ROW_BYTES', rb), ('KNOWN_ZERO', 1), ('KZ_SHIFT', 0))).rstrip('\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--sim-vh', required=True, help="the simulator's foldsim_sim.vh")
    ap.add_argument('--mem-ports', type=int, required=True, help='the memory ports (FOLDSIM_MEM_PORTS)')
    ap.add_argument('--cache-rows', type=int, default=0, help='host-cache rows per controller (0: none)')
    ap.add_argument('--masters', action='store_true', help='print the device-memory masters')
    ap.add_argument('--vmem-masters', type=int, default=None,
                    help="the controllers' own masters (default: one each, or as many as fit; "
                         "0: none, on the memory ports' masters)")
    a = ap.parse_args()
    groups = vmem_groups(a.sim_vh)
    n = len(groups) if groups is not None else vmem_count(a.sim_vh)
    room = MAX_MASTERS - a.mem_ports
    m = min(n, room if a.vmem_masters is None else a.vmem_masters)
    if m < 0:
        sys.exit(f'vmem_ports.py: {a.mem_ports} memory ports, beyond the kernel\'s {MAX_MASTERS} masters')
    if m > room:
        sys.exit(f'vmem_ports.py: {a.mem_ports} memory ports and {m} masters for the virtual memories '
                 f'need {a.mem_ports + m} device-memory masters; the kernel has at most {MAX_MASTERS}')
    if a.cache_rows and (a.cache_rows < 2 or a.cache_rows & (a.cache_rows - 1)):
        sys.exit('vmem_ports.py: --cache-rows is 0 or a power of two of at least 2')
    if a.masters:
        print(a.mem_ports + m)
        return
    if n == 0:
        return
    if m == 0:
        print('`define FOLDSIM_VMEM_REPEAT(f,s) '
              + ' `s '.join(f'`f({k % a.mem_ports}, {k})' for k in range(n)))
        print('`define FOLDSIM_VMEM_ON_PORTS')
    else:
        print('`define FOLDSIM_VMEM_REPEAT(f,s) '
              + ' `s '.join(f'`f({a.mem_ports + k % m}, {k})' for k in range(n)))
    if 0 < m < n:
        print(f'`define FOLDSIM_VMEM_MASTERS {m}')
        print('`define FOLDSIM_VMEM_MASTER_REPEAT(f,s) '
              + ' `s '.join(f'`f({a.mem_ports + j}, {j})' for j in range(m)))
    if a.cache_rows:
        print(f'`define FOLDSIM_VMEM_CACHE_ROWS {a.cache_rows}')
    print('`define FOLDSIM_VMEM_PER_MEMORY_ROWS')
    print('`define FOLDSIM_VMEM_LAYOUT_V7')
    widths = vmem_widths(a.sim_vh)
    if groups is not None:
        print('`define FOLDSIM_VMEM_MEMBERS(f,s) '
              + ' `s '.join(f'`f({g}, {i}, {k})' for g, ks in enumerate(groups) for i, k in enumerate(ks)))
        for g, ks in enumerate(groups):
            print(f'`define FOLDSIM_VMEM_G{g}_REPEAT(f,s) '
                  + ' `s '.join(f'`f({g}, {i}, {k})' for i, k in enumerate(ks)))
            print(f'`ifndef FOLDSIM_VMEM_G{g}_CACHE_ROWS\n`define FOLDSIM_VMEM_G{g}_CACHE_ROWS {a.cache_rows}\n`endif')
            print(layout_defaults(f'G{g}', widths[f'G{g}']))
        return
    for k in range(n):
        print(f'`ifndef FOLDSIM_VMEM_{k}_CACHE_ROWS\n`define FOLDSIM_VMEM_{k}_CACHE_ROWS {a.cache_rows}\n`endif')
        print(layout_defaults(str(k), widths[str(k)]))


if __name__ == '__main__':
    main()
