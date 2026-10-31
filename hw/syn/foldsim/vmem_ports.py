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
    n = vmem_count(a.sim_vh)
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
    for k in range(n):
        print(f'`ifndef FOLDSIM_VMEM_{k}_CACHE_ROWS\n`define FOLDSIM_VMEM_{k}_CACHE_ROWS {a.cache_rows}\n`endif')


if __name__ == '__main__':
    main()
