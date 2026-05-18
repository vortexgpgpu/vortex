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

# Standalone gem5 integration test for vortex.VortexGPGPU.
#
# The SimObject loads the kernel directly via its `kernel=` parameter
# and runs it via its internal vortexTickEvent_ chain — no host CPU,
# no CP, no PIO/DMA. Smoke-tests the gem5↔libvortex-gem5.so wiring:
# dlopen succeeds, SimObject constructs, Processor::cycle() drives
# from the gem5 event loop, sim exits cleanly.
#
# The end-to-end variant ([gem5_test_vortex_app.py](gem5_test_vortex_app.py))
# wires up the host CPU + CP regfile + BAR-mapped VRAM on top.
#
# Configurable via env vars:
#   VORTEX_GEM5_LIB    — path to libvortex-gem5.so (no default)
#   VORTEX_GEM5_KERNEL — path to .vxbin to preload (no default)
#
# Run from the Vortex build dir as:
#   VORTEX_GEM5_LIB=$PWD/sim/simx/libvortex-gem5.so \
#   VORTEX_GEM5_KERNEL=$PWD/tests/kernel/hello/hello.vxbin \
#   $GEM5_HOME/build/X86/gem5.opt ci/gem5_test_vortex_hello.py

import os
import m5
from m5.objects import (
    AddrRange,
    DDR3_1600_8x8,
    MemCtrl,
    Root,
    SrcClockDomain,
    System,
    SystemXBar,
    VoltageDomain,
    VortexGPGPU,
)

LIBRARY = os.environ.get("VORTEX_GEM5_LIB")
KERNEL  = os.environ.get("VORTEX_GEM5_KERNEL")
if not LIBRARY:
    raise RuntimeError("VORTEX_GEM5_LIB env var is required")
if not KERNEL:
    raise RuntimeError("VORTEX_GEM5_KERNEL env var is required")

# Minimal system: just enough to hang the VortexGPGPU off a membus
# so gem5 considers it a properly-wired SimObject. No CPU in this
# test — the kernel runs entirely inside the SimObject's internal
# vortexTickEvent_ chain.
system = System()
system.clk_domain = SrcClockDomain(clock="1GHz",
                                   voltage_domain=VoltageDomain())
system.mem_mode = "atomic"
system.mem_ranges = [AddrRange("512MiB")]

# Membus + a small backing memory so PIO ranges have somewhere to bind.
system.membus = SystemXBar()

# Memory controller (unused at runtime in standalone mode but required
# for the system to instantiate cleanly).
system.mem_ctrl = MemCtrl()
system.mem_ctrl.dram = DDR3_1600_8x8()
system.mem_ctrl.dram.range = system.mem_ranges[0]
system.mem_ctrl.port = system.membus.mem_side_ports

# The Vortex device. It inherits clock from the system clock domain
# (set above to 1GHz) via ClockedObject; no explicit `clock=` param.
system.vortex = VortexGPGPU(
    library = LIBRARY,
    kernel  = KERNEL,
    # Explicitly disable the BAR-mapped VRAM range — the standalone
    # path loads the kernel via the device library's load_kernel()
    # entry, never via host memcpy through PIN. Leaving it enabled
    # here would conflict with this test's DRAM range.
    pin_size = 0,
)
system.vortex.pio = system.membus.mem_side_ports
system.vortex.dma = system.membus.cpu_side_ports

# Root wires the system into the simulator.
root = Root(full_system=False, system=system)
m5.instantiate()

print(f"Standalone: VortexGPGPU library={LIBRARY}")
print(f"Standalone: kernel={KERNEL}")
print("Standalone: running until VortexGPGPU exits the sim loop...")

exit_event = m5.simulate()
print(f"Standalone: exit_event.cause = {exit_event.getCause()!r}")
print(f"Standalone: tick = {m5.curTick()}")
