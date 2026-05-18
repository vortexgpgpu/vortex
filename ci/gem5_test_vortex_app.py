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

# Phase 5 end-to-end gem5 integration test for vortex.VortexGPGPU.
#
# Generic application runner — any Vortex regression test that
# follows the standard shape (host binary + kernel.vxbin in the same
# directory, links against libvortex.so) can run here.
#
# Wires:
#   - x86 SE-mode CPU running an unmodified Vortex regression test
#     (same binary the SimX backend uses).
#   - VortexGPGPU device on the system membus at pio=0x20000000.
#   - Identity-mapped PIO range (CPU → device MMIO) and pinned region
#     (host DRAM accessed by both the CPU's userspace via virt and
#     the device's DmaPort via phys) via Process.map() — the same
#     mechanism gem5's AMD GPU integration uses at apu_se.py:1055.
#
# The simulated process loads libvortex.so (the stub), which in turn
# dlopens libvortex-gem5-x86_64.so based on the VORTEX_DRIVER env
# var. From there:
#   1. vx_dev_open → drv_init (no-op; mappings already in place)
#   2. vx_upload_kernel_bytes → DMA write of the .vxbin into VRAM
#   3. vx_copy_to_dev (×N) → DMA writes of input buffers
#   4. vx_start → MMIO CMD_RUN; kernel computes
#   5. vx_copy_from_dev → cache flush (per-core DCR_READ) + DMA read
#   6. Host verifies result, prints PASSED / FAILED
#
# Configurable via env vars:
#   VORTEX_GEM5_DEV_LIB     — path to sim/simx/libvortex-gem5.so
#                             (device-side; dlopened by the gem5 SimObject)
#   VORTEX_GEM5_HOST_RT_DIR — directory containing libvortex.so (the stub)
#                             AND libvortex-gem5-x86_64.so (the host
#                             runtime backend). Both are added to the
#                             simulated process's LD_LIBRARY_PATH.
#   VORTEX_TEST_DIR         — directory containing the test binary +
#                             kernel.vxbin
#   VORTEX_TEST_BIN         — name of the test binary inside that dir
#                             (default: vecadd)
#   VORTEX_TEST_ARGS        — args passed to the binary (default: -n16)
#   VORTEX_DRIVER           — backend selector for the stub library
#                             (default: gem5-x86_64; use gem5-aarch64
#                             when running the ARM matrix)

import os
import shlex

import m5
from m5.objects import (
    AddrRange,
    DDR3_1600_8x8,
    MemCtrl,
    Process,
    RedirectPath,
    Root,
    SEWorkload,
    SrcClockDomain,
    System,
    SystemXBar,
    AtomicSimpleCPU,
    VoltageDomain,
    VortexGPGPU,
)

DEV_LIB     = os.environ.get("VORTEX_GEM5_DEV_LIB")
HOST_RT_DIR = os.environ.get("VORTEX_GEM5_HOST_RT_DIR")
TEST_DIR    = os.environ.get("VORTEX_TEST_DIR")
TEST_BIN    = os.environ.get("VORTEX_TEST_BIN", "vecadd")
TEST_ARGS   = os.environ.get("VORTEX_TEST_ARGS", "-n16")
DRIVER      = os.environ.get("VORTEX_DRIVER",   "gem5-x86_64")

for name, val in [
    ("VORTEX_GEM5_DEV_LIB",     DEV_LIB),
    ("VORTEX_GEM5_HOST_RT_DIR", HOST_RT_DIR),
    ("VORTEX_TEST_DIR",         TEST_DIR),
]:
    if not val:
        raise RuntimeError(f"{name} env var is required")

APP_BIN = f"{TEST_DIR}/{TEST_BIN}"

# Fixed mappings used by the gem5 host runtime (see
# sw/runtime/gem5/driver.h). The Python config and the C runtime
# share these constants by convention; if you change one, change
# both.
PIO_BASE   = 0x20000000
PIO_SIZE   = 0x1000        # 4 KB — one page is enough for the OPAE regs
PIN_BASE   = 0x10000000
PIN_SIZE   = 0x10000000    # 256 MB — large enough for vecadd staging

# ---------------------------------------------------------------------------
# System construction
# ---------------------------------------------------------------------------
system = System()
system.clk_domain = SrcClockDomain(clock="3GHz",
                                   voltage_domain=VoltageDomain())
system.mem_mode = "atomic"
system.mem_ranges = [AddrRange("1GiB")]   # covers both DRAM and the
                                          # PIN_BASE identity-mapped region
                                          # (PIN_BASE=0x10000000 < 1GB)

# Cross-arch interp + runtime library redirection.
# Two separate gem5 mechanisms are at play:
#   (1) `setInterpDir(prefix)` prepends `prefix` to PT_INTERP when
#       gem5 loads the dynamic linker (e.g. /lib/ld-linux-aarch64.so.1
#       → /usr/aarch64-linux-gnu/lib/ld-linux-aarch64.so.1). The
#       linker is opened directly by gem5's loader, NOT via SE-mode
#       syscall, so RedirectPath doesn't help here.
#   (2) `system.redirect_paths` redirects open()/stat()/etc syscalls
#       the GUEST process makes — used when the dynamic linker
#       later looks up libc.so.6, libstdc++.so.6, libvortex.so, etc.
# Both are no-ops for native x86.
if DRIVER == "gem5-aarch64":
    from m5.core import setInterpDir
    setInterpDir("/usr/aarch64-linux-gnu")
    system.redirect_paths = [
        RedirectPath(app_path="/lib/aarch64-linux-gnu",
                     host_paths=["/usr/aarch64-linux-gnu/lib"]),
        RedirectPath(app_path="/usr/lib/aarch64-linux-gnu",
                     host_paths=["/usr/aarch64-linux-gnu/lib"]),
    ]

# Membus connects CPU ↔ memory ↔ VortexGPGPU.
system.membus = SystemXBar()
system.system_port = system.membus.cpu_side_ports

# CPU. Atomic for now — the cycle counts inside the Vortex device are
# driven by the device's own clock anyway; timing CPU adds gem5 wall
# time without changing the kernel result.
system.cpu = AtomicSimpleCPU()
system.cpu.createInterruptController()
system.cpu.icache_port = system.membus.cpu_side_ports
system.cpu.dcache_port = system.membus.cpu_side_ports
# X86's InterruptController has explicit pio/int_requestor/int_responder
# ports that must be wired to the membus (per
# learning_gem5/part1/two_level.py:111-114). ARM's interrupt model
# doesn't expose these — skip the wiring on ARM. Tested via the
# DRIVER env var (the same one that selects the simulated host ISA).
if DRIVER == "gem5-x86_64":
    system.cpu.interrupts[0].pio           = system.membus.mem_side_ports
    system.cpu.interrupts[0].int_requestor = system.membus.cpu_side_ports
    system.cpu.interrupts[0].int_responder = system.membus.mem_side_ports

# Memory controller. The DRAM range starts at 0; PIO_BASE=0x20000000
# lives ABOVE the 1 GB range (since 0x20000000 = 512 MB) — wait, it's
# inside. mem_ranges above is just a hint; the actual MemCtrl range
# is what determines what's routed where.
system.mem_ctrl = MemCtrl()
system.mem_ctrl.dram = DDR3_1600_8x8()
# DRAM serves [0, 512MB). PIO at 0x20000000 (=512MB) sits at the top
# edge, so let DRAM serve [0, 512MB) and let the membus route
# 0x20000000+ to the VortexGPGPU.
system.mem_ctrl.dram.range = AddrRange(0, size="512MiB")
system.mem_ctrl.port = system.membus.mem_side_ports

# The Vortex device. The `library` parameter points at the
# device-side libvortex-gem5.so (no arch suffix; gem5 itself is
# always x86-host). The host-side runtime is loaded separately by
# the simulated process via VORTEX_DRIVER below.
system.vortex = VortexGPGPU(
    library = DEV_LIB,
    kernel  = "",   # NO preload — the host binary uploads the kernel
                    # via the OPAE MMIO protocol, the way a real
                    # accelerator runtime works.
)
system.vortex.pio_addr = PIO_BASE
system.vortex.pio_size = PIO_SIZE
system.vortex.pio = system.membus.mem_side_ports
system.vortex.dma = system.membus.cpu_side_ports

# ---------------------------------------------------------------------------
# Workload (the host test binary)
# ---------------------------------------------------------------------------
argv = [APP_BIN] + shlex.split(TEST_ARGS)
process = Process(
    pid=100,
    cwd=TEST_DIR,
    cmd=argv,
    executable=argv[0],
    env=[
        # Tells the stub to dlopen our backend
        # (libvortex.so does dlopen("libvortex-${VORTEX_DRIVER}.so")).
        f"VORTEX_DRIVER={DRIVER}",
        # Library search path inside the simulated process. Must
        # contain libvortex.so AND libvortex-gem5-$ARCH.so (both
        # are in HOST_RT_DIR by construction).
        f"LD_LIBRARY_PATH={HOST_RT_DIR}",
    ],
)

system.workload = SEWorkload.init_compatible(APP_BIN)
system.cpu.workload = process
system.cpu.createThreads()

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
root = Root(full_system=False, system=system)
m5.instantiate()

# Identity-map the device PIO range and the pinned DMA region into
# the simulated process's address space. Must happen AFTER
# m5.instantiate() — the process needs a backing C++ object before
# map() is callable. Mirrors apu_se.py:1055 (gem5's AMD GPU pattern).
# The CPU's userspace then touches PIO_BASE / PIN_BASE as ordinary
# memory; the membus routes PIO_BASE → device, PIN_BASE → DRAM.
system.cpu.workload[0].map(PIO_BASE, PIO_BASE, PIO_SIZE, cacheable=False)
system.cpu.workload[0].map(PIN_BASE, PIN_BASE, PIN_SIZE, cacheable=True)

print(f"Phase 5: app={APP_BIN} {TEST_ARGS}")
print(f"Phase 5: VortexGPGPU.library={DEV_LIB}")
print(f"Phase 5: VORTEX_DRIVER={DRIVER}")
print(f"Phase 5: LD_LIBRARY_PATH={HOST_RT_DIR}")
print(f"Phase 5: PIO @0x{PIO_BASE:x}+0x{PIO_SIZE:x}, PIN @0x{PIN_BASE:x}+0x{PIN_SIZE:x}")
print("Phase 5: starting simulation...")

exit_event = m5.simulate()
print(f"Phase 5: exit_event.cause = {exit_event.getCause()!r}")
print(f"Phase 5: tick = {m5.curTick()}")
