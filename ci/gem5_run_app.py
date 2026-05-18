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

# End-to-end gem5 integration test for vortex.VortexGPGPU.
#
# Generic application runner — any Vortex regression test that
# follows the standard shape (host binary + kernel.vxbin in the same
# directory, links against libvortex.so) can run here.
#
# Wires (gem5_v2_cp_migration_proposal §3):
#   - SE-mode CPU(s) running an unmodified Vortex regression test
#     (same binary the SimX backend uses).
#   - VortexGPGPU device on the system membus, claiming two ranges:
#     CP regfile at PIO_BASE (32-bit MMIO) and BAR-mapped VRAM at
#     PIN_BASE (host memcpy lands in in-process simx::RAM).
#   - Identity-mapped via Process.map() — the same mechanism gem5's
#     AMD GPU integration uses at apu_se.py:1055.
#
# The simulated process loads libvortex.so (the upstream dispatcher),
# which dlopens libvortex-gem5-x86_64.so based on VORTEX_DRIVER. The
# dispatcher's CP submission path then:
#   1. mem_alloc + mem_upload → ring buffer / head / cmpl slots in VRAM
#   2. cp_mmio_write(Q_*, ...) → program CP regfile, enable Q0 + CP
#   3. vx_enqueue_launch / vx_enqueue_write / etc. → CMD_* descriptors
#      written into the ring (mem_upload), Q_TAIL_HI doorbell (cp_mmio_write),
#      Q_SEQNUM polled to wait (cp_mmio_read).
# The host runtime is a thin platform shim — no per-command logic.
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

# Number of CPU thread contexts. The upstream dispatcher spawns a
# per-Queue worker thread (commit 157e7a1) and the legacy_runtime
# helpers may spawn additional internal threads. Each thread needs a
# free HW context — we provision 4 (one main + headroom). Each is a
# separate AtomicSimpleCPU instance per the gem5 SE-mode pthread
# pattern (deprecated/example/se.py:188-189): clone() in
# syscall_emul finds the next idle context across all CPUs.
NUM_CPUS = 4

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
PIO_SIZE   = 0x0200        # CP regfile (0x40 globals + 4 × 0x40 queues + pad)
PIN_BASE   = 0x100000000   # BAR-mapped VRAM, above 4 GiB to clear the
                           # simulated process's natural low-VA layout
PIN_SIZE   = 0x100000000   # 4 GB — full XLEN=32 device address space

# ---------------------------------------------------------------------------
# System construction
# ---------------------------------------------------------------------------
system = System()
system.clk_domain = SrcClockDomain(clock="3GHz",
                                   voltage_domain=VoltageDomain())
system.mem_mode = "atomic"
system.mem_ranges = [AddrRange("1GiB")]   # advisory; actual routing
                                          # is by per-SimObject ranges
                                          # (DRAM owns [0, 1GB);
                                          # VortexGPGPU owns the PIO
                                          # and PIN ranges, both above)

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

# CPUs. Atomic — the cycle counts inside the Vortex device are
# driven by the device's own clock; timing CPU adds wall time without
# changing the kernel result. We provision NUM_CPUS instances so the
# dispatcher's per-Queue worker thread (commit 157e7a1) and any
# transient helper threads have free HW contexts to clone() into.
system.cpu = [AtomicSimpleCPU(cpu_id=i) for i in range(NUM_CPUS)]
system.multi_thread = True
for cpu in system.cpu:
    cpu.createInterruptController()
    cpu.icache_port = system.membus.cpu_side_ports
    cpu.dcache_port = system.membus.cpu_side_ports
    # X86's InterruptController has explicit pio/int_requestor/
    # int_responder ports that must be wired to the membus (per
    # learning_gem5/part1/two_level.py:111-114). ARM's interrupt model
    # doesn't expose these — skip on ARM. Tested via the DRIVER env
    # var (the same one that selects the simulated host ISA).
    if DRIVER == "gem5-x86_64":
        cpu.interrupts[0].pio           = system.membus.mem_side_ports
        cpu.interrupts[0].int_requestor = system.membus.cpu_side_ports
        cpu.interrupts[0].int_responder = system.membus.mem_side_ports

# Memory controller. DRAM serves the simulated process's normal
# low-VA address space ([0, 1 GiB) is plenty for ELF code + heap +
# stack of any in-tree regression test). The VortexGPGPU device owns
# disjoint ranges higher up:
#   - [PIO_BASE,  PIO_BASE+PIO_SIZE)  — CP regfile (32-bit MMIO)
#   - [PIN_BASE,  PIN_BASE+PIN_SIZE)  — BAR-mapped VRAM; host CPU
#     writes land in the same bytes the CP and Vortex see via in-process
#     simx::RAM (gem5_v2_cp_migration §2.2 single data plane).
# Placing PIN_BASE above 4 GiB keeps it well clear of both the DRAM
# range and the simulated process's natural VA layout.
system.mem_ctrl = MemCtrl()
system.mem_ctrl.dram = DDR3_1600_8x8()
system.mem_ctrl.dram.range = AddrRange(0, PIO_BASE)
system.mem_ctrl.port = system.membus.mem_side_ports

# The Vortex device. The `library` parameter points at the
# device-side libvortex-gem5.so (no arch suffix; gem5 itself is
# always x86-host). The host-side runtime is loaded separately by
# the simulated process via VORTEX_DRIVER below.
system.vortex = VortexGPGPU(
    library = DEV_LIB,
    kernel  = "",   # NO preload — the host binary uploads the kernel
                    # via the dispatcher's CP submission path, the way
                    # a real accelerator runtime works.
)
system.vortex.pio_addr = PIO_BASE
system.vortex.pio_size = PIO_SIZE
system.vortex.pin_addr = PIN_BASE
system.vortex.pin_size = PIN_SIZE
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
# gem5 SE-mode requires each CPU to have an assigned workload; the
# secondary CPUs are halted at boot and wake when clone() finds them
# (deprecated/example/se.py:294). Assign the same Process to all
# four CPUs — only CPU[0] starts running; the rest sit idle until
# pthread spawn.
for cpu in system.cpu:
    cpu.workload = process
    cpu.createThreads()

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
root = Root(full_system=False, system=system)
m5.instantiate()

# Identity-map both device-owned ranges into the simulated process's
# address space. Must happen AFTER m5.instantiate(). Mirrors
# apu_se.py:1055 (gem5's AMD GPU pattern). The CPU's userspace then
# touches PIO_BASE / PIN_BASE as ordinary memory; the membus routes
# both ranges to the VortexGPGPU SimObject (PIN range = BAR-mapped
# VRAM, PIO range = CP regfile).
#
# cacheable=False on PIN ensures host stores to VRAM are immediately
# visible to the CP — otherwise a cache line could hold the new ring
# entry while Q_TAIL_HI is observed by the device.
system.cpu[0].workload[0].map(PIO_BASE, PIO_BASE, PIO_SIZE, cacheable=False)
system.cpu[0].workload[0].map(PIN_BASE, PIN_BASE, PIN_SIZE, cacheable=False)

print(f"E2E: app={APP_BIN} {TEST_ARGS}")
print(f"E2E: VortexGPGPU.library={DEV_LIB}")
print(f"E2E: VORTEX_DRIVER={DRIVER}")
print(f"E2E: LD_LIBRARY_PATH={HOST_RT_DIR}")
print(f"E2E: PIO @0x{PIO_BASE:x}+0x{PIO_SIZE:x}, PIN @0x{PIN_BASE:x}+0x{PIN_SIZE:x}")
print("E2E: starting simulation...")

exit_event = m5.simulate()
print(f"E2E: exit_event.cause = {exit_event.getCause()!r}")
print(f"E2E: tick = {m5.curTick()}")
