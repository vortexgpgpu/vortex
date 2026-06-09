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

# Python SimObject binding for the gem5-side VortexGPGPU device.

from m5.objects.Device import DmaDevice
from m5.params import *


class VortexGPGPU(DmaDevice):
    type = "VortexGPGPU"
    cxx_header = "dev/vortex/vortex_gpgpu_dev.hh"
    cxx_class = "gem5::VortexGPGPU"

    # Path to libvortex-gem5.so produced by `make -C sim/simx
    # USE_GEM5=1` in the Vortex build dir. Required; the C++ ctor
    # fatals if empty.
    library = Param.String("Absolute path to libvortex-gem5.so")

    # Optional kernel image preloaded at startup() via
    # vortex_gem5_load_kernel. When set, the device runs the kernel to
    # completion via its own vortexTickEvent_ scheduler and exits the
    # sim loop on done — no host CPU or MMIO traffic required. Hosted
    # mode (kernel="" or unset) starts idle; the host runtime drives
    # the CP via MMIO and the CP schedules its own ticks.
    kernel = Param.String("", "Optional .vxbin/.bin/.hex to preload at boot")

    # PIO range covers the CP regfile: 0x40 of globals + 4 × 0x40
    # per-queue slots = 0x140 used, 0x200 reserved for headroom.
    pio_addr    = Param.Addr(0x20000000, "PIO base address (CP regfile)")
    pio_size    = Param.Addr(0x0200, "PIO region size (CP regfile, bytes)")
    pio_latency = Param.Latency("1ns", "PIO access latency")

    # BAR-mapped VRAM. The device exposes its in-process RAM over the
    # same physical-address range the host's PIN_BASE_ADDR identity-maps
    # to via Process::map(). Host CPU writes land in the same bytes the
    # CP's dram_read hook sees — single source of truth for device memory.
    #
    # Disabled by default (pin_size=0); hosted (e2e) tests opt in by
    # setting both pin_addr and pin_size to match PIN_BASE_ADDR / PIN_REGION_SIZE.
    pin_addr    = Param.Addr(0x100000000, "VRAM base address (BAR-mapped)")
    pin_size    = Param.Addr(0, "VRAM region size (bytes); 0 disables")

    # Number of CP queues the PIO map can address. Matches VX_CP_NUM_QUEUES default.
    max_queues  = Param.Unsigned(4, "Number of CP queues the PIO map covers")
