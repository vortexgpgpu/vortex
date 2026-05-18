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
# Mirrors the inheritance graph of the C++ side: DmaDevice → PioDevice
# → ClockedObject.

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

    # Optional kernel image preloaded at startup() via vortex_gem5_
    # load_kernel. When set, the device runs the kernel to completion
    # via its own tick scheduler and exits the sim loop on done — no
    # host CPU or MMIO traffic required. This is the Phase-3 entry
    # point that proves the gem5 wiring without depending on Phase-4's
    # host-runtime work. Phase 4 uploads kernels via the OPAE MMIO
    # protocol instead.
    kernel = Param.String("", "Optional .vxbin/.bin/.hex to preload at boot")

    # PIO range. Default matches the legacy capstone paper (Fig. 4)
    # for backward narrative continuity, though nothing in the design
    # depends on this exact value.
    pio_addr    = Param.Addr(0x20000000, "PIO base address")
    pio_size    = Param.Addr(0x1000, "PIO region size (bytes)")
    pio_latency = Param.Latency("1ns", "PIO access latency")
