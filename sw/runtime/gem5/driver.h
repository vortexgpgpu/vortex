// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Direct-MMIO + pinned-region driver for the gem5 VortexGPGPU device.
//
// Inside a gem5 SE-mode process the device is reached by:
//
//   1. MMIO accesses to the CP regfile via a fixed virtual address that
//      the gem5 Python config maps to the SimObject's PIO range
//      (PIO_BASE_ADDR below; default 0x20000000 — gem5_v2_cp_migration
//      §3). The CP regfile is 32-bit; only 32-bit accesses are used.
//
//   2. Direct memory access to device VRAM via a fixed pinned region
//      that the gem5 Python config identity-maps virtual→physical
//      (PIN_BASE_ADDR; default 0x10000000). The runtime treats it as
//      ordinary memory: regular stores from the host process land in
//      the same physical bytes the SimObject sees as device VRAM.
//      Eliminates the need for a separate "DMA staging buffer" path —
//      gem5_v2_cp_migration §2.2.

#pragma once

#include <stddef.h>
#include <stdint.h>

namespace vortex {

// Fixed virtual addresses the runtime expects to find mapped by the
// gem5 Python config. PIN_BASE..PIN_BASE+PIN_REGION_SIZE is the
// host-visible window onto device VRAM — `memcpy(PIN_BASE+dev_addr,
// host_src, sz)` lands in the same in-process simx::RAM bytes the CP
// and Vortex see. Sized to cover the full VX_CFG_XLEN device address space
// so any address mem_alloc / mem_reserve can hand out is reachable
// via the host BAR; placed above 4 GiB so it doesn't collide with the
// simulated process's natural low-VA layout (heap/stack/code).
constexpr uintptr_t PIN_BASE_ADDR    = 0x100000000ull;
constexpr size_t    PIN_REGION_SIZE  = 0x100000000ull;  // 4 GB (= VX_CFG_XLEN=32 device VRAM)
constexpr uintptr_t PIO_BASE_ADDR    = 0x20000000ull;
constexpr size_t    PIO_REGION_SIZE  = 0x00000200ull;   // 0x200 — CP regfile

// Init / shutdown. Both are idempotent in practice but should be
// paired 1:1.
int  drv_init();
void drv_close();

// CP regfile MMIO. `offset` is the CP-internal byte offset
// (sim/common/cmd_processor.h §address map). All accesses are 32-bit
// — the CP regfile is 32-bit wide, and gem5's PIO model honors the
// packet width verbatim.
//
// mmio_fence() emits the right barrier for HOST_ARCH (mfence on x86,
// dmb sy on AArch64/ARMv7). The host runtime issues a fence between
// any non-MMIO publication (e.g. seeding a ring buffer through
// PIN_BASE_ADDR) and the doorbell write (Q_TAIL_HI) so the device
// sees the new ring entries before the tail advance.
uint32_t mmio_read32 (uint32_t offset);
void     mmio_write32(uint32_t offset, uint32_t value);
void     mmio_fence();

} // namespace vortex
