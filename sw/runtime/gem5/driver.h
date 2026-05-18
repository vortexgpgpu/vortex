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

// Direct-MMIO driver for the gem5 VortexGPGPU device.
//
// Replaces the libopae abstraction layer used by sw/runtime/opae/.
// Inside a gem5 SE-mode process, we access the device by:
//   1. Reading/writing MMIO registers via a fixed virtual address that
//      the gem5 Python config maps to the device's PIO range
//      (PIO_BASE_ADDR below; default 0x20000000 matches the legacy
//      capstone paper).
//   2. DMA staging through a fixed pinned region that the Python
//      config maps with identity virtual→physical addressing
//      (PIN_BASE_ADDR; default 0x10000000). The runtime uses host
//      virtual addresses; the gem5 DmaPort sees the same value as
//      physical because of the identity mapping.
//
// Phase 5 covers the gem5-side wiring of these mappings; Phase 4 just
// produces the runtime library.

#pragma once

#include <stddef.h>
#include <stdint.h>

namespace vortex {

// Fixed virtual addresses the runtime expects to find mapped by the
// gem5 Python config. PIN_BASE_ADDR is the runtime's heap for DMA
// staging buffers; PIO_BASE_ADDR is the device's MMIO command-and-
// status window. Sizes (PIN_REGION_SIZE / PIO_REGION_SIZE) are caps
// the runtime enforces — overruns are bugs, not malloc failures.
constexpr uintptr_t PIN_BASE_ADDR    = 0x10000000ull;
constexpr size_t    PIN_REGION_SIZE  = 0x10000000ull;  // 256 MB
constexpr uintptr_t PIO_BASE_ADDR    = 0x20000000ull;
constexpr size_t    PIO_REGION_SIZE  = 0x1000ull;      // 4 KB (1 page)

// Init / shutdown. drv_init mmaps both regions; drv_close munmaps.
// Both are idempotent in practice but should be paired 1:1.
int  drv_init();
void drv_close();

// MMIO register access. Offsets are byte offsets into the device's
// PIO range; values are written/read 64-bit at a time (the OPAE
// protocol's natural width). mmio_fence() emits the right barrier
// for HOST_ARCH (mfence on x86, dmb sy on AArch64/ARMv7) — call
// before triggering a command (B14 in proposal §2.2).
uint64_t mmio_read64 (uint64_t offset);
void     mmio_write64(uint64_t offset, uint64_t value);
void     mmio_fence();

// Staging-buffer allocation in the pinned region. Returns 0 on
// success and fills *host_ptr + *ioaddr; returns -1 on OOM in the
// pinned region. Caller owns the slot until drv_release_buffer.
//
// Under Phase 5's identity v→p mapping, *host_ptr == *ioaddr; on a
// future setup with non-identity mapping, *ioaddr is the value the
// device must DMA against and *host_ptr is what the runtime writes
// through.
int  drv_pin_buffer    (uint64_t size, void** host_ptr, uint64_t* ioaddr);
void drv_release_buffer(void* host_ptr);

} // namespace vortex
