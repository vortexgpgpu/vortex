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

// gem5 host runtime backend (pure-v2 callbacks_t).
//
// Implements vx_device with the platform primitives expected by
// sw/runtime/common/callbacks.inc: init / get_caps / mem_info /
// mem_{alloc,reserve,free,access} / upload / download / copy /
// cp_mmio_{write,read}. All kernel launches and DCR ops flow through
// the upstream dispatcher (sw/runtime/common/vx_device.cpp) which
// builds CMD_* descriptors into the CP ring buffer and bumps Q_TAIL
// via cp_mmio_write.
//
// gem5-specific shape (vs. xrt/opae):
//   - mem_upload/download/copy are direct memcpy through PIN_BASE_ADDR
//     which the gem5 SE-mode process has identity-mapped to device VRAM
//     via Process::map. No DMA descriptor; no PIO trigger.
//   - cp_mmio_{write,read} are 32-bit PIO accesses at PIO_BASE_ADDR + off
//     (no CP_BASE 0x1000 offset because the gem5 device's PIO range IS
//     the CP regfile; there is no AFU bit-12 split).
//
// See docs/proposals/gem5_v2_cp_migration_proposal.md for the full
// design rationale.

#include <common.h>
#include <util.h>          // log2floor / log2ceil / is_aligned / aligned_size
#include "driver.h"

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

using namespace vortex;

class vx_device {
public:
    vx_device()
        : global_mem_(ALLOC_BASE_ADDR,
                      GLOBAL_MEM_SIZE - ALLOC_BASE_ADDR,
                      RAM_PAGE_SIZE,
                      CACHE_BLOCK_SIZE) {}

    ~vx_device() {
        drv_close();
    }

    int init() {
        if (drv_init() != 0) {
            std::fprintf(stderr, "[VXDRV] drv_init failed\n");
            return -1;
        }
        return 0;
    }

    // Compile-time capability table — host runtime and SimX-side device
    // library share the build tree so VX_config.h macros agree on both
    // sides by construction.
    int get_caps(uint32_t caps_id, uint64_t* value) {
        switch (caps_id) {
        case VX_CAPS_VERSION:         *value = VX_ISA_IMPL_ID; break;
        case VX_CAPS_NUM_THREADS:     *value = VX_CFG_NUM_THREADS; break;
        case VX_CAPS_NUM_WARPS:       *value = VX_CFG_NUM_WARPS; break;
        case VX_CAPS_NUM_CORES:       *value = VX_CFG_NUM_CORES * VX_CFG_NUM_CLUSTERS; break;
        case VX_CAPS_NUM_CLUSTERS:    *value = VX_CFG_NUM_CLUSTERS; break;
        case VX_CAPS_SOCKET_SIZE:     *value = VX_CFG_SOCKET_SIZE; break;
        case VX_CAPS_ISSUE_WIDTH:     *value = VX_CFG_ISSUE_WIDTH; break;
        case VX_CAPS_CACHE_LINE_SIZE: *value = CACHE_BLOCK_SIZE; break;
        case VX_CAPS_GLOBAL_MEM_SIZE: *value = GLOBAL_MEM_SIZE; break;
        case VX_CAPS_LOCAL_MEM_SIZE:  *value = (1 << VX_CFG_LMEM_LOG_SIZE); break;
        case VX_CAPS_ISA_FLAGS:
            *value = ((uint64_t(VX_CFG_MISA_EXT)) << 32)
                   | ((log2floor(VX_CFG_XLEN) - 4) << 30)
                   |   VX_CFG_MISA_STD;
            break;
        case VX_CAPS_NUM_MEM_BANKS:   *value = VX_CFG_PLATFORM_MEMORY_NUM_BANKS; break;
        case VX_CAPS_MEM_BANK_SIZE:   *value = 1ull << (VX_CFG_MEM_ADDR_WIDTH / VX_CFG_PLATFORM_MEMORY_NUM_BANKS); break;
        case VX_CAPS_CLOCK_RATE:      *value = 0; break;
        case VX_CAPS_PEAK_MEM_BW:     *value = VX_CFG_PLATFORM_MEMORY_PEAK_BW; break;
        default:
            std::fprintf(stderr, "[VXDRV] invalid caps id: %u\n", caps_id);
            return -1;
        }
        return 0;
    }

    int mem_alloc(uint64_t size, int flags, uint64_t* dev_addr) {
        uint64_t addr;
        CHECK_ERR(global_mem_.allocate(size, &addr), { return err; });
        CHECK_ERR(this->mem_access(addr, size, flags), {
            global_mem_.release(addr);
            return err;
        });
        *dev_addr = addr;
        return 0;
    }

    int mem_reserve(uint64_t dev_addr, uint64_t size, int flags) {
        CHECK_ERR(global_mem_.reserve(dev_addr, size), { return err; });
        CHECK_ERR(this->mem_access(dev_addr, size, flags), {
            global_mem_.release(dev_addr);
            return err;
        });
        return 0;
    }

    int mem_free(uint64_t dev_addr) {
        return global_mem_.release(dev_addr);
    }

    int mem_access(uint64_t /*dev_addr*/, uint64_t /*size*/, int /*flags*/) {
        // Access control is enforced by the device's RAM ACL inside
        // libvortex-gem5.so. The host runtime has nothing to do here.
        return 0;
    }

    int mem_info(uint64_t* mem_free, uint64_t* mem_used) const {
        if (mem_free) *mem_free = global_mem_.free();
        if (mem_used) *mem_used = global_mem_.allocated();
        return 0;
    }

    // ---- Data plane (cold-start only) ----
    // PIN_BASE_ADDR is identity-mapped into the host process's VA via
    // Process::map (driver.h §"identity v→p"), and into the SimObject's
    // PA view of device VRAM. A memcpy through PIN_BASE_ADDR is the
    // same physical bytes the CP's DMA engine and Vortex's MemSim see —
    // zero PIO bounce, zero DMA descriptor, zero command. The dispatcher
    // uses these to seed CP ring buffers and to preload kernel ELFs;
    // ordered host↔device transfers from user code go through CMD_MEM_*
    // in the CP queue.

    int upload(uint64_t dev_addr, const void* host_ptr, uint64_t size) {
        if (size == 0) return 0;
        if (dev_addr + size > GLOBAL_MEM_SIZE) return -1;
        std::memcpy(reinterpret_cast<void*>(PIN_BASE_ADDR + dev_addr),
                    host_ptr, size);
        mmio_fence();
        return 0;
    }

    int download(void* host_ptr, uint64_t dev_addr, uint64_t size) {
        if (size == 0) return 0;
        if (dev_addr + size > GLOBAL_MEM_SIZE) return -1;
        mmio_fence();
        std::memcpy(host_ptr,
                    reinterpret_cast<const void*>(PIN_BASE_ADDR + dev_addr),
                    size);
        return 0;
    }

    int copy(uint64_t dest_addr, uint64_t src_addr, uint64_t size) {
        if (size == 0) return 0;
        if (dest_addr + size > GLOBAL_MEM_SIZE
         || src_addr  + size > GLOBAL_MEM_SIZE) return -1;
        std::memmove(reinterpret_cast<void*>(PIN_BASE_ADDR + dest_addr),
                     reinterpret_cast<const void*>(PIN_BASE_ADDR + src_addr),
                     size);
        mmio_fence();
        return 0;
    }

    // ---- Control plane (sole) ----
    // `off` is the CP-internal regfile offset (sim/common/cmd_processor.h
    // §address map). The gem5 device exposes the CP regfile starting at
    // PIO_BASE_ADDR + 0 — no AFU bit-12 split — so the wrapper is a
    // straight PIO access.
    int cp_mmio_write(uint32_t off, uint32_t value) {
        mmio_write32(off, value);
        return 0;
    }
    int cp_mmio_read(uint32_t off, uint32_t* value) {
        *value = mmio_read32(off);
        return 0;
    }

private:
    MemoryAllocator global_mem_;
};

#include <callbacks.inc>
