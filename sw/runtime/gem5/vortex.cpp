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

// ============================================================================
// gem5 host runtime backend — a pure transport HAL (see callbacks.h):
//   * device lifecycle      — init() / ~vx_device()
//   * CP register channel   — cp_reg_read / cp_reg_write  (32-bit PIO)
//   * CP-visible host memory — host_mem_alloc / host_mem_free
//
// Device-memory allocation and caps decoding live in the common core; the
// Command Processor is the sole memory engine.
//
// gem5 is architecturally unique: the CP runs natively inside the device
// SimObject, while the host runtime runs as simulated-CPU code — separate
// domains. The ONLY memory both can reach is device VRAM: the CP addresses
// it directly, and the host reaches it through the PIN_BASE_ADDR window.
// So CP-visible host memory must be VRAM.
//
// host_mem_alloc carves the command ring + DMA staging from a dedicated
// aperture at the top of the PIN window — host_ptr = PIN_BASE_ADDR + addr,
// cp_addr = addr. The common-core device allocator grows bottom-up from
// ALLOC_BASE_ADDR; this aperture sits above that region.
//
// cp_reg_{write,read} are 32-bit PIO accesses at PIO_BASE_ADDR + off — no
// CP_BASE 0x1000 offset (the gem5 device's PIO range IS the CP regfile).
// ============================================================================

#include <common.h>
#include "driver.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>

using namespace vortex;

// CP-visible host-memory aperture: the top of the PIN window onto VRAM.
static constexpr uint64_t GEM5_HOST_APERTURE = 64ull << 20;   // 64 MB
static constexpr uint64_t GEM5_HOST_BASE     = PIN_REGION_SIZE - GEM5_HOST_APERTURE;

class vx_device {
public:
    vx_device()
        : host_mem_(GEM5_HOST_BASE, GEM5_HOST_APERTURE,
                    RAM_PAGE_SIZE, CACHE_BLOCK_SIZE) {}

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

    // ----- CP register channel -----
    // `off` is the CP-internal regfile offset. The gem5 device exposes the
    // CP regfile at PIO_BASE_ADDR + 0 (no AFU bit-12 split). The fences
    // order ring/staging publication (host stores through PIN_BASE_ADDR)
    // against the doorbell write and the completion read.
    int cp_reg_write(uint32_t off, uint32_t value) {
        mmio_fence();             // ring writes visible before the doorbell
        mmio_write32(off, value);
        return 0;
    }
    int cp_reg_read(uint32_t off, uint32_t* value) {
        *value = mmio_read32(off);
        mmio_fence();             // device writes visible before host reads
        return 0;
    }

    // ----- CP-visible host memory (command ring + DMA staging) -----
    // VRAM carved from the PIN-window aperture: the CP addresses cp_addr
    // directly; the host writes the same bytes through PIN_BASE_ADDR.
    int host_mem_alloc(uint64_t size, void** host_ptr, uint64_t* cp_addr) {
        uint64_t asize = aligned_size(size, CACHE_BLOCK_SIZE);
        uint64_t addr  = 0;
        std::lock_guard<std::mutex> g(host_mu_);
        CHECK_ERR(host_mem_.allocate(asize, &addr), { return err; });
        *host_ptr = reinterpret_cast<void*>(PIN_BASE_ADDR + addr);
        *cp_addr  = addr;
        return 0;
    }

    int host_mem_free(uint64_t cp_addr) {
        std::lock_guard<std::mutex> g(host_mu_);
        return host_mem_.release(cp_addr);
    }

private:
    std::mutex      host_mu_;
    MemoryAllocator host_mem_;   // CP-visible host-memory aperture
};

#include <callbacks.inc>
