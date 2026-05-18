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

// gem5 host runtime backend. Provides the standard Vortex `vx_*`
// C API (declared in sw/runtime/include/vortex.h) on top of the
// OPAE-shaped MMIO command protocol talking to the gem5 VortexGPGPU
// device through driver.{cpp,h}.
//
// Shape mirrors sw/runtime/opae/vortex.cpp but is simpler:
//   - No libopae dispatch; driver.h's mmio_{read,write}64 talks
//     directly to PIO_BASE_ADDR.
//   - No UUID enumeration / fpga_token dance — the gem5 device is
//     always at the fixed PIO range.
//   - Device caps come from compile-time VX_config.h macros (the
//     host runtime and the device library are built from the same
//     source tree, so they agree by construction).
//   - mmio_fence() before every CMD_TYPE write (B14 in proposal §2.2).

#include <common.h>
#include <util.h>          // log2floor / log2ceil / is_aligned / aligned_size
#include "driver.h"

#include <vortex_opae.h>
#include <sched.h>         // sched_yield (gem5 SE-mode-safe back-off)

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <iostream>
#include <sstream>
#include <unordered_map>

using namespace vortex;

// MMIO offsets (byte addresses). Sourced from vortex_opae.h's
// AFU_IMAGE_MMIO_* DWORD offsets times 4. Same layout as
// sw/runtime/opae/vortex.cpp:47–56.
#define CMD_MEM_READ     AFU_IMAGE_CMD_MEM_READ
#define CMD_MEM_WRITE    AFU_IMAGE_CMD_MEM_WRITE
#define CMD_RUN          AFU_IMAGE_CMD_RUN
#define CMD_DCR_WRITE    AFU_IMAGE_CMD_DCR_WRITE
#define CMD_DCR_READ     AFU_IMAGE_CMD_DCR_READ

#define MMIO_CMD_TYPE    (AFU_IMAGE_MMIO_CMD_TYPE * 4)
#define MMIO_CMD_ARG0    (AFU_IMAGE_MMIO_CMD_ARG0 * 4)
#define MMIO_CMD_ARG1    (AFU_IMAGE_MMIO_CMD_ARG1 * 4)
#define MMIO_CMD_ARG2    (AFU_IMAGE_MMIO_CMD_ARG2 * 4)
#define MMIO_STATUS      (AFU_IMAGE_MMIO_STATUS * 4)
#define MMIO_DCR_RSP     (AFU_IMAGE_MMIO_DCR_RSP * 4)

#define STATUS_STATE_BITS 8

// Issue a CMD_TYPE write. Centralised so the memory barrier before
// the trigger MMIO is impossible to forget (B14). All callers must
// have written ARG0/1/2 first.
static inline void issue_cmd(uint64_t cmd) {
    mmio_fence();
    mmio_write64(MMIO_CMD_TYPE, cmd);
}

///////////////////////////////////////////////////////////////////////////////

class vx_device {
public:
    vx_device()
        : global_mem_(ALLOC_BASE_ADDR,
                      GLOBAL_MEM_SIZE - ALLOC_BASE_ADDR,
                      RAM_PAGE_SIZE,
                      CACHE_BLOCK_SIZE),
          staging_ioaddr_(0),
          staging_ptr_(nullptr),
          staging_size_(0) {}

    ~vx_device() {
        if (staging_ptr_ != nullptr) {
            drv_release_buffer(staging_ptr_);
            staging_ptr_   = nullptr;
            staging_size_  = 0;
            staging_ioaddr_ = 0;
        }
        drv_close();
    }

    int init() {
        if (drv_init() != 0) {
            std::fprintf(stderr, "[VXDRV] drv_init failed\n");
            return -1;
        }
        return 0;
    }

    // Compile-time capability table. Mirrors sw/runtime/simx/vortex.cpp:
    // 51–103: the runtime and the SimX-side device library share a
    // build tree, so the same VX_config.h macros are authoritative
    // on both sides.
    int get_caps(uint32_t caps_id, uint64_t *value) {
        switch (caps_id) {
        case VX_CAPS_VERSION:         *value = IMPLEMENTATION_ID; break;
        case VX_CAPS_NUM_THREADS:     *value = NUM_THREADS; break;
        case VX_CAPS_NUM_WARPS:       *value = NUM_WARPS; break;
        case VX_CAPS_NUM_CORES:       *value = NUM_CORES * NUM_CLUSTERS; break;
        case VX_CAPS_NUM_CLUSTERS:    *value = NUM_CLUSTERS; break;
        case VX_CAPS_SOCKET_SIZE:     *value = SOCKET_SIZE; break;
        case VX_CAPS_ISSUE_WIDTH:     *value = ISSUE_WIDTH; break;
        case VX_CAPS_CACHE_LINE_SIZE: *value = CACHE_BLOCK_SIZE; break;
        case VX_CAPS_GLOBAL_MEM_SIZE: *value = GLOBAL_MEM_SIZE; break;
        case VX_CAPS_LOCAL_MEM_SIZE:  *value = (1 << LMEM_LOG_SIZE); break;
        case VX_CAPS_ISA_FLAGS:
            *value = ((uint64_t(MISA_EXT)) << 32)
                   | ((log2floor(XLEN) - 4) << 30)
                   |   MISA_STD;
            break;
        case VX_CAPS_NUM_MEM_BANKS:   *value = PLATFORM_MEMORY_NUM_BANKS; break;
        case VX_CAPS_MEM_BANK_SIZE:   *value = 1ull << (MEM_ADDR_WIDTH / PLATFORM_MEMORY_NUM_BANKS); break;
        case VX_CAPS_CLOCK_RATE:      *value = 0; break;
        case VX_CAPS_PEAK_MEM_BW:     *value = PLATFORM_MEMORY_PEAK_BW; break;
        default:
            std::fprintf(stderr, "[VXDRV] invalid caps id: %u\n", caps_id);
            return -1;
        }
        return 0;
    }

    int mem_alloc(uint64_t size, int flags, uint64_t *dev_addr) {
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
        // Access control is enforced by the device's RAM ACL (in
        // libvortex-gem5.so). The host runtime has nothing to do here.
        return 0;
    }

    int mem_info(uint64_t *mem_free, uint64_t *mem_used) const {
        if (mem_free) *mem_free = global_mem_.free();
        if (mem_used) *mem_used = global_mem_.allocated();
        return 0;
    }

    int copy(uint64_t /*dest*/, uint64_t /*src*/, uint64_t /*size*/) {
        // Device-to-device copy not in the OPAE command set (no
        // CMD_MEM_COPY); the OPAE FPGA path goes through libopae's
        // fpgaCopyBuffer which we don't have. Leave unimplemented
        // for Phase 4; can be added by extending the device with a
        // new CMD type in a later phase.
        std::fprintf(stderr, "[VXDRV] copy() not supported in gem5 backend\n");
        return -1;
    }

    int upload(uint64_t dev_addr, const void *host_ptr, uint64_t size) {
        if (!is_aligned(dev_addr, CACHE_BLOCK_SIZE)) return -1;
        const uint64_t asize = aligned_size(size, CACHE_BLOCK_SIZE);
        if (dev_addr + asize > GLOBAL_MEM_SIZE) return -1;

        if (this->ready_wait(VX_MAX_TIMEOUT) != 0) return -1;
        if (this->ensure_staging(asize) != 0)     return -1;

        std::memcpy(staging_ptr_, host_ptr, size);

        const auto ls_shift = log2ceil(CACHE_BLOCK_SIZE);
        mmio_write64(MMIO_CMD_ARG0, staging_ioaddr_ >> ls_shift);
        mmio_write64(MMIO_CMD_ARG1, dev_addr        >> ls_shift);
        mmio_write64(MMIO_CMD_ARG2, asize           >> ls_shift);
        issue_cmd(CMD_MEM_WRITE);

        return this->ready_wait(VX_MAX_TIMEOUT);
    }

    int download(void *host_ptr, uint64_t dev_addr, uint64_t size) {
        if (!is_aligned(dev_addr, CACHE_BLOCK_SIZE)) return -1;
        const uint64_t asize = aligned_size(size, CACHE_BLOCK_SIZE);
        if (dev_addr + asize > GLOBAL_MEM_SIZE) return -1;

        // Drain dirty cache lines all the way to VRAM before reading
        // back, per B9 in proposal §2.2. One DCR_READ on the magic
        // cache-flush DCR per core; the device routes it through
        // Processor::flush_caches().
        {
            uint64_t num_cores;
            CHECK_ERR(this->get_caps(VX_CAPS_NUM_CORES, &num_cores), { return err; });
            uint32_t dummy;
            for (uint32_t cid = 0; cid < (uint32_t)num_cores; ++cid) {
                CHECK_ERR(this->dcr_read(VX_DCR_BASE_CACHE_FLUSH, cid, &dummy),
                          { return err; });
            }
        }

        if (this->ready_wait(VX_MAX_TIMEOUT) != 0) return -1;
        if (this->ensure_staging(asize) != 0)     return -1;

        const auto ls_shift = log2ceil(CACHE_BLOCK_SIZE);
        mmio_write64(MMIO_CMD_ARG0, staging_ioaddr_ >> ls_shift);
        mmio_write64(MMIO_CMD_ARG1, dev_addr        >> ls_shift);
        mmio_write64(MMIO_CMD_ARG2, asize           >> ls_shift);
        issue_cmd(CMD_MEM_READ);

        if (this->ready_wait(VX_MAX_TIMEOUT) != 0) return -1;

        std::memcpy(host_ptr, staging_ptr_, size);
        return 0;
    }

    int start() {
        issue_cmd(CMD_RUN);
        return 0;
    }

    // Poll MMIO_STATUS; the high bits carry stdout/stderr text from
    // device-side printf — same protocol as sw/runtime/opae/vortex.cpp.
    // Uses sched_yield() to back off between polls (gem5 SE-mode
    // doesn't implement clock_nanosleep which glibc's nanosleep()
    // routes through; sched_yield is in the syscall_tbl64 ignore
    // list and returns immediately, which inside gem5 just means
    // the next poll happens on the next simulated CPU instruction).
    int ready_wait(uint64_t timeout) {
        std::unordered_map<uint32_t, std::stringstream> print_bufs;
        const uint64_t step_ms = 1;

        for (;;) {
            uint64_t status = mmio_read64(MMIO_STATUS);

            // Drain any console data the device produced.
            uint32_t cout_data = status >> STATUS_STATE_BITS;
            if (cout_data & 0x1) {
                do {
                    const char     cout_char = (cout_data >> 1) & 0xff;
                    const uint32_t cout_tid  = (cout_data >> 9) & 0xff;
                    auto& ss = print_bufs[cout_tid];
                    ss << cout_char;
                    if (cout_char == '\n') {
                        std::cout << std::dec << "#" << cout_tid
                                  << ": " << ss.str() << std::flush;
                        ss.str("");
                    }
                    status = mmio_read64(MMIO_STATUS);
                    cout_data = status >> STATUS_STATE_BITS;
                } while (cout_data & 0x1);
            }

            const uint32_t state = status & ((1 << STATUS_STATE_BITS) - 1);
            if (state == 0 || timeout == 0) {
                for (auto& kv : print_bufs) {
                    auto s = kv.second.str();
                    if (!s.empty()) {
                        std::cout << "#" << kv.first << ": " << s << std::endl;
                    }
                }
                if (state != 0) {
                    std::fprintf(stdout, "[VXDRV] ready-wait timed out: state=%u\n", state);
                    return -1;
                }
                return 0;
            }

            sched_yield();
            timeout -= step_ms;
        }
    }

    int dcr_write(uint32_t addr, uint32_t value) {
        mmio_write64(MMIO_CMD_ARG0, addr);
        mmio_write64(MMIO_CMD_ARG1, value);
        issue_cmd(CMD_DCR_WRITE);
        return 0;
    }

    int dcr_read(uint32_t addr, uint32_t tag, uint32_t *value) {
        mmio_write64(MMIO_CMD_ARG0, addr);
        mmio_write64(MMIO_CMD_ARG1, tag);
        issue_cmd(CMD_DCR_READ);
        if (this->ready_wait(VX_MAX_TIMEOUT) != 0) return -1;
        *value = static_cast<uint32_t>(mmio_read64(MMIO_DCR_RSP));
        return 0;
    }

private:
    int ensure_staging(uint64_t size) {
        if (staging_size_ >= size) return 0;
        if (staging_ptr_ != nullptr) {
            drv_release_buffer(staging_ptr_);
            staging_ptr_   = nullptr;
            staging_size_  = 0;
            staging_ioaddr_ = 0;
        }
        if (drv_pin_buffer(size, reinterpret_cast<void**>(&staging_ptr_),
                           &staging_ioaddr_) != 0) {
            return -1;
        }
        staging_size_ = size;
        return 0;
    }

    MemoryAllocator global_mem_;
    uint64_t staging_ioaddr_;
    uint8_t* staging_ptr_;
    uint64_t staging_size_;
};

#include <callbacks.inc>
