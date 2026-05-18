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

#include "driver.h"

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <unordered_map>

namespace vortex {

namespace {

// Trivial bump allocator for the pinned region. A real implementation
// would use a free-list; for now this is the simplest thing that lets
// upload/download cache a single staging buffer indefinitely.
struct PinAllocator {
    uintptr_t base = PIN_BASE_ADDR;
    uintptr_t cur  = PIN_BASE_ADDR;
    std::unordered_map<uintptr_t, uint64_t> live;  // ptr → size for free()

    int allocate(uint64_t size, void** host_ptr, uint64_t* ioaddr) {
        // Cache-line align (64) to match the OPAE staging-buffer model.
        const uint64_t aligned = (size + 63) & ~uint64_t(63);
        if (cur + aligned > base + PIN_REGION_SIZE) {
            std::fprintf(stderr,
                         "[VXDRV-gem5] pin region OOM: requested %llu, "
                         "available %llu\n",
                         (unsigned long long)aligned,
                         (unsigned long long)(base + PIN_REGION_SIZE - cur));
            return -1;
        }
        const uintptr_t ptr = cur;
        cur += aligned;
        live.emplace(ptr, aligned);
        *host_ptr = reinterpret_cast<void*>(ptr);
        *ioaddr   = static_cast<uint64_t>(ptr);  // identity v→p (see driver.h)
        return 0;
    }

    void release(void* host_ptr) {
        // Trivial allocator: no reclaim until close(). The legacy OPAE
        // driver's `ensure_staging` recycles its single buffer the same
        // way; this is fine for the OPAE-shaped workload (one staging
        // buffer per device handle, grown on demand).
        live.erase(reinterpret_cast<uintptr_t>(host_ptr));
    }

    void reset() { cur = base; live.clear(); }
};

PinAllocator g_pin;
bool         g_inited = false;

} // namespace

int drv_init() {
    if (g_inited) return 0;
    // The two fixed regions (PIO and PIN) are expected to be already
    // mapped by the gem5 SE-mode setup before this binary runs. We do
    // NOT call mmap() here because SE-mode has no /dev/vortex; the
    // Python config arranges the address space directly.
    //
    // If/when this runtime is ported to a real OS with a kernel driver,
    // drv_init() will become an open("/dev/vortex_gem5") + mmap() pair.
    g_inited = true;
    g_pin.reset();
    return 0;
}

void drv_close() {
    if (!g_inited) return;
    g_pin.reset();
    g_inited = false;
}

uint64_t mmio_read64(uint64_t offset) {
    auto* p = reinterpret_cast<volatile uint64_t*>(PIO_BASE_ADDR + offset);
    return *p;
}

void mmio_write64(uint64_t offset, uint64_t value) {
    auto* p = reinterpret_cast<volatile uint64_t*>(PIO_BASE_ADDR + offset);
    *p = value;
}

// Memory barrier before kicking a command. The host CPU model in
// gem5 (especially out-of-order variants like O3CPU) can reorder
// MMIO writes; the runtime must publish the args before the
// CMD_TYPE write or the device sees stale/uninitialized args. B14
// in the proposal's bug catalog calls this out explicitly.
void mmio_fence() {
#if defined(__x86_64__) || defined(__i386__)
    __asm__ __volatile__ ("mfence" ::: "memory");
#elif defined(__aarch64__) || defined(__arm__)
    __asm__ __volatile__ ("dmb sy" ::: "memory");
#else
    // Fall back to a compiler-only fence. Untested architectures
    // should add their own asm.
    __asm__ __volatile__ ("" ::: "memory");
#endif
}

int drv_pin_buffer(uint64_t size, void** host_ptr, uint64_t* ioaddr) {
    if (!g_inited) {
        std::fprintf(stderr, "[VXDRV-gem5] drv_pin_buffer called before drv_init\n");
        return -1;
    }
    return g_pin.allocate(size, host_ptr, ioaddr);
}

void drv_release_buffer(void* host_ptr) {
    if (!g_inited) return;
    g_pin.release(host_ptr);
}

} // namespace vortex
