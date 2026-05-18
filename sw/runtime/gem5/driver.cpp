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

namespace vortex {

namespace {
bool g_inited = false;
}

int drv_init() {
    // The two fixed regions (PIO and PIN) are mapped by the gem5
    // SE-mode setup before this binary runs. No mmap() here because
    // SE-mode has no /dev/vortex; the Python config arranges the
    // address space directly. If this runtime is ever ported to a
    // real OS with a kernel driver, drv_init() becomes
    // open("/dev/vortex_gem5") + mmap() for both regions.
    g_inited = true;
    return 0;
}

void drv_close() {
    g_inited = false;
}

uint32_t mmio_read32(uint32_t offset) {
    auto* p = reinterpret_cast<volatile uint32_t*>(PIO_BASE_ADDR + offset);
    return *p;
}

void mmio_write32(uint32_t offset, uint32_t value) {
    auto* p = reinterpret_cast<volatile uint32_t*>(PIO_BASE_ADDR + offset);
    *p = value;
}

// Publish prior stores before the next MMIO write. The host CPU model
// in gem5 (especially out-of-order variants like O3CPU) can reorder
// MMIO writes and surrounding stores; the dispatcher must guarantee
// that ring-buffer payloads land in device memory before Q_TAIL_HI is
// observed by the CP. The barrier is per-HOST_ARCH.
void mmio_fence() {
#if defined(__x86_64__) || defined(__i386__)
    __asm__ __volatile__ ("mfence" ::: "memory");
#elif defined(__aarch64__) || defined(__arm__)
    __asm__ __volatile__ ("dmb sy" ::: "memory");
#else
    __asm__ __volatile__ ("" ::: "memory");
#endif
}

} // namespace vortex
