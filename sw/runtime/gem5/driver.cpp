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
    // PIO and PIN regions are mapped by the gem5 SE-mode Python config
    // before this binary runs; no mmap() is needed.
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

// Full store barrier: ensures ring-buffer payloads are visible before
// Q_TAIL_HI is written. Arch-specific fence instruction.
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
