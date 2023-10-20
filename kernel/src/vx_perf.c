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


#include <VX_config.h>
#include <VX_types.h>
#include <vx_intrinsics.h>
#include <stdint.h>

#define DUMP_CSR_4(d, s) \
    csr_mem[d + 0] = csr_read(s + 0); \
    csr_mem[d + 1] = csr_read(s + 1); \
    csr_mem[d + 2] = csr_read(s + 2); \
    csr_mem[d + 3] = csr_read(s + 3);

#define DUMP_CSR_32(d, s) \
    DUMP_CSR_4(d + 0,  s + 0)  \
    DUMP_CSR_4(d + 4,  s + 4)  \
    DUMP_CSR_4(d + 8,  s + 8)  \
    DUMP_CSR_4(d + 12, s + 12) \
    DUMP_CSR_4(d + 16, s + 16) \
    DUMP_CSR_4(d + 20, s + 20) \
    DUMP_CSR_4(d + 24, s + 24) \
    DUMP_CSR_4(d + 28, s + 28)

#ifdef __cplusplus
extern "C" {
#endif

void vx_perf_dump() {
    int core_id = vx_core_id();
    uint32_t* const csr_mem = (uint32_t*)(IO_CSR_ADDR + 64 * sizeof(uint32_t) * core_id);
    DUMP_CSR_32(0,  VX_CSR_MPM_BASE)
    DUMP_CSR_32(32, VX_CSR_MPM_BASE_H)
}

#ifdef __cplusplus
}
#endif
