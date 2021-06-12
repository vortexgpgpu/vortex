
#include <VX_config.h>
#include <vx_intrinsics.h>
#include <stdint.h>

#define DUMP_CSR_4(d, s) \
    csr_mem[d + 0] = vx_csr_read(s + 0); \
    csr_mem[d + 1] = vx_csr_read(s + 1); \
    csr_mem[d + 2] = vx_csr_read(s + 2); \
    csr_mem[d + 3] = vx_csr_read(s + 3);

#define DUMP_CSR_32(d, s) \
    DUMP_CSR_4(d + 0,  s + 0)  \
    DUMP_CSR_4(d + 4,  s + 4)  \
    DUMP_CSR_4(d + 8,  s + 8)  \
    DUMP_CSR_4(d + 12, s + 12) \
    DUMP_CSR_4(d + 16, s + 16) \
    DUMP_CSR_4(d + 20, s + 20) \
    DUMP_CSR_4(d + 24, s + 24) \
    DUMP_CSR_4(d + 28, s + 28)

void vx_perf_dump() {
    int core_id = vx_core_id();
    uint32_t* const csr_mem = (uint32_t*)(IO_CSR_ADDR + 64 * sizeof(uint32_t) * core_id);
    DUMP_CSR_32(0,  CSR_MPM_BASE)
    DUMP_CSR_32(32, CSR_MPM_BASE_H)
}