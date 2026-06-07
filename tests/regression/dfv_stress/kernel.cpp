#include <vx_spawn.h>
#include <vx_intrinsics.h>
#include "common.h"

void kernel_body(kernel_arg_t* __UNIFORM__ arg) {
    auto src0 = reinterpret_cast<TYPE*>(arg->src0_addr);
    auto src1 = reinterpret_cast<TYPE*>(arg->src1_addr);
    auto dst  = reinterpret_cast<TYPE*>(arg->dst_addr);
    uint32_t stride = arg->stride;

    // Strided access: each thread accesses elements separated by 'stride'
    // High stride → different cache lines per access → more cache misses
    // Low stride  → same cache line reuse → fewer misses
    uint32_t idx = blockIdx.x * stride;
    dst[idx] = src0[idx] + src1[idx];
}

int main() {
    kernel_arg_t* arg = (kernel_arg_t*)csr_read(VX_CSR_MSCRATCH);

    if (arg->enable_dfv_test) {
        // Phase-specific DFV configuration
        // Each phase exercises a different contention scenario
        uint32_t phase = arg->dfv_phase;

        csr_write(VX_CSR_DFV_CTRL, 1);
        csr_write(VX_CSR_DFV_RANDOM_SEED, 0xABCDEF00);

        switch (phase) {
        case 0:
            // Phase 0: icache + dcache req stalls (L1 arbiter contention)
            csr_write(VX_CSR_DFV_SET_THRESHOLD, 240);
            csr_write(VX_CSR_DFV_RELEASE_THRESHOLD, 240);
            csr_write(VX_CSR_DFV_RELEASE_DELAY, 0x0000);
            csr_write(VX_CSR_DFV_ICACHE_FILL_REQ_STALL, 1);
            csr_write(VX_CSR_DFV_DCACHE_FILL_REQ_STALL, 1);
            csr_write(VX_CSR_DFV_WRITEBACK_STALL, 0);
            csr_write(VX_CSR_DFV_DCACHE_FILL_RSP_STALL, 0);
            break;
        case 1:
            // Phase 1: dcache + fill stalls (cache bank contention)
            csr_write(VX_CSR_DFV_SET_THRESHOLD, 240);
            csr_write(VX_CSR_DFV_RELEASE_THRESHOLD, 240);
            csr_write(VX_CSR_DFV_RELEASE_DELAY, 0x1000);
            csr_write(VX_CSR_DFV_ICACHE_FILL_REQ_STALL, 0);
            csr_write(VX_CSR_DFV_DCACHE_FILL_REQ_STALL, 1);
            csr_write(VX_CSR_DFV_WRITEBACK_STALL, 0);
            csr_write(VX_CSR_DFV_DCACHE_FILL_RSP_STALL, 1);
            break;
        case 2:
            // Phase 2: writeback + dcache (scoreboard pressure + memory)
            csr_write(VX_CSR_DFV_SET_THRESHOLD, 200);
            csr_write(VX_CSR_DFV_RELEASE_THRESHOLD, 200);
            csr_write(VX_CSR_DFV_RELEASE_DELAY, 0x0000);
            csr_write(VX_CSR_DFV_ICACHE_FILL_REQ_STALL, 0);
            csr_write(VX_CSR_DFV_DCACHE_FILL_REQ_STALL, 1);
            csr_write(VX_CSR_DFV_WRITEBACK_STALL, 1);
            csr_write(VX_CSR_DFV_DCACHE_FILL_RSP_STALL, 0);
            break;
        case 3:
            // Phase 3: all stalls active (maximum stress)
            csr_write(VX_CSR_DFV_SET_THRESHOLD, 240);
            csr_write(VX_CSR_DFV_RELEASE_THRESHOLD, 250);
            csr_write(VX_CSR_DFV_RELEASE_DELAY, 0x1000);
            csr_write(VX_CSR_DFV_ICACHE_FILL_REQ_STALL, 1);
            csr_write(VX_CSR_DFV_DCACHE_FILL_REQ_STALL, 1);
            csr_write(VX_CSR_DFV_WRITEBACK_STALL, 1);
            csr_write(VX_CSR_DFV_DCACHE_FILL_RSP_STALL, 1);
            break;
        }
    }

    int __ret = vx_spawn_threads(1, &arg->num_points, nullptr, (vx_kernel_func_cb)kernel_body, arg);

    if (arg->enable_dfv_test) {
        csr_write(VX_CSR_DFV_CTRL, 0);
    }
    return __ret;
}
