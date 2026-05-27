#include <vx_spawn.h>
#include "common.h"

void kernel_body(kernel_arg_t* __UNIFORM__ arg) {
	uint32_t stride    = arg->stride;
	uint32_t* addr_ptr = (uint32_t*)arg->src0_addr;
	float* src_ptr     = (float*)arg->src1_addr;
	float* dst_ptr     = (float*)arg->dst_addr;

	uint32_t offset = blockIdx.x * stride;

	for (uint32_t i = 0; i < stride; ++i) {
		float value = 0.0f;
		for (uint32_t j = 0; j < NUM_LOADS; ++j) {
			uint32_t addr  = offset + i + j;
			uint32_t index = addr_ptr[addr];
			value *= src_ptr[index];
		}
		dst_ptr[offset+i] = value;
	}
}

int main() {
	kernel_arg_t* arg = (kernel_arg_t*)csr_read(VX_CSR_MSCRATCH);
    if (arg->enable_dfv_test) {
        csr_write(VX_CSR_DFV_CTRL, 1);
        csr_write(VX_CSR_DFV_RANDOM_SEED, 0xABCDEF00);
        csr_write(VX_CSR_DFV_SET_THRESHOLD, 240);
        csr_write(VX_CSR_DFV_RELEASE_THRESHOLD, 65504);  // release when lfsr2[15:0] >= 65504 (~0.04%)
        csr_write(VX_CSR_DFV_RELEASE_DELAY, 0x1000);
        csr_write(VX_CSR_DFV_RELEASE_FOREVER, 1);
        csr_write(VX_CSR_DFV_THROTTLE_THRESHOLD, 0x1800);
        csr_write(VX_CSR_DFV_ICACHE_STALL, 0);
        csr_write(VX_CSR_DFV_DCACHE_STALL, 0);
        csr_write(VX_CSR_DFV_WRITEBACK_STALL, 0);
        csr_write(VX_CSR_DFV_FILL_STALL, 1);
    }
	int __ret = vx_spawn_threads(1, &arg->num_tasks, nullptr, (vx_kernel_func_cb)kernel_body, arg);
	if (arg->enable_dfv_test) {
	    csr_write(VX_CSR_DFV_CTRL, 0);
	}
	return __ret;
}
