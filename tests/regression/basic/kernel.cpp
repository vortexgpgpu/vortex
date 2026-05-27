#include <vx_intrinsics.h>
#include "common.h"

int main() {
	kernel_arg_t* __UNIFORM__ arg = (kernel_arg_t*)csr_read(VX_CSR_MSCRATCH);
	uint32_t count   = arg->count;
	int32_t* src_ptr = (int32_t*)arg->src_addr;
	int32_t* dst_ptr = (int32_t*)arg->dst_addr;

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

	uint32_t offset  = vx_core_id() * count;

	for (uint32_t i = 0; i < count; ++i) {
		dst_ptr[offset + i] = src_ptr[offset + i];
	}

	if (arg->enable_dfv_test) {
		csr_write(VX_CSR_DFV_CTRL, 0);
	}
	return 0;
}
