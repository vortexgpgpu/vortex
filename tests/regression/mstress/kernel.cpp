#include <vx_spawn2.h>
#include "common.h"

__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
	size_t stride      = arg->stride;
	uint32_t* addr_ptr = (uint32_t*)arg->src0_addr;
	uint32_t* src_ptr  = (uint32_t*)arg->src1_addr;
	uint32_t* dst_ptr  = (uint32_t*)arg->dst_addr;

	size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
	size_t offset = gid * stride;

	for (size_t i = 0; i < stride; ++i) {
		uint32_t value = 0;
		for (uint32_t j = 0; j < NUM_LOADS; ++j) {
			size_t addr    = offset + i + j;
			uint32_t index = addr_ptr[addr];
			value = value * 31 + src_ptr[index];
		}
		dst_ptr[offset+i] = value;
	}
}
