#include <vx_spawn2.h>
#include "common.h"

__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
	uint32_t stride    = arg->stride;
	uint32_t* addr_ptr = (uint32_t*)arg->src0_addr;
	float* src_ptr     = (float*)arg->src1_addr;
	float* dst_ptr     = (float*)arg->dst_addr;

	uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t offset = gid * stride;

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
