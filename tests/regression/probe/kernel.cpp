#include <vx_spawn2.h>
#include "common.h"

__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
	auto src0_ptr = reinterpret_cast<TYPE*>(arg->src0_addr);
	auto src1_ptr = reinterpret_cast<TYPE*>(arg->src1_addr);
	auto dst_ptr = reinterpret_cast<TYPE*>(arg->dst_addr);

	// 2D global index: use threadIdx.x and threadIdx.y independently
	// so that CTA thread-ID decomposition errors are not masked by
	// the flat tid = y*bdx+x reconstruction.
	uint32_t gx = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t gy = blockIdx.y * blockDim.y + threadIdx.y;
	uint32_t gid = gy * arg->dim_x + gx;
	uint32_t count = arg->task_size;
	uint32_t offset = gid * count;

	for (uint32_t i = 0; i < count; ++i) {
		dst_ptr[offset+i] = src0_ptr[offset+i] + src1_ptr[offset+i];
	}
}
