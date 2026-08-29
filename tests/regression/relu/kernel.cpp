#include <vx_spawn2.h>
#include "common.h"

__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
	auto src0_ptr = reinterpret_cast<TYPE*>(arg->src0_addr);
	auto dst_ptr  = reinterpret_cast<TYPE*>(arg->dst_addr);

	uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
	TYPE value = src0_ptr[gid];
	dst_ptr[gid] = (value < 0) ? TYPE(0) : value;
}
