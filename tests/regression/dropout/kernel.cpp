#include <vx_spawn2.h>
#include <cstdlib>
#include "common.h"

extern "C" void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
	auto src0_ptr = reinterpret_cast<TYPE*>(arg->src0_addr);
	auto dst_ptr  = reinterpret_cast<TYPE*>(arg->dst_addr);

	auto dropout_p  = arg->dropout_p;
	auto multiplier = arg->multiplier;

	uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
	float rand_value = RandomFloat(WangHash(gid));
	TYPE scaled_value = src0_ptr[gid] * multiplier;
	dst_ptr[gid] = (rand_value < dropout_p) ? 0.0 : scaled_value;
}
