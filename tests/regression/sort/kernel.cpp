#include <vx_spawn2.h>
#include "common.h"

extern "C" void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
	uint32_t num_points = arg->num_points;
	auto src_ptr = (TYPE*)arg->src_addr;
	auto dst_ptr = (TYPE*)arg->dst_addr;

	uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
	auto ref_value = src_ptr[gid];

	uint32_t pos = 0;
	for (uint32_t i = 0; i < num_points; ++i) {
		auto cur_value = src_ptr[i];
		pos += (cur_value < ref_value) || ((cur_value == ref_value) && (i < gid));
	}
	dst_ptr[pos] = ref_value;
}
