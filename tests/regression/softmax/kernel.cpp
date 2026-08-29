#include <vx_spawn2.h>
#include "common.h"

__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
	auto src0_ptr = reinterpret_cast<TYPE*>(arg->src0_addr);
	auto dst_ptr  = reinterpret_cast<TYPE*>(arg->dst_addr);

	uint32_t num_cols = arg->num_cols;
	uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
	if (row >= arg->num_rows)
		return;

	uint32_t offset = row * num_cols;

	// 1. Find the max value
	TYPE max = 0.0;
	for (uint32_t i = 0; i < num_cols; i++) {
		auto curr = src0_ptr[offset + i];
		if (curr > max) max = curr;
	}

	// 2. Subtract max and compute exp
	TYPE sum = 0.0;
	for (uint32_t i = 0; i < num_cols; i++) {
		auto computed = exp(src0_ptr[offset + i] - max);
		src0_ptr[offset + i] = computed;
		sum += computed;
	}

	// 3. Normalize
	for (uint32_t i = 0; i < num_cols; i++) {
		dst_ptr[offset + i] = src0_ptr[offset + i] / sum;
	}
}
