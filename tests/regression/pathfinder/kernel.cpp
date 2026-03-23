#include <vx_spawn2.h>
#include "common.h"

#define MIN(a, b) ((a) < (b) ? (a) : (b))

extern "C" void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
	auto wall = reinterpret_cast<TYPE*>(arg->src0_addr);
	auto src  = reinterpret_cast<TYPE*>(arg->src1_addr);
	auto dst  = reinterpret_cast<TYPE*>(arg->dst_addr);
	auto num_cols = arg->num_cols;

	uint32_t n = blockIdx.x * blockDim.x + threadIdx.x;
	if (n >= num_cols)
		return;

	TYPE min = src[n];
	if (n > 0)
		min = MIN(min, src[n - 1]);
	if (n < num_cols - 1)
		min = MIN(min, src[n + 1]);

	dst[n] = wall[n] + min;
}
