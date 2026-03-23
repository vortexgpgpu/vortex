#include <vx_spawn2.h>
#include "common.h"

extern "C" void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
	auto A     = reinterpret_cast<TYPE*>(arg->src0_addr);
	auto x_old = reinterpret_cast<TYPE*>(arg->src1_addr);
	auto b     = reinterpret_cast<TYPE*>(arg->src2_addr);
	auto x_new = reinterpret_cast<TYPE*>(arg->dst_addr);
	auto n     = arg->size;

	uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= n)
		return;

	uint32_t index = i * n;
	double sum = 0.0;
	for (uint32_t j = 0; j < n; j++) {
		if (j != i)
			sum += A[index + j] * x_old[j];
	}
	x_new[i] = (b[i] - sum) / A[index + i];
}
