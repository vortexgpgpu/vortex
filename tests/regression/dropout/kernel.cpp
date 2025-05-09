#include <vx_spawn.h>
#include <cstdlib>
#include "common.h"

void kernel_body(kernel_arg_t* __UNIFORM__ arg) {
	auto src0_ptr = reinterpret_cast<TYPE*>(arg->src0_addr);
	auto dst_ptr  = reinterpret_cast<TYPE*>(arg->dst_addr);
	
	auto dropout_p = arg->dropout_p;
	auto multiplier = arg->multiplier;

	float rand_value = RandomFloat(WangHash(blockIdx.x));

	TYPE scaled_value = src0_ptr[blockIdx.x] * multiplier;
	dst_ptr[blockIdx.x] = (rand_value < dropout_p) ? 0.0 : scaled_value;
}

int main() {
	kernel_arg_t* arg = (kernel_arg_t*)csr_read(VX_CSR_MSCRATCH);
	return vx_spawn_threads(1, &arg->num_points, nullptr, (vx_kernel_func_cb)kernel_body, arg);
}
