#include <vx_spawn.h>
#include "common.h"

void kernel_body(kernel_arg_t* __UNIFORM__ arg) {
	auto src0_ptr = reinterpret_cast<TYPE*>(arg->src0_addr);
	auto dst_ptr  = reinterpret_cast<TYPE*>(arg->dst_addr);
	
	TYPE value = src0_ptr[blockIdx.x];

	dst_ptr[blockIdx.x] = (value < 0) ? 0 : value;
}

int main() {
	kernel_arg_t* arg = (kernel_arg_t*)csr_read(VX_CSR_MSCRATCH);
	return vx_spawn_threads(1, &arg->num_points, nullptr, (vx_kernel_func_cb)kernel_body, arg);
}
