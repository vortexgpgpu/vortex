#include <vx_spawn.h>
#include "common.h"

void kernel_body(kernel_arg_t* __UNIFORM__ arg) {
	uint64_t* src_ptr = (uint64_t*)arg->src_addr;
	uint32_t* dst_ptr = (uint32_t*)arg->dst_addr;

	int32_t* addr_ptr = (int32_t*)(src_ptr[blockIdx.x]);

	dst_ptr[blockIdx.x] = *addr_ptr;
}

int main() {
	kernel_arg_t* arg = (kernel_arg_t*)csr_read(VX_CSR_MSCRATCH);
	return vx_spawn_threads(1, &arg->num_points, nullptr, (vx_kernel_func_cb)kernel_body, arg);
}
