#include <vx_spawn.h>
#include "common.h"

void kernel_body(kernel_arg_t* __UNIFORM__ arg) {
	uint32_t num_points = arg->num_points;
	auto src_ptr = (TYPE*)arg->src_addr;
	auto dst_ptr = (TYPE*)arg->dst_addr;

	auto ref_value = src_ptr[blockIdx.x];

	uint32_t pos = 0;
	for (uint32_t i = 0; i < num_points; ++i) {
		auto cur_value = src_ptr[i];
		pos += (cur_value < ref_value) || ((cur_value == ref_value) && (i < blockIdx.x));
	}
	dst_ptr[pos] = ref_value;
}

int main() {
	kernel_arg_t* arg = (kernel_arg_t*)csr_read(VX_CSR_MSCRATCH);
	return vx_spawn_threads(1, &arg->num_points, nullptr, (vx_kernel_func_cb)kernel_body, arg);
}
