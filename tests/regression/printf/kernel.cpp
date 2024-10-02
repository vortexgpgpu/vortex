#include <vx_spawn.h>
#include <vx_print.h>
#include "common.h"

void kernel_body(kernel_arg_t* __UNIFORM__ arg) {
	uint32_t cid = vx_core_id();
	char* src_ptr = (char*)arg->src_addr;
	char value = 'A' + src_ptr[blockIdx.x];
	vx_printf("cid=%d: task=%d, value=%c\n", cid, blockIdx.x, value);
}

int main() {
	kernel_arg_t* arg = (kernel_arg_t*)csr_read(VX_CSR_MSCRATCH);
	return vx_spawn_threads(1, &arg->num_points, nullptr, (vx_kernel_func_cb)kernel_body, arg);
}
