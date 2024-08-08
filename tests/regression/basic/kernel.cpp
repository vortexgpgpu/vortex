#include <vx_intrinsics.h>
#include "common.h"

int main() {
	kernel_arg_t* __UNIFORM__ arg = (kernel_arg_t*)csr_read(VX_CSR_MSCRATCH);
	uint32_t count   = arg->count;
	int32_t* src_ptr = (int32_t*)arg->src_addr;
	int32_t* dst_ptr = (int32_t*)arg->dst_addr;

	uint32_t offset  = vx_core_id() * count;

	for (uint32_t i = 0; i < count; ++i) {
		dst_ptr[offset + i] = src_ptr[offset + i];
	}

	return 0;
}
