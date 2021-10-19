#include <stdint.h>
#include <vx_intrinsics.h>
#include "common.h"

void main() {
	kernel_arg_t* arg = (kernel_arg_t*)KERNEL_ARG_DEV_MEM_ADDR;
	uint32_t count   = arg->count;
	int32_t* src_ptr = (int32_t*)arg->src_ptr;
	int32_t* dst_ptr = (int32_t*)arg->dst_ptr;
	
	uint32_t offset  = vx_core_id() * count;
	
	for (uint32_t i = 0; i < count; ++i) {
		dst_ptr[offset + i] = src_ptr[offset + i];
	}
}