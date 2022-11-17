#include <stdint.h>
#include <vx_intrinsics.h>
#include <vx_spawn.h>
#include "common.h"

int main() {
	kernel_arg_t* arg = (kernel_arg_t*)KERNEL_ARG_DEV_MEM_ADDR;

	uint32_t size    = arg->size;
	int32_t* src_ptr = (int32_t*)arg->src_addr;
	int32_t* dst_ptr = (int32_t*)arg->dst_addr;
	
	for (uint32_t i = 0; i < size; ++i) {
		dst_ptr[i] = src_ptr[i];
	}

	return 0;
}
