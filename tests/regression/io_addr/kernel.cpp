#include <vx_spawn2.h>
#include "common.h"

extern "C" void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
	uint64_t* src_ptr = (uint64_t*)arg->src_addr;
	uint32_t* dst_ptr = (uint32_t*)arg->dst_addr;

	uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
	int32_t* addr_ptr = (int32_t*)(src_ptr[gid]);
	dst_ptr[gid] = *addr_ptr;
}
