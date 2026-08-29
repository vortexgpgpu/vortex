#include <vx_spawn2.h>
#include <vx_print.h>
#include "common.h"

__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
	uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t cid = vx_core_id();
	char* src_ptr = (char*)arg->src_addr;
	char value = 'A' + src_ptr[gid];
	vx_printf("cid=%d: task=%d, value=%c\n", cid, gid, value);
}
