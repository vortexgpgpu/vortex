#include <stdlib.h>
#include <stdio.h>
#include "intrinsics/vx_intrinsics.h"
#include "vx_api/vx_api.h"
#include "common.h"

void kernel_body(void* arg) {
	struct kernel_arg_t* _arg = (struct kernel_arg_t*)(arg);
	int* x = (int*)_arg->src0_ptr;
	int* y = (int*)_arg->src1_ptr;
	int* z = (int*)_arg->dst_ptr;

	unsigned wid = vx_warp_gid();
	unsigned tid = vx_thread_id();

	unsigned i = ((wid * _arg->num_threads) + tid) * _arg->stride;

	for (unsigned j = 0; j < _arg->stride; ++j) {
		z[i+j] = x[i+j] + y[i+j];
	}
}

void main() {
	struct kernel_arg_t* arg = (struct kernel_arg_t*)KERNEL_ARG_DEV_MEM_ADDR;
	/*printf("num_warps=%d\n", arg->num_warps);
	printf("num_threads=%d\n", arg->num_threads);
	printf("stride=%d\n", arg->stride);
	printf("src0_ptr=0x%x\n", arg->src0_ptr);
	printf("src1_ptr=0x%x\n", arg->src1_ptr);
	printf("dst_ptr=0x%x\n", arg->dst_ptr);*/
	vx_spawn_warps(arg->num_warps, arg->num_threads, kernel_body, arg);
}