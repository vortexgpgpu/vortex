#include <stdlib.h>
#include <stdio.h>
#include "intrinsics/vx_intrinsics.h"
#include "vx_api/vx_api.h"
#include "common.h"

void kernel_body(void* arg) {
	struct kernel_arg_t* _arg = (struct kernel_arg_t*)(arg);
	int* src0_ptr = (int*)_arg->src0_ptr;
	int* src1_ptr = (int*)_arg->src1_ptr;
	int* dst_ptr  = (int*)_arg->dst_ptr;

	unsigned offset = vx_thread_gid() * _arg->stride;

	for (unsigned i = 0; i < _arg->stride; ++i) {
		dst_ptr[offset+i] = src0_ptr[offset+i] + src1_ptr[offset+i];
	}
}

void main() {
	struct kernel_arg_t* arg = (struct kernel_arg_t*)KERNEL_ARG_DEV_MEM_ADDR;
	/*printf("stride=%d\n", arg->stride);
	printf("src0_ptr=0x%src0\n", arg->src0_ptr);
	printf("src1_ptr=0x%src0\n", arg->src1_ptr);
	printf("dst_ptr=0x%src0\n", arg->dst_ptr);*/
	int num_warps = vx_num_warps();
	int num_threads = vx_num_threads();
	vx_spawn_warps(num_warps, num_threads, kernel_body, arg);
}