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

	unsigned wNo = vx_warpNum();
	unsigned tid = vx_threadID();

	unsigned i = ((wNo * MAX_THREADS) + tid) * _arg->stride;

	for (unsigned j = 0; j < _arg->stride; ++j) {
		z[i+j] = x[i+j] * y[i+j];
	}
}

void main() {
	struct kernel_arg_t* arg = (struct kernel_arg_t*)KERNEL_ARG_DEV_MEM_ADDR;
	vx_spawnWarps(MAX_WARPS, MAX_THREADS, kernel_body, arg);
}