#include <stdint.h>
#include <vx_intrinsics.h>
#include <vx_spawn.h>
#include "common.h"

void kernel_body(int task_id, void* arg) {
	struct kernel_arg_t* _arg = (struct kernel_arg_t*)(arg);
	uint32_t count    = _arg->task_size;
	int32_t* src0_ptr = (int32_t*)_arg->src0_ptr;
	int32_t* src1_ptr = (int32_t*)_arg->src1_ptr;
	int32_t* dst_ptr  = (int32_t*)_arg->dst_ptr;
	
	uint32_t offset = task_id * count;

	for (uint32_t i = 0; i < count; ++i) {
		dst_ptr[offset+i] = src0_ptr[offset+i] + src1_ptr[offset+i];
	}

	vx_fence();
}

void main() {
	struct kernel_arg_t* arg = (struct kernel_arg_t*)KERNEL_ARG_DEV_MEM_ADDR;
	vx_spawn_tasks(arg->num_tasks, kernel_body, arg);
}