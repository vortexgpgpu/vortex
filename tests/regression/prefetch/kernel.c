#include <stdint.h>
#include <vx_intrinsics.h>
#include <vx_spawn.h>
#include "common.h"

void kernel_body(int task_id, const kernel_arg_t* arg) {
	uint32_t count    = arg->task_size;
	int32_t* src0_ptr = (int32_t*)arg->src0_ptr;
	int32_t* src1_ptr = (int32_t*)arg->src1_ptr;
	int32_t* dst_ptr  = (int32_t*)arg->dst_ptr;
	
	uint32_t offset = task_id * count;

	for (uint32_t i = 0; i < count; ++i) {
		dst_ptr[offset+i] = src0_ptr[offset+i] + src1_ptr[offset+i];
		vx_prefetch((uint32_t)(src0_ptr) + offset + i);
     	vx_prefetch((uint32_t)(src1_ptr) + offset + i);
	}
}

void main() {
	const kernel_arg_t* arg = (const kernel_arg_t*)KERNEL_ARG_DEV_MEM_ADDR;
	vx_spawn_tasks(arg->num_tasks, kernel_body, arg);
}