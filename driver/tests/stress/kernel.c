#include <stdint.h>
#include <vx_intrinsics.h>
#include <vx_spawn.h>
#include "common.h"

void kernel_body(int task_id, void* arg) {
	struct kernel_arg_t* _arg = (struct kernel_arg_t*)(arg);	
	uint32_t stride    = _arg->stride;
	int32_t* src0_ptr  = (int32_t*)_arg->src0_ptr;
	uint32_t* src1_ptr = (uint32_t*)_arg->src1_ptr;
	int32_t* dst_ptr   = (int32_t*)_arg->dst_ptr;
	
	uint32_t offset = task_id * stride;

	for (uint32_t i = 0; i < stride; ++i) {
		int value = 0;
		for (uint32_t j = 0; j < NUM_LOADS; ++j) {
			uint32_t addr  = offset + i + j;
			uint32_t index = src1_ptr[addr];
			value += src0_ptr[index];
		}
		dst_ptr[offset+i] = value;
	}
}

void main() {
	struct kernel_arg_t* arg = (struct kernel_arg_t*)KERNEL_ARG_DEV_MEM_ADDR;
	vx_spawn_tasks(arg->num_tasks, kernel_body, arg);
}