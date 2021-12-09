#include <stdint.h>
#include <vx_intrinsics.h>
#include <vx_spawn.h>
#include <vx_print.h>
#include "common.h"

#define BLOCK_SIZE 64

void kernel_body(int task_id, kernel_arg_t* arg) {
	uint32_t count = arg->task_size;
	uint32_t offset = task_id * count;
	uint32_t num_blocks = (count * 4 + BLOCK_SIZE-1) / BLOCK_SIZE;

	int32_t* src0_ptr = (int32_t*)arg->src0_ptr + offset;
	int32_t* src1_ptr = (int32_t*)arg->src1_ptr + offset;
	int32_t* dst_ptr  = (int32_t*)arg->dst_ptr + offset;

	uint32_t src0_end = (uint32_t)(src0_ptr + count);
	uint32_t src1_end = (uint32_t)(src1_ptr + count);

	for (uint32_t i = 0; i < count; ++i) {
		dst_ptr[i] = src0_ptr[i] + src1_ptr[i];

		uint32_t src0_mask = ((uint32_t)(src0_ptr + i)) % BLOCK_SIZE;		
		uint32_t src0_next = (uint32_t)(src0_ptr + i + BLOCK_SIZE/4);
		if (src0_mask == 0 && src0_next < src0_end) {
			//vx_printf("src0_next=%d\n", src0_next);
			vx_prefetch(src0_next);			
		}
		
		uint32_t src1_mask = ((uint32_t)(src1_ptr + i)) % BLOCK_SIZE;
		uint32_t src1_next = (uint32_t)(src1_ptr + i + BLOCK_SIZE/4);
		if (src1_mask == 0 && src1_next < src1_end) {
			//vx_printf("src1_next=%d\n", src1_next);
			vx_prefetch(src1_next);
		}		
	}
}

void main() {
	kernel_arg_t* arg = (kernel_arg_t*)KERNEL_ARG_DEV_MEM_ADDR;
	vx_spawn_tasks(arg->num_tasks, (vx_spawn_tasks_cb)kernel_body, arg);
}