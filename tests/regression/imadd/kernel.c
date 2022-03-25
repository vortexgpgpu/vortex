#include <stdint.h>
#include <vx_intrinsics.h>
#include <vx_print.h>
#include <vx_spawn.h>
#include "common.h"

void kernel_body(int task_id, kernel_arg_t* arg) {
	uint32_t count    = arg->task_size;
	int32_t* src0_ptr = (int32_t*)arg->src0_addr;
	int32_t* src1_ptr = (int32_t*)arg->src1_addr;
	int32_t* src2_ptr = (int32_t*)arg->src2_addr;
	int32_t* src3_ptr = (int32_t*)arg->src3_addr;
	int32_t* dst_ptr  = (int32_t*)arg->dst_addr;	
	uint32_t offset = task_id * count;

	for (uint32_t i = 0; i < count; ++i) {
		int32_t a = src0_ptr[offset+i];
		int32_t b = src1_ptr[offset+i];
		int32_t c = src2_ptr[offset+i];
		int32_t s = src3_ptr[offset+i];
		int32_t d = vx_imadd(a, b, c, s);
		dst_ptr[offset+i] = d;
	}
}

void main() {
	kernel_arg_t* arg = (kernel_arg_t*)KERNEL_ARG_DEV_MEM_ADDR;
	vx_spawn_tasks(arg->num_tasks, (vx_spawn_tasks_cb)kernel_body, arg);
}