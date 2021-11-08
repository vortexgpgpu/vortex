#include <stdint.h>
#include <vx_intrinsics.h>
#include <vx_spawn.h>
#include "common.h"

void kernel_body(int __DIVERGENT__ task_id, kernel_arg_t* arg) {
	uint32_t num_points = arg->num_points;
	int32_t* src_ptr = (int32_t*)arg->src_ptr;
	int32_t* dst_ptr = (int32_t*)arg->dst_ptr;

	int32_t ref_value = src_ptr[task_id];

	uint32_t pos = 0;
	for (uint32_t i = 0; i < num_points; ++i) {
		int32_t cur_value = src_ptr[i];		
		pos += (cur_value < ref_value) || ((cur_value == ref_value) && (i < task_id));
	}
	dst_ptr[pos] = ref_value;
}

void main() {
	kernel_arg_t* arg = (kernel_arg_t*)KERNEL_ARG_DEV_MEM_ADDR;
	vx_spawn_tasks(arg->num_points, (vx_spawn_tasks_cb)kernel_body, arg);
}