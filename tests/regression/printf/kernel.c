#include <stdint.h>
#include <vx_intrinsics.h>
#include <vx_print.h>
#include <vx_spawn.h>
#include "common.h"

void kernel_body(int task_id, const kernel_arg_t* arg) {
	int* src_ptr = (int*)arg->src_ptr;
	vx_printf("task=%d, value=%d\n", task_id, src_ptr[task_id]);
}

void main() {
	const kernel_arg_t* arg = (const kernel_arg_t*)KERNEL_ARG_DEV_MEM_ADDR;
	vx_spawn_tasks(arg->num_points, kernel_body, arg);
}