#include <stdint.h>
#include <vx_intrinsics.h>
#include <vx_print.h>
#include <vx_spawn.h>
#include "common.h"

void kernel_body(int task_id, kernel_arg_t* __UNIFORM__ arg) {
	int cid = vx_core_id();
	int* src_ptr = (int*)arg->src_addr;
	char value = 'A' + src_ptr[task_id];
	vx_printf("cid=%d: task=%d, value=%c\n", cid, task_id, value);
}

int main() {
	kernel_arg_t* arg = (kernel_arg_t*)KERNEL_ARG_DEV_MEM_ADDR;
	vx_spawn_tasks(arg->num_points, (vx_spawn_tasks_cb)kernel_body, arg);
	return 0;
}
