#include <stdint.h>
#include <vx_intrinsics.h>
#include <vx_spawn.h>
#include "common.h"

typedef void (*PFN_Kernel)(int task_id, kernel_arg_t* arg);

void kernel_mind(int __DIVERGENT__ task_id, kernel_arg_t* arg) {
	int32_t* cost_ptr = (int32_t*)arg->gcost_addr;
	uint32_t num_points = arg->no_of_nodes;
	for (uint32_t i = 0; i < num_points; ++i) {
		cost_ptr[i] = cost_ptr[i]+8;
	}
}

void kernel_body(int __DIVERGENT__ task_id, kernel_arg_t* arg) {
	int32_t* cost_ptr = (int32_t*)arg->gcost_addr;
	(int32_t*) hover = (int32_t*)arg->hover_addr;
	uint32_t num_points = arg->no_of_nodes;
	for (uint32_t i = 0; i < num_points; ++i) {
		cost_ptr[i] = cost_ptr[i]+7;
		if(cost_ptr[i]<20)
			hover[0] = 1;
		else
			hover[0] = 0;
	}
}

static const PFN_Kernel sc_tests[] = {
	kernel_mind,
	kernel_body,
};

void main() {
	kernel_arg_t* arg = (kernel_arg_t*)KERNEL_ARG_DEV_MEM_ADDR;
	vx_spawn_tasks(arg->no_of_nodes, (vx_spawn_tasks_cb)sc_tests[arg->testid], arg);
}
