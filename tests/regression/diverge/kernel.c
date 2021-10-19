#include <stdint.h>
#include <vx_intrinsics.h>
#include <vx_spawn.h>
#include "common.h"

// Parallel Selection sort

void kernel_body(int task_id, kernel_arg_t* arg) {
	int32_t* src_ptr = (int32_t*)arg->src_ptr;
	int32_t* dst_ptr = (int32_t*)arg->dst_ptr;

	int value = src_ptr[task_id];

	// none taken
	__if (task_id >= 0x7fffffff) {
		value = 0;
	}__else {
		value += 2;
	}__endif	
	
	// diverge
	__if (task_id > 1) {
		__if (task_id > 2) {
			value += 6;
		}__else {
			value += 5;
		}__endif
	}__else {
		__if (task_id > 0) {
			value += 4;
		}__else {
			value += 3;
		}__endif
	}__endif

	// all taken
	__if (task_id >= 0) {
		value += 7;
	}__else {
		value = 0;
	}__endif

	dst_ptr[task_id] = value;
}

void main() {
	kernel_arg_t* arg = (kernel_arg_t*)KERNEL_ARG_DEV_MEM_ADDR;
	vx_spawn_tasks(arg->num_points, (vx_spawn_tasks_cb)kernel_body, arg);
}