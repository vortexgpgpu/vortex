#include <stdint.h>
#include <assert.h>
#include <algorithm>
#include <vx_intrinsics.h>
#include <vx_spawn.h>
#include "common.h"

// Parallel Selection sort

void kernel_body(int task_id, kernel_arg_t* __UNIFORM__ arg) {
	int32_t* src_ptr = (int32_t*)arg->src_addr;
	int32_t* dst_ptr = (int32_t*)arg->dst_addr;

	int value = src_ptr[task_id];

	// none taken
	if (task_id >= 0x7fffffff) {
		value = 0;
	} else {
		value += 2;
	}	
	
	// diverge
	if (task_id > 1) {
		if (task_id > 2) {
			value += 6;
		} else {
			value += 5;
		}
	} else {
		if (task_id > 0) {
			value += 4;
		} else {
			value += 3;
		}
	}

	// all taken
	if (task_id >= 0) {
		value += 7;
	} else {
		value = 0;
	}

	// loop
	for (int i = 0, n = task_id; i < n; ++i) {
		value += src_ptr[i];
	}

	// switch
	switch (task_id) {
	case 0:
		value += 1;
		break;
	case 1:
		value -= 1;
		break;
	case 2:
		value *= 3;
		break;
	case 3:
		value *= 5;
		break;
	default:
		assert(task_id < arg->num_points);
		break;
	}

	// select
	value += (task_id >= 0) ? ((task_id > 5) ? src_ptr[0] : task_id) : ((task_id < 5) ? src_ptr[1] : -task_id);

	// min/max
	value += std::min(src_ptr[task_id], value);
	value += std::max(src_ptr[task_id], value);

	dst_ptr[task_id] = value;
}

int main() {
	kernel_arg_t* arg = (kernel_arg_t*)KERNEL_ARG_DEV_MEM_ADDR;
	vx_spawn_tasks(arg->num_points, (vx_spawn_tasks_cb)kernel_body, arg);
	return 0;
}
