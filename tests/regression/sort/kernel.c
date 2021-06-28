#include <stdint.h>
#include <vx_intrinsics.h>
#include <vx_spawn.h>
#include <vx_print.h>
#include "common.h"

// Parallel Selection sort

int __attribute__((noinline)) __smaller(int index, int tid, int32_t cur_value, int32_t ref_value) {
	int ret = 0;
	__if (cur_value < ref_value) {
		ret = 1;
	} __else {
		__if (cur_value == ref_value) {
			__if (index < tid) {
				ret = 1;
			} __endif
		} __endif
	} __endif
	return ret;
}

void kernel_body(int task_id, void* arg) {
	struct kernel_arg_t* _arg = (struct kernel_arg_t*)(arg);
	uint32_t num_points = _arg->num_points;
	int32_t* src_ptr = (int32_t*)_arg->src_ptr;
	int32_t* dst_ptr = (int32_t*)_arg->dst_ptr;

	int32_t ref_value = src_ptr[task_id];

	uint32_t pos = 0;
	for (uint32_t i = 0; i < num_points; ++i) {
		int32_t cur_value = src_ptr[i];		
		pos += __smaller(i, task_id, cur_value, ref_value);
	}
	dst_ptr[pos] = ref_value;
	vx_printf("taskid=%d, pos=%d, value=%d\n", task_id, pos, ref_value);
}

void main() {
	struct kernel_arg_t* arg = (struct kernel_arg_t*)KERNEL_ARG_DEV_MEM_ADDR;
	vx_spawn_tasks(arg->num_points, kernel_body, arg);
}