#include <stdint.h>
#include <vx_intrinsics.h>
#include <vx_spawn.h>
#include "common.h"

void kernel_body(int task_id, kernel_arg_t* __UNIFORM__ arg) {
	auto src0_ptr = reinterpret_cast<TYPE*>(arg->src0_addr);
	auto src1_ptr = reinterpret_cast<TYPE*>(arg->src1_addr);
	auto dst_ptr = reinterpret_cast<TYPE*>(arg->dst_addr);
	
	uint32_t count = arg->task_size;
	uint32_t offset = task_id * count;

	for (uint32_t i = 0; i < count; ++i) {
		dst_ptr[offset+i] = src0_ptr[offset+i] + src1_ptr[offset+i];
	}
}

int main() {
	kernel_arg_t* arg = (kernel_arg_t*)KERNEL_ARG_DEV_MEM_ADDR;
	vx_spawn_tasks(arg->num_tasks, (vx_spawn_tasks_cb)kernel_body, arg);
	return 0;
}
