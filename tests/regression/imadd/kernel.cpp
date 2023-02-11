#include <stdint.h>
#include <vx_intrinsics.h>
#include <vx_print.h>
#include <vx_spawn.h>
#include "common.h"

inline int32_t vx_imadd_sw(int32_t a, int32_t b, int32_t c, int32_t s) {
	int64_t x = a;
	int64_t y = b;
	int64_t p = (x * y) >> (s << 3);
	int32_t r = p;
	return r + c;
}

void kernel_body(int task_id, kernel_arg_t* __UNIFORM__ arg) {
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
		int32_t d;
		if (arg->use_sw) {
			switch (s) {
			case 0:  d = vx_imadd_sw(a, b, c, 0); break;
			case 1:  d = vx_imadd_sw(a, b, c, 1); break;
			case 2:  d = vx_imadd_sw(a, b, c, 2); break;
			case 3:  d = vx_imadd_sw(a, b, c, 3); break;
			default: d = 0; break;
			}
		} else {
			switch (s) {
			case 0:  d = vx_imadd(a, b, c, 0); break;
			case 1:  d = vx_imadd(a, b, c, 1); break;
			case 2:  d = vx_imadd(a, b, c, 2); break;
			case 3:  d = vx_imadd(a, b, c, 3); break;
			default: d = 0; break;
			}
		}
		dst_ptr[offset+i] = d;
	}
}

int main() {
	kernel_arg_t* arg = (kernel_arg_t*)KERNEL_ARG_DEV_MEM_ADDR;
	vx_spawn_tasks(arg->num_tasks, (vx_spawn_tasks_cb)kernel_body, arg);
	return 0;
}
