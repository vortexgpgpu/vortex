#include <stdint.h>
#include <vx_intrinsics.h>
#include <vx_spawn.h>
#include "common.h"

static uint32_t rotr(uint32_t, int);

void kernel_body(int task_id, const void* arg) {
	struct kernel_arg_t* _arg = (struct kernel_arg_t*)(arg);
	uint32_t count    = _arg->task_size;
	int32_t* src0_ptr = (int32_t*)_arg->src0_ptr;
	int32_t* src1_ptr = (int32_t*)_arg->src1_ptr;
	int32_t* dst_ptr  = (int32_t*)_arg->dst_ptr;
	
	uint32_t offset = task_id * count;

	for (uint32_t i = 0; i < count; ++i) {
		int32_t temp = rotr(src0_ptr[offset+i], 0) + src1_ptr[offset+i];
		dst_ptr[offset+i] = rotr(temp, 0);
	}
}

#define ROTR
static inline uint32_t rotr(uint32_t x, int n) {
    #ifdef ROTR
    return __intrin_rotr_imm(x, n);
    #else
    return (x >> n) | (x << (32 - n));
    #endif
}

void main() {
	struct kernel_arg_t* arg = (struct kernel_arg_t*)KERNEL_ARG_DEV_MEM_ADDR;
	vx_spawn_tasks(arg->num_tasks, kernel_body, arg);
}
