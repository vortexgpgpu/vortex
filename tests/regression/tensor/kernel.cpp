#include <stdint.h>
#include <vx_intrinsics.h>
#include <vx_spawn.h>
#include "common.h"

inline char is_log2(uint32_t x) {
    return ((x & (x-1)) == 0);
}

inline uint32_t log2_fast(uint32_t x) { 
    return 31 - __builtin_clz (x);
}

void kernel_body(uint32_t task_id, kernel_arg_t* __UNIFORM__ arg) {
	auto A = reinterpret_cast<TYPE*>(arg->A_addr);
	auto B = reinterpret_cast<TYPE*>(arg->B_addr);
	auto C = reinterpret_cast<TYPE*>(arg->C_addr);
    auto size  = arg->size;

    uint32_t row, col;
    if (is_log2(size)) {
        uint32_t log_size = log2_fast(size);
        row = task_id >> log_size;
        col = task_id & (size-1);
    } else {
        row = task_id / size;
        col = task_id % size;
    }

    TYPE sum (0);
    for (int e = 0; e < size; ++e) {
        sum += A[row * size + e] * B[e * size + col];
    }
    C[row * size + col] = sum;
}

int main() {
	kernel_arg_t* arg = (kernel_arg_t*)KERNEL_ARG_DEV_MEM_ADDR;
	vx_spawn_tasks(arg->num_tasks, (vx_spawn_tasks_cb)kernel_body, arg);
	return 0;
}
