#include <vx_spawn2.h>
#include "common.h"

__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
	auto A = reinterpret_cast<TYPE*>(arg->A_addr);
	auto B = reinterpret_cast<TYPE*>(arg->B_addr);
	auto C = reinterpret_cast<TYPE*>(arg->C_addr);
    auto size = arg->size;

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    TYPE sum(0);
    for (int e = 0; e < size; ++e) {
        sum += A[row * size + e] * B[e * size + col];
    }

    C[row * size + col] = sum;
}
