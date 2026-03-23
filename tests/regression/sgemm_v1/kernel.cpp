#include <vx_spawn.h>
#include "common.h"

void kernel_body(kernel_arg_t* __UNIFORM__ arg) {
	auto A = reinterpret_cast<TYPE*>(arg->A_addr);
	auto B = reinterpret_cast<TYPE*>(arg->B_addr);
	auto C = reinterpret_cast<TYPE*>(arg->C_addr);
    auto size = arg->size;

    int col = blockIdx.x;
    int row = blockIdx.y;

    TYPE sum(0);
    for (int e = 0; e < size; ++e) {
        sum += A[row * size + e] * B[e * size + col];
    }

    C[row * size + col] = sum;
}

int main() {
	kernel_arg_t* arg = (kernel_arg_t*)csr_read(VX_CSR_MSCRATCH);
	return vx_spawn_threads(2, arg->grid_dim, nullptr, (vx_kernel_func_cb)kernel_body, arg);
}
