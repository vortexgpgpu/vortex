#include <vx_spawn.h>
#include "common.h"
#include "vx_print.h"


void kernel_body(kernel_arg_t* __UNIFORM__ arg) {
	auto A     = reinterpret_cast<TYPE*>(arg->src0_addr);
	auto x_old = reinterpret_cast<TYPE*>(arg->src1_addr);
	auto b     = reinterpret_cast<TYPE*>(arg->src2_addr);
	auto x_new = reinterpret_cast<TYPE*>(arg->dst_addr);
    auto n     = arg->size;

    uint32_t i = blockIdx.x;
    uint32_t index = i * n;

    double sum = 0.0;
    for (int j = 0; j < n; j++) {
        if (j != i){
            sum += A[index + j] * x_old[j];
        }
    }

    x_new[i] = (b[i] - sum) / A[index + i];
}

int main() {
	kernel_arg_t* arg = (kernel_arg_t*)csr_read(VX_CSR_MSCRATCH);
    return vx_spawn_threads(1, &arg->size, nullptr, (vx_kernel_func_cb)kernel_body, arg);
}
