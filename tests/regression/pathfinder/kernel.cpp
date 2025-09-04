#include <vx_spawn.h>
#include "common.h"
#include "vx_print.h"

#define MIN(a, b) ((a) < (b) ? (a) : (b))


void kernel_body(kernel_arg_t* __UNIFORM__ arg) {
	auto wall = reinterpret_cast<TYPE*>(arg->src0_addr);
	auto src  = reinterpret_cast<TYPE*>(arg->src1_addr);
	auto dst  = reinterpret_cast<TYPE*>(arg->dst_addr);
    auto num_cols = arg->num_cols; 
    auto num_rows = arg->num_rows; 

    TYPE min;
    TYPE* temp;
    uint32_t n = blockIdx.x;

    min = src[n];
    if (n > 0)
        min = MIN(min, src[n - 1]);
    if (n < num_cols - 1)
        min = MIN(min, src[n + 1]);

    dst[n] = wall[n] + min;
}

int main() {
	kernel_arg_t* arg = (kernel_arg_t*)csr_read(VX_CSR_MSCRATCH);
    return vx_spawn_threads(1, &arg->num_cols, nullptr, (vx_kernel_func_cb)kernel_body, arg);
}
