#include <vx_spawn2.h>
#include "common.h"

__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
	auto A = reinterpret_cast<TYPE*>(arg->A_addr);
	auto B = reinterpret_cast<TYPE*>(arg->B_addr);
	auto C = reinterpret_cast<TYPE*>(arg->C_addr);
    size_t size = arg->size;

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    TYPE sum(0);
#ifdef VX_CFG_EXT_LDST_WIDE_ENABLE
    static_assert(__is_same(TYPE, float), "the wide loads read float words");
    const TYPE* a_row = A + row * size;
    const TYPE* b_col = B + col;
    size_t e = 0;
    for (; e + 8 <= size; e += 8) {
        vx_float4 a0 = vx_flw_q<24>(a_row + e);
        vx_float4 a1 = vx_flw_q<28, 16>(a_row + e);
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            sum += a0[k] * *b_col;
            b_col += size;
        }
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            sum += a1[k] * *b_col;
            b_col += size;
        }
    }
    for (; e < size; ++e) {
        sum += a_row[e] * *b_col;
        b_col += size;
    }
#else
    #pragma unroll 8
    for (size_t e = 0; e < size; ++e) {
        sum += A[row * size + e] * B[e * size + col];
    }
#endif

    C[row * size + col] = sum;
}
