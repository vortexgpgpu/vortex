#include <vx_spawn.h>
#include "common.h"

void kernel_body(kernel_arg_t* __UNIFORM__ arg) {
	auto A = reinterpret_cast<TYPE*>(arg->A_addr);
	auto B = reinterpret_cast<TYPE*>(arg->B_addr);
	auto C = reinterpret_cast<TYPE*>(arg->C_addr);
    size_t size = arg->size;

    int col = blockIdx.x;
    int row = blockIdx.y;

    TYPE sum(0);
    const TYPE* a_row = A + row * size;
    size_t e = 0;
    for (; e + 4 <= size; e += 4) {
#if LDST_WIDE
        static_assert(__is_same(TYPE, float), "LDST_WIDE needs float");
        vx_float4 a = vx_flw_q(a_row + e);
#else
        TYPE a[4] = {a_row[e], a_row[e + 1], a_row[e + 2], a_row[e + 3]};
#endif
        sum += a[0] * B[e * size + col];
        sum += a[1] * B[(e + 1) * size + col];
        sum += a[2] * B[(e + 2) * size + col];
        sum += a[3] * B[(e + 3) * size + col];
    }
    for (; e < size; ++e) {
        sum += a_row[e] * B[e * size + col];
    }

    C[row * size + col] = sum;
}

int main() {
	kernel_arg_t* arg = (kernel_arg_t*)csr_read(VX_CSR_MSCRATCH);
	return vx_spawn_threads(2, arg->grid_dim, nullptr, (vx_kernel_func_cb)kernel_body, arg);
}
