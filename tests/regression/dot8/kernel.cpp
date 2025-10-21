#include <vx_spawn.h>
#include "common.h"

void kernel_body(kernel_arg_t* __UNIFORM__ arg) {
	auto A = reinterpret_cast<int8_t*>(arg->A_addr);
	auto B = reinterpret_cast<int8_t*>(arg->B_addr);
	auto C = reinterpret_cast<int32_t*>(arg->C_addr);
    auto size = arg->size;

    int col = blockIdx.x;
    int row = blockIdx.y;

    int32_t sum(0);
    // for (int e = 0; e < size; ++e) {
    //     sum += A[row * size + e] * B[e * size + col]; // int32
    // }
    for (int k = 0; k < size; k += 4) {
        // Pack 4 int8 from A row (contiguous)
        uint32_t packedA = *reinterpret_cast<const uint32_t*>(&A[row * size + k]);

        // Pack 4 int8 from B column (strided)
        uint32_t packedB =
            (uint32_t)(uint8_t)B[(k+0)*size + col]        |
            (uint32_t)(uint8_t)B[(k+1)*size + col] << 8   |
            (uint32_t)(uint8_t)B[(k+2)*size + col] << 16  |
            (uint32_t)(uint8_t)B[(k+3)*size + col] << 24;

        sum += vx_dot8(packedA, packedB);
    }

    C[row * size + col] = sum;
}

int main() {
	kernel_arg_t* arg = (kernel_arg_t*)csr_read(VX_CSR_MSCRATCH);
	return vx_spawn_threads(2, arg->grid_dim, nullptr, (vx_kernel_func_cb)kernel_body, arg);
}
