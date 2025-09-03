#include <vx_spawn.h>
#include "common.h"

void kernel_body(kernel_arg_t* __UNIFORM__ arg) {
	auto A = reinterpret_cast<TYPE*>(arg->A_addr);
	auto B = reinterpret_cast<TYPE*>(arg->B_addr);
	auto C = reinterpret_cast<TYPE*>(arg->C_addr);
    auto size = arg->size;

    int col = blockIdx.x;
    int row = blockIdx.y;
    int idx = row * size + col;

    int wid = vx_warp_id();
    bool is_compute_warp = ((wid & 1) == 0);

    TYPE sum(0);

    if (is_compute_warp) {
        //vx_wsched(1, 0); // set warp to high priority
        auto a = A[idx];
        auto b = B[idx];
        for (int e = 0; e < size; ++e) {
            sum += a * b;
        }
    } else {
        for (int e = 0; e < size; ++e) {
            const int PDIST = 4;
            TYPE a_buf[PDIST];
            TYPE b_buf[PDIST];
            int issued = 0;
            #pragma unroll
            for (; issued < PDIST && e < size; ++issued, ++e) {
                int a_idx = row * size + e;
                int b_idx = e * size + col;
                a_buf[issued] = A[a_idx];
                b_buf[issued] = B[b_idx];
            }
            //vx_wsched(0, 31); // set warp to 31 cycles yield
            #pragma unroll
            for (int i = 0; i < issued; ++i) {
                sum += a_buf[i] * b_buf[i];
            }
        }
    }
    C[idx] = sum;
}

int main() {
	kernel_arg_t* arg = (kernel_arg_t*)csr_read(VX_CSR_MSCRATCH);
	return vx_spawn_threads(2, arg->grid_dim, nullptr, (vx_kernel_func_cb)kernel_body, arg);
}
