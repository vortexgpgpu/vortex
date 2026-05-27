#include <vx_spawn.h>
#include <vx_intrinsics.h>
#include "common.h"

void kernel_body(kernel_arg_t* __UNIFORM__ arg) {
    auto I = reinterpret_cast<TYPE*>(arg->I_addr);
    auto W = reinterpret_cast<TYPE*>(arg->use_lmem ? __local_mem(0) : (void*)arg->W_addr);
	auto O = reinterpret_cast<TYPE*>(arg->O_addr);
    auto width = arg->width;

    int col = blockIdx.x;
    int row = blockIdx.y;

    // Adjust for padded borders
    int paddedWidth = width + 2;
    int paddedX = col + 1;
    int paddedY = row + 1;

    // Compute 3x3 convolution sum
    float sum = 0.0f;

    sum += I[(paddedY - 1) * paddedWidth + (paddedX - 1)] * W[0]; // Top-left
    sum += I[(paddedY - 1) * paddedWidth + paddedX] * W[1];       // Top-center
    sum += I[(paddedY - 1) * paddedWidth + (paddedX + 1)] * W[2]; // Top-right

    sum += I[paddedY * paddedWidth + (paddedX - 1)] * W[3];       // Middle-left
    sum += I[paddedY * paddedWidth + paddedX] * W[4];             // Center
    sum += I[paddedY * paddedWidth + (paddedX + 1)] * W[5];       // Middle-right

    sum += I[(paddedY + 1) * paddedWidth + (paddedX - 1)] * W[6]; // Bottom-left
    sum += I[(paddedY + 1) * paddedWidth + paddedX] * W[7];       // Bottom-center
    sum += I[(paddedY + 1) * paddedWidth + (paddedX + 1)] * W[8]; // Bottom-right

    O[row * width + col] = sum;
}

int main() {
    kernel_arg_t* arg = (kernel_arg_t*)csr_read(VX_CSR_MSCRATCH);
    if (arg->enable_dfv_test) {
        csr_write(VX_CSR_DFV_CTRL, 1);
        csr_write(VX_CSR_DFV_RANDOM_SEED, 0xABCDEF00);
        csr_write(VX_CSR_DFV_SET_THRESHOLD, 240);
        csr_write(VX_CSR_DFV_RELEASE_THRESHOLD, 65518);  // release when lfsr2[15:0] >= 65504 (~0.04%)
        csr_write(VX_CSR_DFV_RELEASE_DELAY, 0);
        csr_write(VX_CSR_DFV_RELEASE_FOREVER, 1);
        csr_write(VX_CSR_DFV_THROTTLE_THRESHOLD, 6144);
        csr_write(VX_CSR_DFV_ICACHE_STALL, 0);
        csr_write(VX_CSR_DFV_DCACHE_STALL, 0);
        csr_write(VX_CSR_DFV_WRITEBACK_STALL, 0);
        csr_write(VX_CSR_DFV_FILL_STALL, 1);
    }
    if (arg->use_lmem) {
        // populate local memory
        auto W = reinterpret_cast<TYPE*>(arg->W_addr);
        auto L = reinterpret_cast<TYPE*>(__local_mem(0));
        for (int i = 0; i < (3*3); ++i) {
            L[i] = W[i];
        }
    }
    int __ret = vx_spawn_threads(2, arg->grid_dim, nullptr, (vx_kernel_func_cb)kernel_body, arg);
    if (arg->enable_dfv_test) {
        csr_write(VX_CSR_DFV_CTRL, 0);
    }
    return __ret;
}
