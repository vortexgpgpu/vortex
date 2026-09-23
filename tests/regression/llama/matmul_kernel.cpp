#include <vx_spawn.h>
#include "common.h"

void matmul_kernel_(matmul_kernel_args_t* __UNIFORM__ args) {
    auto A = reinterpret_cast<float*>(args->A_addr);
    auto B = reinterpret_cast<float*>(args->B_addr);
    auto C = reinterpret_cast<float*>(args->C_addr);

    int M = args->M;
    int N = args->N;
    int K = args->K;

    // 2D indexing - each thread computes one output element
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

int main() {
    auto arg = (matmul_kernel_args_t*)csr_read(VX_CSR_MSCRATCH);
    return vx_spawn_threads(2, arg->grid_dim, arg->block_dim, (vx_kernel_func_cb)matmul_kernel_, arg);
}