#include <vx_spawn2.h>
#include "common.h"

// Derived compile-time constants
#define BLOCK_SIZE_M  (BLOCK_DIM_Y * THREAD_SIZE_Y)
#define BLOCK_SIZE_N  (BLOCK_DIM_X * THREAD_SIZE_X)

__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
    const TYPE* A = reinterpret_cast<const TYPE*>(arg->A_addr);
    const TYPE* B = reinterpret_cast<const TYPE*>(arg->B_addr);
    TYPE*       C = reinterpret_cast<TYPE*>(arg->C_addr);
    int N      = (int)(uint32_t)arg->N;
    int K      = (int)(uint32_t)arg->K;
    TYPE alpha = arg->alpha;
    TYPE beta  = arg->beta;

    const int tx = (int)(uint32_t)threadIdx.x;
    const int ty = (int)(uint32_t)threadIdx.y;
    const int bx = (int)(uint32_t)blockIdx.x;
    const int by = (int)(uint32_t)blockIdx.y;

    const int row = by * BLOCK_SIZE_M + ty * THREAD_SIZE_Y;
    const int col = bx * BLOCK_SIZE_N + tx * THREAD_SIZE_X;

    // Shared memory: As[BLOCK_SIZE_M][BLOCK_SIZE_K], Bs[BLOCK_SIZE_K][BLOCK_SIZE_N]
    TYPE* smem = reinterpret_cast<TYPE*>(__local_mem());
    TYPE* As   = smem;
    TYPE* Bs   = smem + BLOCK_SIZE_M * BLOCK_SIZE_K;

    TYPE accum[THREAD_SIZE_Y][THREAD_SIZE_X] = {};
    TYPE a_frag[THREAD_SIZE_Y];
    TYPE b_frag[THREAD_SIZE_X];

    for (int k_tile = 0; k_tile < (K + BLOCK_SIZE_K - 1) / BLOCK_SIZE_K; ++k_tile) {

        // --- MEMORY PHASE: Global to Shared ---

        // Load A into shared memory
        #pragma unroll
        for (int i = 0; i < THREAD_SIZE_Y; ++i) {
            int a_row = by * BLOCK_SIZE_M + ty * THREAD_SIZE_Y + i;
            int a_col = k_tile * BLOCK_SIZE_K + tx;
            As[(ty * THREAD_SIZE_Y + i) * BLOCK_SIZE_K + tx] = A[a_row * K + a_col];
        }

        // Load B into shared memory
        #pragma unroll
        for (int j = 0; j < THREAD_SIZE_X; ++j) {
            int b_row = k_tile * BLOCK_SIZE_K + ty;
            int b_col = bx * BLOCK_SIZE_N + tx * THREAD_SIZE_X + j;
            Bs[ty * BLOCK_SIZE_N + tx * THREAD_SIZE_X + j] = B[b_row * N + b_col];
        }

        __syncthreads();

        // --- COMPUTE PHASE: Shared to Registers to ALUs ---
        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE_K; ++k) {
            #pragma unroll
            for (int i = 0; i < THREAD_SIZE_Y; ++i) {
                a_frag[i] = As[(ty * THREAD_SIZE_Y + i) * BLOCK_SIZE_K + k];
            }
            #pragma unroll
            for (int j = 0; j < THREAD_SIZE_X; ++j) {
                b_frag[j] = Bs[k * BLOCK_SIZE_N + tx * THREAD_SIZE_X + j];
            }

            #pragma unroll
            for (int i = 0; i < THREAD_SIZE_Y; ++i) {
                #pragma unroll
                for (int j = 0; j < THREAD_SIZE_X; ++j) {
                    accum[i][j] += a_frag[i] * b_frag[j];
                }
            }
        }

        __syncthreads();
    }

    // --- Write Back to Global Memory ---
    #pragma unroll
    for (int i = 0; i < THREAD_SIZE_Y; ++i) {
        #pragma unroll
        for (int j = 0; j < THREAD_SIZE_X; ++j) {
            int c_row = row + i;
            int c_col = col + j;
            C[c_row * N + c_col] = alpha * accum[i][j] + beta * C[c_row * N + c_col];
        }
    }
}
