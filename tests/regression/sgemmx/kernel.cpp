#include <vx_spawn2.h>
#include "common.h"

// Derived compile-time constants
#define BLOCK_SIZE_M  (BLOCK_DIM_Y * THREAD_SIZE_Y)
#define BLOCK_SIZE_N  (BLOCK_DIM_X * THREAD_SIZE_X)

#ifdef VX_CFG_EXT_LDST_WIDE_ENABLE
static_assert(__is_same(TYPE, float), "the wide loads and stores move float words");

// N consecutive words, in groups of 4, then 2, then 1, at constant offsets;
// consecutive quads take different register groups.
template <int N, int J = 0>
static inline void wide_load(TYPE* dst, const TYPE* src) {
    if constexpr (J + 4 <= N) {
        vx_float4 v = vx_flw_q<((J / 4) % 2) ? 28 : 24, J * int32_t(sizeof(TYPE))>(src);
        dst[J] = v[0];
        dst[J + 1] = v[1];
        dst[J + 2] = v[2];
        dst[J + 3] = v[3];
        wide_load<N, J + 4>(dst, src);
    } else if constexpr (J + 2 <= N) {
        vx_float2 v = vx_flw_d<22, J * int32_t(sizeof(TYPE))>(src);
        dst[J] = v[0];
        dst[J + 1] = v[1];
        wide_load<N, J + 2>(dst, src);
    } else if constexpr (J < N) {
        dst[J] = src[J];
    }
}

template <int N, int J = 0>
static inline void wide_store(TYPE* dst, const TYPE* src) {
    if constexpr (J + 4 <= N) {
        vx_fsw_q<((J / 4) % 2) ? 28 : 24, J * int32_t(sizeof(TYPE))>(dst, vx_float4{src[J], src[J + 1], src[J + 2], src[J + 3]});
        wide_store<N, J + 4>(dst, src);
    } else if constexpr (J + 2 <= N) {
        vx_fsw_d<22, J * int32_t(sizeof(TYPE))>(dst, vx_float2{src[J], src[J + 1]});
        wide_store<N, J + 2>(dst, src);
    } else if constexpr (J < N) {
        dst[J] = src[J];
    }
}
#endif

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
    #ifdef VX_CFG_EXT_LDST_WIDE_ENABLE
        {
            TYPE b_row_data[THREAD_SIZE_X];
            int b_row = k_tile * BLOCK_SIZE_K + ty;
            wide_load<THREAD_SIZE_X>(b_row_data, &B[b_row * N + bx * BLOCK_SIZE_N + tx * THREAD_SIZE_X]);
            wide_store<THREAD_SIZE_X>(&Bs[ty * BLOCK_SIZE_N + tx * THREAD_SIZE_X], b_row_data);
        }
    #else
        #pragma unroll
        for (int j = 0; j < THREAD_SIZE_X; ++j) {
            int b_row = k_tile * BLOCK_SIZE_K + ty;
            int b_col = bx * BLOCK_SIZE_N + tx * THREAD_SIZE_X + j;
            Bs[ty * BLOCK_SIZE_N + tx * THREAD_SIZE_X + j] = B[b_row * N + b_col];
        }
    #endif

        __syncthreads();

        // --- COMPUTE PHASE: Shared to Registers to ALUs ---
        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE_K; ++k) {
            #pragma unroll
            for (int i = 0; i < THREAD_SIZE_Y; ++i) {
                a_frag[i] = As[(ty * THREAD_SIZE_Y + i) * BLOCK_SIZE_K + k];
            }
        #ifdef VX_CFG_EXT_LDST_WIDE_ENABLE
            wide_load<THREAD_SIZE_X>(b_frag, &Bs[k * BLOCK_SIZE_N + tx * THREAD_SIZE_X]);
        #else
            #pragma unroll
            for (int j = 0; j < THREAD_SIZE_X; ++j) {
                b_frag[j] = Bs[k * BLOCK_SIZE_N + tx * THREAD_SIZE_X + j];
            }
        #endif

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
    #ifdef VX_CFG_EXT_LDST_WIDE_ENABLE
        TYPE c_data[THREAD_SIZE_X];
        TYPE* c_ptr = &C[(row + i) * N + col];
        wide_load<THREAD_SIZE_X>(c_data, c_ptr);
        #pragma unroll
        for (int j = 0; j < THREAD_SIZE_X; ++j) {
            c_data[j] = alpha * accum[i][j] + beta * c_data[j];
        }
        wide_store<THREAD_SIZE_X>(c_ptr, c_data);
    #else
        #pragma unroll
        for (int j = 0; j < THREAD_SIZE_X; ++j) {
            int c_row = row + i;
            int c_col = col + j;
            C[c_row * N + c_col] = alpha * accum[i][j] + beta * C[c_row * N + c_col];
        }
    #endif
    }
}
