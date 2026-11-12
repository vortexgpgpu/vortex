#include <vx_spawn2.h>
#include "common.h"

#ifdef VX_CFG_EXT_LDST_WIDE_ENABLE
// The dot product of a row of As and a column of Bs, reading the row in quads
// at constant offsets; consecutive quads take different register groups.
template <uint32_t BLOCK_SIZE, uint32_t K = 0>
static inline void dot_wide(TYPE& sum, const TYPE* a_row, const TYPE* b_col) {
  if constexpr (K < BLOCK_SIZE) {
    vx_float4 a = vx_flw_q<((K / 4) % 2) ? 28 : 24, int32_t(K * sizeof(TYPE))>(a_row);
    sum += a[0] * b_col[K * BLOCK_SIZE];
    sum += a[1] * b_col[(K + 1) * BLOCK_SIZE];
    sum += a[2] * b_col[(K + 2) * BLOCK_SIZE];
    sum += a[3] * b_col[(K + 3) * BLOCK_SIZE];
    dot_wide<BLOCK_SIZE, K + 4>(sum, a_row, b_col);
  }
}
#endif

// Shared-memory tiled matrix multiply, as in the CUDA samples' matrixMul: each
// thread computes one element of C, and the block steps BLOCK_SIZE x
// BLOCK_SIZE tiles of A and B through local memory.
template <uint32_t BLOCK_SIZE>
static void matrix_mul(const TYPE* A, const TYPE* B, TYPE* C, uint32_t size) {
  auto As = reinterpret_cast<TYPE*>(__local_mem()); // [BLOCK_SIZE][BLOCK_SIZE]
  auto Bs = As + BLOCK_SIZE * BLOCK_SIZE;            // [BLOCK_SIZE][BLOCK_SIZE]

  uint32_t tx = threadIdx.x;
  uint32_t ty = threadIdx.y;
  uint32_t row = blockIdx.y * BLOCK_SIZE + ty;
  uint32_t col = blockIdx.x * BLOCK_SIZE + tx;

  TYPE sum = 0;
  for (uint32_t t = 0; t < size; t += BLOCK_SIZE) {
    As[ty * BLOCK_SIZE + tx] = A[row * size + t + tx];
    Bs[ty * BLOCK_SIZE + tx] = B[(t + ty) * size + col];
    __syncthreads();

  #ifdef VX_CFG_EXT_LDST_WIDE_ENABLE
    static_assert(__is_same(TYPE, float), "the wide loads read float words");
    static_assert((BLOCK_SIZE % 4) == 0, "the wide loads read A in groups of 4");
    dot_wide<BLOCK_SIZE>(sum, &As[ty * BLOCK_SIZE], &Bs[tx]);
  #else
    #pragma unroll
    for (uint32_t k = 0; k < BLOCK_SIZE; ++k) {
      sum += As[ty * BLOCK_SIZE + k] * Bs[k * BLOCK_SIZE + tx];
    }
  #endif
    __syncthreads();
  }

  C[row * size + col] = sum;
}

__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  auto A = reinterpret_cast<const TYPE*>(arg->A_addr);
  auto B = reinterpret_cast<const TYPE*>(arg->B_addr);
  auto C = reinterpret_cast<TYPE*>(arg->C_addr);
  switch (arg->block_size) {
  case 4:  matrix_mul<4>(A, B, C, arg->size);  break;
  case 8:  matrix_mul<8>(A, B, C, arg->size);  break;
  case 16: matrix_mul<16>(A, B, C, arg->size); break;
  case 32: matrix_mul<32>(A, B, C, arg->size); break;
  }
}
