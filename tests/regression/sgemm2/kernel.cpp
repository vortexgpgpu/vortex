#include <vx_spawn2.h>
#include "common.h"

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

  #if LDST_WIDE
    static_assert(__is_same(TYPE, float), "LDST_WIDE needs float");
    static_assert((BLOCK_SIZE % 4) == 0, "LDST_WIDE reads A in groups of 4");
    #pragma unroll
    for (uint32_t k = 0; k < BLOCK_SIZE; k += 4) {
      vx_float4 a = vx_flw_q(&As[ty * BLOCK_SIZE + k]);
      sum += a[0] * Bs[k * BLOCK_SIZE + tx];
      sum += a[1] * Bs[(k + 1) * BLOCK_SIZE + tx];
      sum += a[2] * Bs[(k + 2) * BLOCK_SIZE + tx];
      sum += a[3] * Bs[(k + 3) * BLOCK_SIZE + tx];
    }
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
