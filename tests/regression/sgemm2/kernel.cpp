#include <vx_spawn2.h>
#include "common.h"

static_assert(sizeof(TYPE) == sizeof(uint32_t), "copy_word moves 32-bit elements");

typedef uint32_t __attribute__((may_alias)) word_t;

// Copies one element through an integer register, which leaves the floating-
// point register file to the accumulators.
static inline __attribute__((always_inline))
void copy_word(TYPE* dst, const TYPE* src) {
  *reinterpret_cast<word_t*>(dst) = *reinterpret_cast<const word_t*>(src);
}

// Shared-memory tiled SGEMM with register blocking.
//
// A block computes a TILE x TILE tile of C with (TILE/REG)^2 threads; each
// thread owns REG x REG outputs at rows ty + i*TX and columns tx + j*TX
// (TX = TILE/REG). threadIdx.x walks columns, so a warp reads consecutive B
// words and shares the A words of its row(s). Each k-step stages a
// TILE x CHUNK slice of A and a CHUNK x TILE slice of B in local memory.
template <uint32_t TILE, uint32_t CHUNK, uint32_t REG>
static inline __attribute__((always_inline))
void sgemm_tile(const TYPE* A, const TYPE* B, TYPE* C, uint32_t size) {
  constexpr uint32_t TX = TILE / REG;
  constexpr uint32_t NT = TX * TX;
  constexpr uint32_t LOADS = (TILE * CHUNK) / NT;
  static_assert((TILE % REG) == 0, "tile must be a multiple of reg");
  static_assert(((TILE * CHUNK) % NT) == 0, "slice must split evenly over threads");
  static_assert((NT % CHUNK) == 0 && (NT % TILE) == 0, "a copy pass must cover whole rows");

  uint32_t tx = threadIdx.x;
  uint32_t ty = threadIdx.y;
  uint32_t tid = ty * TX + tx;
  uint32_t row0 = blockIdx.y * TILE;
  uint32_t col0 = blockIdx.x * TILE;

  auto local_A = reinterpret_cast<TYPE*>(__local_mem()); // [TILE][CHUNK]
  auto local_B = local_A + TILE * CHUNK;                  // [CHUNK][TILE]

  // Cooperative-load sources: consecutive threads copy consecutive words, and
  // each pass covers NT/CHUNK rows of A and NT/TILE rows of B.
  const TYPE* src_A = A + (row0 + tid / CHUNK) * size + tid % CHUNK;
  const TYPE* src_B = B + (tid / TILE) * size + col0 + tid % TILE;
  uint32_t step_A = (NT / CHUNK) * size;
  uint32_t step_B = (NT / TILE) * size;

  TYPE acc[REG][REG];
  #pragma unroll
  for (uint32_t i = 0; i < REG; ++i) {
    #pragma unroll
    for (uint32_t j = 0; j < REG; ++j) {
      acc[i][j] = 0;
    }
  }

  for (uint32_t k0 = 0; k0 < size; k0 += CHUNK) {
    #pragma unroll
    for (uint32_t l = 0; l < LOADS; ++l) {
      copy_word(&local_A[tid + l * NT], &src_A[l * step_A]);
    }
    #pragma unroll
    for (uint32_t l = 0; l < LOADS; ++l) {
      copy_word(&local_B[tid + l * NT], &src_B[l * step_B]);
    }
    src_A += CHUNK;
    src_B += CHUNK * size;

    __syncthreads();

    #pragma unroll
    for (uint32_t k = 0; k < CHUNK; ++k) {
      TYPE a[REG], b[REG];
      #pragma unroll
      for (uint32_t i = 0; i < REG; ++i) {
        a[i] = local_A[(ty + i * TX) * CHUNK + k];
      }
      #pragma unroll
      for (uint32_t j = 0; j < REG; ++j) {
        b[j] = local_B[k * TILE + tx + j * TX];
      }
      #pragma unroll
      for (uint32_t i = 0; i < REG; ++i) {
        #pragma unroll
        for (uint32_t j = 0; j < REG; ++j) {
          acc[i][j] += a[i] * b[j];
        }
      }
    }

    __syncthreads();
  }

  #pragma unroll
  for (uint32_t i = 0; i < REG; ++i) {
    #pragma unroll
    for (uint32_t j = 0; j < REG; ++j) {
      C[(row0 + ty + i * TX) * size + col0 + tx + j * TX] = acc[i][j];
    }
  }
}

__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  auto A = reinterpret_cast<const TYPE*>(arg->A_addr);
  auto B = reinterpret_cast<const TYPE*>(arg->B_addr);
  auto C = reinterpret_cast<TYPE*>(arg->C_addr);
  uint32_t size  = arg->size;
  uint32_t tile  = arg->tile_size;
  uint32_t chunk = arg->chunk_k;
  uint32_t reg   = arg->reg_size;
#define SGEMM2_DISPATCH(T, K, R) \
  if (tile == T && chunk == K && reg == R) { sgemm_tile<T, K, R>(A, B, C, size); return; }
  SGEMM2_CONFIGS(SGEMM2_DISPATCH)
#undef SGEMM2_DISPATCH
}
