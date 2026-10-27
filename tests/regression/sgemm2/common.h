#ifndef _COMMON_H_
#define _COMMON_H_

#ifndef TYPE
#define TYPE float
#endif

// Supported (tile, chunk_k, reg) shapes. The kernel is instantiated for each
// one, so the tile loops have compile-time bounds; the host rejects others.
//   tile : the block computes a tile x tile block of C
//   chunk: the k-slice of A and B staged in local memory per step
//   reg  : each thread computes reg x reg outputs (block = (tile/reg)^2 threads)
#define SGEMM2_CONFIGS(X) \
  X(4, 4, 1)   X(4, 16, 1)               \
  X(8, 8, 1)   X(8, 32, 1)               \
  X(16, 16, 1) X(16, 32, 1)              \
  X(32, 32, 1)                           \
  X(8, 8, 2)   X(16, 16, 2) X(32, 16, 2) X(64, 16, 2) \
  X(16, 16, 4) X(32, 16, 4) X(64, 16, 4) X(64, 8, 4)  \
  X(128, 8, 4)

typedef struct {
  uint32_t size;
  uint32_t tile_size;
  uint32_t chunk_k;
  uint32_t reg_size;
  uint64_t A_addr;
  uint64_t B_addr;
  uint64_t C_addr;
} kernel_arg_t;

#endif
