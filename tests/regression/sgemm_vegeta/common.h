#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>

// T Tile dimensions: 16x16 fp32 = 1KB per tile register
#define TILE_SIZE 16
#define T_TILE_BYTES (TILE_SIZE * TILE_SIZE * sizeof(float))  // 1KB
#define U_TILE_BYTES (2 * T_TILE_BYTES)  // 2KB (U-reg = 2 T-regs for dense)
#define V_TILE_BYTES (4 * T_TILE_BYTES)  // 4KB (V-reg = 4 T-regs for dense)
#define M_TILE_BYTES (TILE_SIZE * TILE_SIZE / 2)  // 128 bytes (metadata: 2 nibbles per byte)

// GEMM modes
typedef enum {
  GEMM_MODE_TGEMM = 0,  // T x T -> T (dense x dense)
  GEMM_MODE_UGEMM = 1,  // T x U -> T (sparse 2:4 packed x dense 2x)
  GEMM_MODE_VGEMM = 2,  // T x V -> T (sparse 1:4 packed x dense 4x)
  GEMM_MODE_RGEMM = 3   // T x U -> U (row-wise N:4 sparse x dense 2x)
} gemm_mode_t;

typedef struct {
  uint64_t A_addr;      // Matrix A (1KB T-tile, sparse for UGEMM/VGEMM)
  uint64_t B_addr;      // Matrix B (1KB/2KB/4KB depending on mode, always dense)
  uint64_t M_addr;      // Metadata for sparse A (128 bytes, only for UGEMM/VGEMM)
  uint64_t C_addr;      // Matrix C result (1KB T-tile)
  uint32_t mode;        // GEMM mode (TGEMM=0, UGEMM=1, VGEMM=2)
} kernel_arg_t;

#endif // _COMMON_H_
