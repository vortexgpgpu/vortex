#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>

// T Tile dimensions: 16x16 fp32 = 1KB per tile register
#define TILE_SIZE 16
#define T_TILE_BYTES (TILE_SIZE * TILE_SIZE * sizeof(float))  // 1KB

typedef struct {
  uint64_t A_addr;  // Matrix A (16x16 fp32)
  uint64_t B_addr;  // Matrix B (16x16 fp32)
  uint64_t C_addr;  // Matrix C (16x16 fp32)
} kernel_arg_t;

#endif // _COMMON_H_
