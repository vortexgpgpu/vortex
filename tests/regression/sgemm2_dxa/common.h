#ifndef _SGEMM2_DXA_COMMON_H_
#define _SGEMM2_DXA_COMMON_H_

#ifndef TYPE
#define TYPE float
#endif

typedef struct {
  uint32_t grid_dim[2];
  uint32_t block_dim[2];
  uint32_t size;
  uint32_t tile_size;
  uint32_t chunk_k;
  // 1: single-buffered DXA, 2: double-buffered DXA
  uint32_t mode;
  uint64_t A_addr;
  uint64_t B_addr;
  uint64_t C_addr;
} kernel_arg_t;

#endif
