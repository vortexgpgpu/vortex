#ifndef _DXA_COPY_COMMON_H_
#define _DXA_COPY_COMMON_H_

#include <stdint.h>

#ifndef TYPE
#define TYPE float
#endif

#define DXA_MAX_DIMS 5

typedef struct {
  uint32_t ndim;               // number of dimensions (1-5)
  uint32_t sizes[DXA_MAX_DIMS];  // per-dimension array sizes
  uint32_t tiles[DXA_MAX_DIMS];  // per-dimension tile sizes
  uint32_t grids[DXA_MAX_DIMS];  // per-dimension grid sizes (sizes[d]/tiles[d])
#ifdef DXA_LIFETIME_FIRE_FORGET
  uint32_t fire_and_forget;       // lifetime test: issue DXA, then return without waiting
#endif
  uint64_t src_addr;
  uint64_t dst_addr;
} kernel_arg_t;

#endif
