#ifndef _DXA_COPY_MULTICAST_COMMON_H_
#define _DXA_COPY_MULTICAST_COMMON_H_

#include <stdint.h>

#ifndef TYPE
#define TYPE float
#endif

typedef struct {
  uint32_t tile_rows;
  uint32_t tile_cols;
  uint32_t ncols;
  uint32_t nrows;
  uint64_t src_addr;
  uint64_t dst_addr;    // verification buffer in GMEM
  uint32_t num_ctas;    // co-resident CTAs (for multicast mask)
  uint32_t active_ctas; // CTAs that actually do work (0 = all)
} kernel_arg_t;

#endif
