#ifndef _DXA_COPY_MW_COMMON_H_
#define _DXA_COPY_MW_COMMON_H_

#include <stdint.h>

#ifndef TYPE
#define TYPE float
#endif

// Intra-core multicast copy test arguments.
typedef struct {
  uint32_t tile_rows;
  uint32_t tile_cols;
  uint32_t src_row_stride;  // elements between source rows (descriptor stride)
  uint32_t num_recv;        // multicast group size (= CTAs sharing the tile)
  uint64_t src_addr;
  uint64_t dst_addr;        // tightly packed: num_recv × tile_rows × tile_cols
} kernel_arg_t;

#endif
