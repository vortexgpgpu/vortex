#ifndef _DXA_COPY_MW_COMMON_H_
#define _DXA_COPY_MW_COMMON_H_

#include <stdint.h>

#ifndef TYPE
#define TYPE float
#endif

enum {
  DXA_COPY_MCAST_MODE_SMOKE  = 0,
  DXA_COPY_MCAST_MODE_PERCTA = 1,
  DXA_COPY_MCAST_MODE_MCAST  = 2,
};

enum {
  DXA_COPY_MCAST_WRITEBACK_FULL   = 0,
  DXA_COPY_MCAST_WRITEBACK_SAMPLE = 1,
  DXA_COPY_MCAST_WRITEBACK_NONE   = 2,
};

#define DXA_COPY_MCAST_MAX_PIPELINE_DEPTH 8

// Intra-core multicast copy test arguments.
typedef struct {
  uint32_t mode;
  uint32_t writeback_mode;
  uint32_t pipeline_depth;
  uint32_t tile_rows;
  uint32_t tile_cols;
  uint32_t src_rows;
  uint32_t src_cols;
  uint32_t src_row_stride;  // elements between source rows (descriptor stride)
  uint32_t num_recv;        // multicast group size (= CTAs sharing the tile)
  uint64_t src_addr;
  uint64_t dst_addr;        // smoke: num_recv x tile; benchmark: num_recv x matrix
} kernel_arg_t;

#endif
