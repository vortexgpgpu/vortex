#ifndef _DXA_COPY_COMMON_H_
#define _DXA_COPY_COMMON_H_

#include <stdint.h>

#ifndef TYPE
#define TYPE float
#endif

typedef struct {
  uint32_t tile_rows;    // rows per CTA tile
  uint32_t tile_cols;    // cols per CTA tile
  uint32_t ncols;        // total columns in source array
  uint32_t nrows;        // total rows in source array
  uint64_t src_addr;
} kernel_arg_t;

#endif
