#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>

typedef struct {
  uint32_t dst_width;
  uint32_t dst_height;
  uint64_t cbuf_addr;
  uint8_t  cbuf_stride;
  uint32_t cbuf_pitch;
  uint64_t prim_addr;

  // Software fine-rasterizer path (gfx_rast::rast_walk_primitive) — gfx_v2 §5
  // on-device routing (the -z/use_sw fork). When sw_path != 0 the kernel walks
  // the resident primitives itself (one thread per primitive) instead of being
  // launched per covered-quad wave by the FF RASTER work distributor (push).
  uint32_t sw_path;
  uint32_t num_prims;     // dense rast_prim_t count in prim_addr
  uint32_t tile_logsize;  // walk tile = 1 << tile_logsize
} kernel_arg_t;

#endif
