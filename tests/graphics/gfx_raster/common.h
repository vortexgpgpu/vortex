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

  // Software fine-rasterizer path (gfx_rast::rast_walk_primitive) — on-device
  // routing (the -z/use_sw fork). When sw_path != 0 the kernel walks
  // the resident primitives itself (one thread per primitive) instead of being
  // launched per covered-quad wave by the FF RASTER work distributor (push).
  uint32_t sw_path;
  uint32_t num_prims;     // dense rast_prim_t count in prim_addr
  uint32_t tile_logsize;  // walk tile = 1 << tile_logsize
  uint32_t scissor_x;     // the draw rectangle the scissor DCRs hold, packed alike
  uint32_t scissor_y;

  // When pid_color != 0 a covered pixel takes prim_id_color(pid_base + pid), the
  // draw-wide index of the record the stamp's pass-local pid names.
  uint32_t pid_color;
  uint32_t pid_base;      // draw-wide index of the pass's first record

  // Samples per pixel (1 or 4). At 4 the colour buffer is sample-interleaved --
  // sample k of (x, y) at cbuf_addr + y*cbuf_pitch + x*cbuf_stride + k*4 -- and a
  // covered pixel writes every sample its coverage mask names.
  uint32_t samples;
} kernel_arg_t;

// A distinct colour per draw-wide primitive index, never the clear colour.
static inline uint32_t prim_id_color(uint32_t id) {
  return 0xff000000u | (id + 1);
}

#endif
