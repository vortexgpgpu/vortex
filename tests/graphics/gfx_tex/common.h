#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>
#include <VX_types.h>

// KMU launch kernel-arg layout. Host pre-computes per-pixel stride and
// fixed-point deltas and passes them as scalar fields.
typedef struct {
  uint64_t dst_addr;
  uint32_t dst_width;
  uint32_t dst_height;
  uint32_t dst_pitch;
  uint8_t  dst_stride;
  uint8_t  filter;     // 0=POINT, 1=BILINEAR
  uint8_t  use_trilinear;     // software-composed: two vx_tex + kernel lerp
  uint32_t deltaX;     // (1 << TEX_FXD_FRAC) / dst_width
  uint32_t deltaY;     // (1 << TEX_FXD_FRAC) / dst_height
  uint32_t lod;        // chosen mip level
  uint32_t frac;       // trilinear interpolation weight (0..255)
  // Border mode only: a coordinate span wider than the texture, so the outer
  // margin of the destination lands outside [0,1) where a border wrap differs
  // from a clamp. Appended so the fields above keep their offsets and the
  // single-span shader keeps its instruction sequence.
  int32_t  uv_bias;
  int32_t  uv_delta;
} kernel_arg_t;

#endif
