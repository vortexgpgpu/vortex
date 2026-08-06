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
  uint8_t  use_hw_trilinear;  // hardware: one vx_tex, mip-filter=LINEAR
  uint32_t deltaX;     // (1 << TEX_FXD_FRAC) / dst_width
  uint32_t deltaY;     // (1 << TEX_FXD_FRAC) / dst_height
  uint32_t lod;        // chosen mip level
  uint32_t frac;       // trilinear interpolation weight (0..255)
} kernel_arg_t;

#endif
