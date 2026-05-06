#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>
#include <VX_config.h>
#include <VX_types.h>

// vortex2 KMU launch kernel-arg layout. Host pre-computes per-pixel
// stride / fixed-point delta and passes everything as scalar fields
// instead of running a TextureSampler on-device.
typedef struct {
  uint64_t dst_addr;
  uint32_t dst_width;
  uint32_t dst_height;
  uint32_t dst_pitch;
  uint8_t  dst_stride;
  uint8_t  filter;     // 0=POINT, 1=BILINEAR
  uint8_t  use_trilinear;
  uint8_t  _pad;
  uint32_t deltaX;     // (1 << VX_TEX_FXD_FRAC) / dst_width
  uint32_t deltaY;     // (1 << VX_TEX_FXD_FRAC) / dst_height
  uint32_t lod;        // chosen mip level
  uint32_t frac;       // trilinear interpolation weight (0..255)
} kernel_arg_t;

#endif
