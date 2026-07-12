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

  // Software-sampler path (gfx_sw::tex_sample_sw) — gfx_v2 §5 on-device routing.
  // When sw_path != 0 the kernel samples the resident texture via the LSU using
  // these (the same state the TEX DCRs carry), instead of vx_tex4. Validates the
  // SW sampler against the FF golden on real hardware (the §7 oracle relation).
  uint8_t  sw_path;
  uint64_t tex_addr;                       // texture device base (== src_addr)
  uint32_t tex_logdim;                     // (logheight << 16) | logwidth
  uint32_t tex_format;                     // VX_TEX_FORMAT_*
  uint32_t tex_filter;                     // POINT/BILINEAR (+ MIP_LINEAR bit)
  uint32_t tex_wrap;                       // (wrap << 16) | wrap
  uint32_t tex_mipoff[VX_TEX_LOD_MAX + 1]; // per-LOD byte offset from tex_addr
} kernel_arg_t;

#endif
