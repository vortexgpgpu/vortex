#ifndef _COMMON_H_
#define _COMMON_H_

#include <VX_types.h>
#include <stdint.h>
#include <gfx_sw.h>   // gfx_sw::om_state_t / om_fragment — software OM path

typedef struct {
  uint32_t  dst_width;
  uint32_t  dst_height;
  uint32_t  color;        // packed ARGB src color (used as base before scaling)
  uint32_t  depth;
  uint32_t  backface;     // 0/1
  uint32_t  blend_enable; // 0/1
  uint32_t  a_scale_q16;  // 16.16 fixed-point
  uint32_t  r_scale_q16;
  uint32_t  g_scale_q16;
  uint32_t  b_scale_q16;

  // Fragment-export aperture encoding (must match the OM aperture DCRs).
  uint32_t  aperture_xbits;
  uint32_t  aperture_ybits;
  uint32_t  aperture_record_shift;

  // Software output-merger path (gfx_sw::om_fragment) — on-device routing.
  // When sw_path != 0 the kernel merges each fragment via the LSU using
  // `om` (filled + resolve_om_state()'d on the host, mirroring the OM DCRs)
  // instead of vx_om4. Validates the SW OM against the FF golden on real HW.
  uint32_t           sw_path;
  gfx_sw::om_state_t om;
} kernel_arg_t;

#endif
