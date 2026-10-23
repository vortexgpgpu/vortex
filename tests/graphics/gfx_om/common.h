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
  uint32_t  rt2;          // the second colour attachment of the MRT kernel
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

  // Per-sample merging (kernel_main_msaa): under 4x every fragment covers the
  // samples msaa_frag_mask names, each at its own depth, and exports once per
  // sample; at one sample it exports sample 0 alone.
  uint32_t  samples;
  uint32_t  depth_float;   // D32F: the depth is the whole word, face in the address
  uint32_t  depth_only;    // one-word depth records, colour masked off
  uint32_t  mrt;           // also export to attachment rt2
} kernel_arg_t;

// The 4x pattern the multisample kernel draws and the host reference replays:
// the samples (x, y) covers (never none, sometimes all), the depth of sample k
// (Q24 for a unorm target, a float for D32F) and the depth each sample starts at.
static inline uint32_t msaa_frag_mask(uint32_t x, uint32_t y) {
  return ((x * 5u + y * 3u) % 15u) + 1u;
}
static inline uint32_t msaa_frag_depth(uint32_t x, uint32_t y, uint32_t k) {
  return ((x * 37u + y * 11u + k * 53u) & 0xffu) << 16;
}
static inline float msaa_frag_depth_f(uint32_t x, uint32_t y, uint32_t k) {
  return (float)((x * 37u + y * 11u + k * 53u) & 0xffu) / 128.0f - 1.0f;
}
static inline uint32_t msaa_init_depth(uint32_t x, uint32_t y, uint32_t k) {
  return ((x * 13u + y * 7u + k * 29u) & 0xffu) << 16;
}
static inline float msaa_init_depth_f(uint32_t x, uint32_t y, uint32_t k) {
  return (float)((x * 13u + y * 7u + k * 29u) & 0xffu) / 128.0f - 1.0f;
}

#endif
