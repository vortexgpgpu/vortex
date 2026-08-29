#ifndef _COMMON_H_
#define _COMMON_H_

#include <stdint.h>
#include <VX_types.h>
#include <gfx_sw.h>   // gfx_sw::TexState / om_state_t — software TEX/OM emitters

// KMU kernel-arg layout for the TEX+RASTER+OM pipeline.
//
// Each stage runs on its fixed-function unit when that unit is in the build and
// on its gfx_sw software emitter when it is not (NO_TEX / NO_OM / NO_RASTER).
// Host and kernel both key off VX_CFG_EXT_*_ENABLED, so they cannot disagree
// about which path a stage takes, and no stage can issue an FF instruction on a
// device whose FF unit was never built.
typedef struct {
  uint64_t prim_addr;
  uint32_t depth_enabled;
  uint32_t color_enabled;
  uint32_t tex_enabled;
  uint32_t tex_modulate;
  // Fragment-export aperture encoding. The shader builds its own store address
  // from these; they MUST match the OM aperture DCRs the runtime programs, or the
  // ingress misreads every record. Both come from one call to set_aperture().
  uint32_t aperture_xbits;
  uint32_t aperture_ybits;
  uint32_t aperture_record_shift;

  // ── software-routing state (unused fields stay zero) ──
  uint32_t dst_width;
  uint32_t dst_height;
  // SW RASTER: the FF work distributor is absent, so the host launches a grid of
  // one thread per resident primitive and the kernel walks the screen itself.
  uint32_t num_prims;
  uint32_t tile_logsize;      // walk tile = 1 << tile_logsize
  gfx_sw::TexState   tex;     // SW TEX: mirrors the TEX DCRs
  gfx_sw::om_state_t om;      // SW OM: mirrors the OM DCRs, resolve_om_state()'d
} kernel_arg_t;

// RASTER dispatch v2 (push): the raster work distributor launches the kernel once
// per covered-quad wave; it runs straight-line and exits (no worker loop). Under
// NO_RASTER the kernel is instead a normal grid over the resident primitives.

#endif
