#ifndef _COMMON_H_
#define _COMMON_H_

// gfx_v2 device front end -> RASTER -> fragment interpolation -> OM, end to end.
//
// The fused setup+binning pipeline (shared test_setup_dims.h / gfx_frontend_k.h)
// produces RASTER's tilebuf + primbuf (pinned, dense tile grid); RASTER + the
// interpolate kernel + the OM fixed-function unit then turn that into shaded
// pixels with no host Binning() in the loop. The rendered colour image is
// checked against the gfx-v1 reference (gfx_draw3d).

#include <stdint.h>
#include <test_setup_dims.h>   // pipe_arg_t, PIPE_* constants/stages (-I gfx_setup_kernel)

// Fragment kernel args (gfx_draw3d): interpolate from primbuf, write via vx_om.
typedef struct {
  uint64_t prim_addr;
  uint32_t depth_enabled;
  uint32_t color_enabled;
  uint32_t tex_enabled;
  uint32_t tex_modulate;
} frag_arg_t;

#endif
