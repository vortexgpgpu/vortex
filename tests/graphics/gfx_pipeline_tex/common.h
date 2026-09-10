#ifndef _COMMON_H_
#define _COMMON_H_

// gfx_v2 device front end -> RASTER -> interpolate uv -> TEX -> OM, end to end.
//
// The fused setup+binning pipeline (shared test_setup_dims.h / gfx_frontend_k.h)
// produces RASTER's tilebuf + primbuf; RASTER + the fragment kernel (interpolate
// uv from the device-produced primbuf, sample via the TEX fixed-function unit) +
// OM turn that into a textured image, with no host Binning() in the loop.
// Validated dual-path: the device render == the host-Binning render of the same
// textured quad through the identical RASTER+TEX+OM back end.

#include <stdint.h>
#include <test_setup_dims.h>   // pipe_arg_t, PIPE_* constants/stages (-I gfx_setup_kernel)

// Fragment kernel args (gfx_draw3d): interpolate from primbuf, sample TEX, OM.
typedef struct {
  uint64_t prim_addr;
  uint32_t depth_enabled;
  uint32_t color_enabled;
  uint32_t tex_enabled;
  uint32_t tex_modulate;
  // Fragment-export aperture encoding (must match the OM aperture DCRs).
  uint32_t aperture_xbits;
  uint32_t aperture_ybits;
  uint32_t aperture_record_shift;
} frag_arg_t;

#endif
