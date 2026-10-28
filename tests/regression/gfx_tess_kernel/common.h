#ifndef _COMMON_H_
#define _COMMON_H_

// The SIMT tessellator, checked bit for bit against the reference tessellator
// llvmpipe runs (ref/, the D3D11 reference CHWTessellator).
//
// A batch of patches of every domain, spacing and output, each with its own
// levels, runs through the kernels vortexpipe launches:
//   SETUP   one thread per patch: the control stage's epilogue,
//           gfx_tess_patch_setup, into the patch's descriptor and counts
//   scan_k  the device-wide scan of the counts (gfx_scan_k.h)
//   POINTS  one thread per domain point: the evaluation stage's prologue,
//           gfx_tess_eval_point, into the point's patch and (u, v)
//   tess_stitch_k  one thread per stitch unit into the primitive list
// and the host compares the counts, every point and every primitive with the
// reference's, in the reference's order.

#include <stdint.h>

#define TESS_STAGE_SETUP  0
#define TESS_STAGE_POINTS 1

typedef struct {
  uint32_t stage;
  uint32_t n;              // patches (SETUP) or domain points (POINTS)
  uint64_t levels_addr;    // float[8] per patch: outer[4], inner[2], mode bits, gl_PrimitiveID bits
  uint64_t desc_addr;      // gfx_tess_patch_t per patch
  uint64_t counts_addr;    // uint32[3] per patch
  uint64_t draw_addr;      // gfx_tess_draw_t
  uint64_t points_addr;    // uint32[4] per point: patch, primitive id, u, v
} kernel_arg_t;

#endif
