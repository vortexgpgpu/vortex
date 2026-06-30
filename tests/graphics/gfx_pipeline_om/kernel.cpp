#include <vx_spawn2.h>
#include <vx_graphics.h>
#include <vx_raytrace.h>   // vx_gfx_set (SETW) — stage the vx_om4 payload window
#include <cocogfx/include/color.hpp>
#include <cocogfx/include/math.hpp>
#include "common.h"
#include <gfx_frontend_k.h>   // setup_k + binning_k (-I gfx_setup_kernel)

// vx_om4 payload window: slots 0..3 = colour[0..3], 4..7 = depth[0..3].
static const unsigned OM_WIN = 0;

// gfx_v2 device front end + fragment-interpolation + OM in one module: the
// shared pipeline (setup_k / binning_k) produces RASTER's tilebuf + primbuf;
// this fragment kernel (gfx_draw3d's) interpolates the primitive's colour from
// the device-produced primbuf (frag_payload_t bcoords + PID) and writes via OM.

#define INTERPOLATE_i(i, dst, src) { \
    auto tmp = src.x * dx[i] + src.z; dst[i] = src.y * dy[i] + tmp; }

#define TO_RGBA_i(i, dst, sr, sg, sb, sa) \
    dst[i].r = static_cast<uint8_t>(sr[i] * 255); \
    dst[i].g = static_cast<uint8_t>(sg[i] * 255); \
    dst[i].b = static_cast<uint8_t>(sb[i] * 255); \
    dst[i].a = static_cast<uint8_t>(sa[i] * 255)

#define STAGE_i(i, color, depth) \
    vx_gfx_set(OM_WIN + (i),     color[i].value); \
    vx_gfx_set(OM_WIN + 4 + (i), static_cast<uint32_t>(depth[i] * 65336))

// RASTER dispatch v2 (FWD-5): bcoords come from the per-lane window payload (`p`)
// instead of the bcoord CSRs. p.bcoord[axis][corner] holds the raw Q15.16 bits.
#define BCOORD_PL_AS_FLOAT(axis, i) \
    static_cast<float>(fixed16_t::make(static_cast<int32_t>(p.bcoord[axis][i])))
#define GRADIENTS_PL_i(i) { \
    auto F0 = BCOORD_PL_AS_FLOAT(0, i); auto F1 = BCOORD_PL_AS_FLOAT(1, i); \
    auto F2 = BCOORD_PL_AS_FLOAT(2, i); auto recip = 1.0f / (F0 + F1 + F2); \
    dx[i] = FloatA(recip * F0); dy[i] = FloatA(recip * F1); }
#define GRADIENTS_PL   GRADIENTS_PL_i(0) GRADIENTS_PL_i(1) GRADIENTS_PL_i(2) GRADIENTS_PL_i(3)
#define INTERPOLATE(d, s) INTERPOLATE_i(0,d,s); INTERPOLATE_i(1,d,s); INTERPOLATE_i(2,d,s); INTERPOLATE_i(3,d,s)
#define TO_RGBA(d, r, g, b, a) TO_RGBA_i(0,d,r,g,b,a); TO_RGBA_i(1,d,r,g,b,a); TO_RGBA_i(2,d,r,g,b,a); TO_RGBA_i(3,d,r,g,b,a)
#define OUTPUT_QUAD(pos_mask, face, color, depth) \
    STAGE_i(0, color, depth); STAGE_i(1, color, depth); \
    STAGE_i(2, color, depth); STAGE_i(3, color, depth); \
    vx_om4((pos_mask) | ((unsigned)(face) << 31), OM_WIN)

__kernel void kernel_main(frag_arg_t* __UNIFORM__ arg) {
  using namespace vortex::graphics;
  FloatA z[4], r[4], g[4], b[4], a[4], dx[4], dy[4];
  cocogfx::ColorARGB out_color[4];
  for (int i = 0; i < 4; ++i) {
    z[i] = FloatA(0.0f); r[i] = FloatA(1.0f); g[i] = FloatA(1.0f);
    b[i] = FloatA(1.0f); a[i] = FloatA(1.0f);
  }
  auto prim_ptr = reinterpret_cast<rast_prim_t*>(arg->prim_addr);

  // RASTER dispatch v2 (push): straight-line FS, payload already in the window.
  frag_payload_t p;
  vx_frag_load(p);
  uint32_t pos_mask = p.pos_mask;
  uint32_t pid = p.pid;
  auto& attribs = prim_ptr[pid].attribs;
  GRADIENTS_PL
  if (arg->depth_enabled) { INTERPOLATE(z, attribs.z); }
  if (arg->color_enabled) {
    INTERPOLATE(r, attribs.r); INTERPOLATE(g, attribs.g);
    INTERPOLATE(b, attribs.b); INTERPOLATE(a, attribs.a);
  }
  TO_RGBA(out_color, r, g, b, a);
  OUTPUT_QUAD(pos_mask, 0, out_color, z);
}
