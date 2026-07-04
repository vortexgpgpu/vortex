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

// Depth word: the screen-space plane MAC is a Q7.24 z value; write it saturated
// to the 24-bit zbuf range so edge extrapolation clamps to near / far.
#define DEPTH_WORD(d) \
    ((d).data() < 0 ? 0u \
      : ((uint32_t)(d).data() > (uint32_t)VX_OM_DEPTH_MASK ? (uint32_t)VX_OM_DEPTH_MASK \
                                                           : (uint32_t)(d).data()))
#define STAGE_i(i, color, depth) \
    vx_gfx_set(OM_WIN + (i),     color[i].value); \
    vx_gfx_set(OM_WIN + 4 + (i), DEPTH_WORD(depth[i]))

// P2: per-corner edge value F_axis recomputed in-shader from the primitive's edge
// coefficients (a*X+b*Y+c in Q15.16, bit-identical to the raster HW bcoord); the
// quad origin is (qx*2, qy*2) and corner i offsets by (i&1, i>>1). `edges`, `qx`,
// `qy` are bound in the shader body.
#define EDGE_PIX_X(i) (((int32_t)qx << 1) + ((int32_t)(i) & 1))
#define EDGE_PIX_Y(i) (((int32_t)qy << 1) + ((int32_t)(i) >> 1))
#define BCOORD_PL_AS_FLOAT(axis, i) \
    static_cast<float>(fixed16_t::make( \
        edges[axis].x.data() * EDGE_PIX_X(i) \
      + edges[axis].y.data() * EDGE_PIX_Y(i) \
      + edges[axis].z.data()))
#define GRADIENTS_PL_i(i) { \
    auto F0 = BCOORD_PL_AS_FLOAT(0, i); auto F1 = BCOORD_PL_AS_FLOAT(1, i); \
    auto F2 = BCOORD_PL_AS_FLOAT(2, i); auto recip = 1.0f / (F0 + F1 + F2); \
    dx[i] = FloatA(recip * F0); dy[i] = FloatA(recip * F1); }
#define GRADIENTS_PL   GRADIENTS_PL_i(0) GRADIENTS_PL_i(1) GRADIENTS_PL_i(2) GRADIENTS_PL_i(3)
// P3: depth is a fixed-function screen-space plane Z = A'*X + B'*Y + C' (coeffs in
// attribs.z {x:A', y:B', z:C'}, Q7.24), evaluated by an integer MAC bit-identical
// to the raster early-Z so early-Z and late-Z agree.
#define PLANE_Z_i(i) fixed24_t::make((int32_t)( \
      (int64_t)attribs.z.x.data() * EDGE_PIX_X(i) \
    + (int64_t)attribs.z.y.data() * EDGE_PIX_Y(i) \
    + (int64_t)attribs.z.z.data()))
#define PLANE_Z(dst) \
    dst[0] = PLANE_Z_i(0); dst[1] = PLANE_Z_i(1); \
    dst[2] = PLANE_Z_i(2); dst[3] = PLANE_Z_i(3)
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
  auto& edges = prim_ptr[pid].edges;   // P2: recompute edge values from these
  uint32_t qx = (pos_mask >> 4) & ((1u << (VX_RASTER_DIM_BITS - 1)) - 1);
  uint32_t qy = (pos_mask >> (4 + (VX_RASTER_DIM_BITS - 1))) & ((1u << (VX_RASTER_DIM_BITS - 1)) - 1);
  GRADIENTS_PL
  if (arg->depth_enabled) { PLANE_Z(z); }
  if (arg->color_enabled) {
    INTERPOLATE(r, attribs.r); INTERPOLATE(g, attribs.g);
    INTERPOLATE(b, attribs.b); INTERPOLATE(a, attribs.a);
  }
  TO_RGBA(out_color, r, g, b, a);
  OUTPUT_QUAD(pos_mask, 0, out_color, z);
}
