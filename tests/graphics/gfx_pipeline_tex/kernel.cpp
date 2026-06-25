#include <vx_spawn2.h>
#include <vx_graphics.h>
#include <vx_raytrace.h>   // vx_rt_set (SETW) — stage the vx_om4 payload window
#include <cocogfx/include/color.hpp>
#include <cocogfx/include/math.hpp>
#include "common.h"
#include <pipe_frontend.h>   // setup_k + binning_k (-I gfx_setup_kernel)

// vx_om4 payload window: slots 0..3 = colour[0..3], 4..7 = depth[0..3].
static const unsigned OM_WIN = 0;

// gfx_v2 device front end + fragment (interpolate uv + TEX sample) + OM in one
// module: the shared pipeline produces RASTER's tilebuf + primbuf; this fragment
// kernel is gfx_draw3d's (RASTER bcoord CSRs + PID -> interpolated uv -> vx_tex
// -> OM), exercising the TEX fixed-function unit fed device-side.

using fixeduv_t = vortex::graphics::fixed_t<VX_TEX_FXD_FRAC>;

#define INTERPOLATE_i(i, dst, src) { \
    auto tmp = src.x * dx[i] + src.z; dst[i] = src.y * dy[i] + tmp; }

#define MODULATE_i(i, dst, s1r, s1g, s1b, s1a, s2) \
    dst[i].r = static_cast<uint8_t>(s1r[i] * s2[i].r); \
    dst[i].g = static_cast<uint8_t>(s1g[i] * s2[i].g); \
    dst[i].b = static_cast<uint8_t>(s1b[i] * s2[i].b); \
    dst[i].a = static_cast<uint8_t>(s1a[i] * s2[i].a)

#define TO_RGBA_i(i, dst, sr, sg, sb, sa) \
    dst[i].r = static_cast<uint8_t>(sr[i] * 255); \
    dst[i].g = static_cast<uint8_t>(sg[i] * 255); \
    dst[i].b = static_cast<uint8_t>(sb[i] * 255); \
    dst[i].a = static_cast<uint8_t>(sa[i] * 255)

#define STAGE_i(i, color, depth) \
    vx_rt_set(OM_WIN + (i),     color[i].value); \
    vx_rt_set(OM_WIN + 4 + (i), static_cast<uint32_t>(depth[i].data()))

// RASTER dispatch v2 (FWD-5): bcoords from the per-lane window payload (`p`).
#define BCOORD_PL_AS_FLOAT(axis, i) \
    static_cast<float>(fixed16_t::make(static_cast<int32_t>(p.bcoord[axis][i])))
#define GRADIENTS_PL_i(i) { \
    auto F0 = BCOORD_PL_AS_FLOAT(0, i); auto F1 = BCOORD_PL_AS_FLOAT(1, i); \
    auto F2 = BCOORD_PL_AS_FLOAT(2, i); auto recip = 1.0f / (F0 + F1 + F2); \
    dx[i] = FloatA(recip * F0); dy[i] = FloatA(recip * F1); }
#define GRADIENTS_PL   GRADIENTS_PL_i(0) GRADIENTS_PL_i(1) GRADIENTS_PL_i(2) GRADIENTS_PL_i(3)
#define INTERPOLATE(d, s) INTERPOLATE_i(0,d,s); INTERPOLATE_i(1,d,s); INTERPOLATE_i(2,d,s); INTERPOLATE_i(3,d,s)
#define MODULATE(d, r, g, b, a, s) MODULATE_i(0,d,r,g,b,a,s); MODULATE_i(1,d,r,g,b,a,s); MODULATE_i(2,d,r,g,b,a,s); MODULATE_i(3,d,r,g,b,a,s)
#define REPLACE(d, s) d[0] = s[0]; d[1] = s[1]; d[2] = s[2]; d[3] = s[3]
#define TO_RGBA(d, r, g, b, a) TO_RGBA_i(0,d,r,g,b,a); TO_RGBA_i(1,d,r,g,b,a); TO_RGBA_i(2,d,r,g,b,a); TO_RGBA_i(3,d,r,g,b,a)
#define TEXTURING(d, u, v) \
    d[0] = vx_tex(0, fixeduv_t(u[0]).data(), fixeduv_t(v[0]).data(), 0); \
    d[1] = vx_tex(0, fixeduv_t(u[1]).data(), fixeduv_t(v[1]).data(), 0); \
    d[2] = vx_tex(0, fixeduv_t(u[2]).data(), fixeduv_t(v[2]).data(), 0); \
    d[3] = vx_tex(0, fixeduv_t(u[3]).data(), fixeduv_t(v[3]).data(), 0)
#define OUTPUT_QUAD(pos_mask, face, color, depth) \
    STAGE_i(0, color, depth); STAGE_i(1, color, depth); \
    STAGE_i(2, color, depth); STAGE_i(3, color, depth); \
    vx_om4((pos_mask) | ((unsigned)(face) << 31), OM_WIN)

__kernel void kernel_main(frag_arg_t* __UNIFORM__ arg) {
  using namespace vortex::graphics;
  FloatA z[4], r[4], g[4], b[4], a[4], u[4], v[4], dx[4], dy[4];
  cocogfx::ColorARGB tex_color[4], out_color[4];
  for (int i = 0; i < 4; ++i) {
    z[i] = FloatA(0.0f); r[i] = FloatA(1.0f); g[i] = FloatA(1.0f);
    b[i] = FloatA(1.0f); a[i] = FloatA(1.0f); u[i] = FloatA(0.0f); v[i] = FloatA(0.0f);
  }
  auto prim_ptr = reinterpret_cast<rast_prim_t*>(arg->prim_addr);

  vx_rast_begin();
  for (;;) {
    unsigned drained = vx_frag_fetch();
    if (drained) return;
    frag_payload_t p;
    vx_frag_load(p, drained);
    uint32_t pos_mask = p.pos_mask;
    uint32_t pid = p.pid;
    auto& attribs = prim_ptr[pid].attribs;
    GRADIENTS_PL
    if (arg->depth_enabled) { INTERPOLATE(z, attribs.z); }
    if (arg->color_enabled) {
      INTERPOLATE(r, attribs.r); INTERPOLATE(g, attribs.g);
      INTERPOLATE(b, attribs.b); INTERPOLATE(a, attribs.a);
    }
    if (arg->tex_enabled) { INTERPOLATE(u, attribs.u); INTERPOLATE(v, attribs.v); }

    if (arg->tex_enabled) {
      TEXTURING(tex_color, u, v);
      if (arg->tex_modulate) { MODULATE(out_color, r, g, b, a, tex_color); }
      else { REPLACE(out_color, tex_color); }
    } else {
      TO_RGBA(out_color, r, g, b, a);
    }
    OUTPUT_QUAD(pos_mask, 0, out_color, z);
  }
}
