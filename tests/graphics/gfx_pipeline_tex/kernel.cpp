#include <vx_spawn2.h>
#include <vx_graphics.h>
#include <cocogfx/include/color.hpp>
#include <cocogfx/include/math.hpp>
#include "common.h"
#include <gfx_frontend_k.h>   // setup_k + binning_k (-I gfx_setup_kernel)


// One texture sample at (u, v, lod=0) on stage 0. u/v/lod ride registers and the
// texel comes back in rd -- TEX does not touch the graphics window (P5-B).
static inline uint32_t tex_sample(unsigned u, unsigned v) {
    return vortex::graphics::vx_tex(0, u, v, 0);
}

// Device front end + fragment (interpolate uv + TEX sample) + OM in one
// module: the shared pipeline produces RASTER's tilebuf + primbuf; this fragment
// kernel is gfx_draw3d's (RASTER bcoord CSRs + PID -> interpolated uv -> vx_tex
// -> OM), exercising the TEX fixed-function unit fed device-side.

using fixeduv_t = vortex::graphics::fixed_t<TEX_FXD_FRAC>;

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

// Depth word for the aperture record.
#define DEPTH_WORD(d) (static_cast<uint32_t>((d).data()))

// Per-corner edge value F_axis recomputed in-shader from the primitive's edge
// coefficients (a*X+b*Y+c in Q15.16, bit-identical to the raster HW bcoord); quad
// origin (qx*2, qy*2), corner i offset (i&1, i>>1). `edges`,`qx`,`qy` bound below.
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
// Depth is a fixed-function screen-space plane Z = A'*X + B'*Y + C' (coeffs in
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
#define MODULATE(d, r, g, b, a, s) MODULATE_i(0,d,r,g,b,a,s); MODULATE_i(1,d,r,g,b,a,s); MODULATE_i(2,d,r,g,b,a,s); MODULATE_i(3,d,r,g,b,a,s)
#define REPLACE(d, s) d[0] = s[0]; d[1] = s[1]; d[2] = s[2]; d[3] = s[3]
#define TO_RGBA(d, r, g, b, a) TO_RGBA_i(0,d,r,g,b,a); TO_RGBA_i(1,d,r,g,b,a); TO_RGBA_i(2,d,r,g,b,a); TO_RGBA_i(3,d,r,g,b,a)
#define TEXTURING(d, u, v) \
    d[0] = tex_sample(fixeduv_t(u[0]).data(), fixeduv_t(v[0]).data()); \
    d[1] = tex_sample(fixeduv_t(u[1]).data(), fixeduv_t(v[1]).data()); \
    d[2] = tex_sample(fixeduv_t(u[2]).data(), fixeduv_t(v[2]).data()); \
    d[3] = tex_sample(fixeduv_t(u[3]).data(), fixeduv_t(v[3]).data())
// Export the quad's covered sub-pixels: each is one vx_om_export -- a store to
// the OM aperture. No window staging, no OM bus. Coverage becomes ordinary SIMT
// predication on cov_mask instead of a mask field riding a dedicated bus.
#define OUTPUT_FRAG(i, pos_mask, face, color, depth) \
    if ((pos_mask) & (1u << (i))) { \
        uint32_t _x = ((uint32_t)qx << 1) | ((i) & 1u); \
        uint32_t _y = ((uint32_t)qy << 1) | (((i) >> 1) & 1u); \
        vx_om_export_both( \
            VX_OM_APERTURE_ADDR(arg->aperture_xbits, arg->aperture_ybits, \
                                arg->aperture_record_shift, _x, _y, (face)), \
            color[i].value, DEPTH_WORD(depth[i])); \
    }

#define OUTPUT_QUAD(pos_mask, face, color, depth) \
    OUTPUT_FRAG(0, pos_mask, face, color, depth) \
    OUTPUT_FRAG(1, pos_mask, face, color, depth) \
    OUTPUT_FRAG(2, pos_mask, face, color, depth) \
    OUTPUT_FRAG(3, pos_mask, face, color, depth)

__kernel void kernel_main(frag_arg_t* __UNIFORM__ arg) {
  using namespace vortex::graphics;
  FloatA z[4], r[4], g[4], b[4], a[4], u[4], v[4], dx[4], dy[4];
  cocogfx::ColorARGB tex_color[4], out_color[4];
  for (int i = 0; i < 4; ++i) {
    z[i] = FloatA(0.0f); r[i] = FloatA(1.0f); g[i] = FloatA(1.0f);
    b[i] = FloatA(1.0f); a[i] = FloatA(1.0f); u[i] = FloatA(0.0f); v[i] = FloatA(0.0f);
  }
  auto prim_ptr = reinterpret_cast<rast_prim_t*>(arg->prim_addr);

  // RASTER dispatch v2 (push): straight-line FS, payload already in the window.
  frag_payload_t p;
  vx_frag_load(p);
  uint32_t pos_mask = p.pos_mask;
  uint32_t pid = p.pid;
  auto& attribs = prim_ptr[pid].attribs;
  auto& edges = prim_ptr[pid].edges;   // recompute edge values from these
  uint32_t qx = (pos_mask >> 4) & ((1u << (VX_RASTER_DIM_BITS - 1)) - 1);
  uint32_t qy = (pos_mask >> (4 + (VX_RASTER_DIM_BITS - 1))) & ((1u << (VX_RASTER_DIM_BITS - 1)) - 1);
  GRADIENTS_PL
  if (arg->depth_enabled) { PLANE_Z(z); }
  // This pipeline test draws screen-aligned (w==1) geometry, so the setup's
  // perspective-premultiplied planes a·(1/w) equal the raw attributes and the
  // 1/w plane is 1 — the attributes are read directly, no perspective divide
  // (see gfx_draw3d/kernel.cpp for the general perspective-correct path).
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
