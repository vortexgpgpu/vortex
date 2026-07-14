#include <vx_spawn2.h>
#include <vx_graphics.h>
#include <cocogfx/include/color.hpp>
#include <cocogfx/include/math.hpp>
#include "common.h"
#include <gfx_frontend_k.h>   // setup_k + binning_k (-I gfx_setup_kernel)


// One texture sample at (u, v, lod=0) on stage 0. u/v/lod ride registers and the
// texel comes back in rd -- TEX takes its operands in registers.
static inline uint32_t tex_sample(unsigned u, unsigned v) {
    return vortex::graphics::vx_tex(0, u, v, 0);
}

// Device front end + fragment (interpolate uv + TEX sample) + OM in one
// module: the shared pipeline produces RASTER's tilebuf + primbuf; this fragment
// kernel is gfx_draw3d's (RASTER bcoord CSRs + PID -> interpolated uv -> vx_tex
// -> OM), exercising the TEX fixed-function unit fed device-side.

using fixeduv_t = vortex::graphics::fixed_t<TEX_FXD_FRAC>;

#define INTERPOLATE(dst, src) { \
    auto tmp = src.x * dx + src.z; dst = src.y * dy + tmp; }

#define MODULATE(dst, s1r, s1g, s1b, s1a, s2) \
    dst.r = static_cast<uint8_t>(s1r * s2.r); \
    dst.g = static_cast<uint8_t>(s1g * s2.g); \
    dst.b = static_cast<uint8_t>(s1b * s2.b); \
    dst.a = static_cast<uint8_t>(s1a * s2.a)

#define TO_RGBA(dst, sr, sg, sb, sa) \
    dst.r = static_cast<uint8_t>(sr * 255); \
    dst.g = static_cast<uint8_t>(sg * 255); \
    dst.b = static_cast<uint8_t>(sb * 255); \
    dst.a = static_cast<uint8_t>(sa * 255)

// Depth word for the aperture record.
#define DEPTH_WORD(d) (static_cast<uint32_t>((d).data()))

// Edge value F_axis recomputed in-shader from the primitive's edge coefficients
// (a*X+b*Y+c in Q15.16, bit-identical to the raster HW bcoord). `edges`, `x`, `y`
// are bound in the shader body.
#define BCOORD_PL_AS_FLOAT(axis) \
    static_cast<float>(fixed16_t::make( \
        edges[axis].x.data() * (int32_t)x \
      + edges[axis].y.data() * (int32_t)y \
      + edges[axis].z.data()))
#define GRADIENTS_PL { \
    auto F0 = BCOORD_PL_AS_FLOAT(0); auto F1 = BCOORD_PL_AS_FLOAT(1); \
    auto F2 = BCOORD_PL_AS_FLOAT(2); auto recip = 1.0f / (F0 + F1 + F2); \
    dx = FloatA(recip * F0); dy = FloatA(recip * F1); }
// Depth is a fixed-function screen-space plane Z = A'*X + B'*Y + C' (coeffs in
// attribs.z {x:A', y:B', z:C'}, Q7.24), evaluated by an integer MAC bit-identical
// to the raster early-Z so early-Z and late-Z agree.
#define PLANE_Z(dst) dst = fixed24_t::make((int32_t)( \
      (int64_t)attribs.z.x.data() * (int32_t)x \
    + (int64_t)attribs.z.y.data() * (int32_t)y \
    + (int64_t)attribs.z.z.data()))
#define REPLACE(dst, src) dst = src
#define TEXTURING(dst, u, v) \
    dst = tex_sample(fixeduv_t(u).data(), fixeduv_t(v).data())

__kernel void kernel_main(frag_arg_t* __UNIFORM__ arg) {
  using namespace vortex::graphics;
  FloatA z(0.0f), r(1.0f), g(1.0f), b(1.0f), a(1.0f), u(0.0f), v(0.0f), dx, dy;
  cocogfx::ColorARGB tex_color, out_color;
  auto prim_ptr = reinterpret_cast<rast_prim_t*>(arg->prim_addr);

  // RASTER dispatch (push): straight-line FS, this lane's pixel already in its
  // launch registers.
  frag_payload_t p;
  vx_frag_load(p);
  uint32_t x = vx_frag_x(p);
  uint32_t y = vx_frag_y(p);
  uint32_t pid = p.pid;
  auto& attribs = prim_ptr[pid].attribs;
  auto& edges = prim_ptr[pid].edges;   // recompute edge values from these
  GRADIENTS_PL
  if (arg->depth_enabled) {
    PLANE_Z(z);
  }
  // This pipeline test draws screen-aligned (w==1) geometry, so the setup's
  // perspective-premultiplied planes a*(1/w) equal the raw attributes and the
  // 1/w plane is 1 — the attributes are read directly, no perspective divide
  // (see gfx_draw3d/kernel.cpp for the general perspective-correct path).
  if (arg->color_enabled) {
    INTERPOLATE(r, attribs.r); INTERPOLATE(g, attribs.g);
    INTERPOLATE(b, attribs.b); INTERPOLATE(a, attribs.a);
  }
  if (arg->tex_enabled) {
    INTERPOLATE(u, attribs.u); INTERPOLATE(v, attribs.v);
  }

  if (arg->tex_enabled) {
    TEXTURING(tex_color, u, v);
    if (arg->tex_modulate) {
      MODULATE(out_color, r, g, b, a, tex_color);
    } else {
      REPLACE(out_color, tex_color);
    }
  } else {
    TO_RGBA(out_color, r, g, b, a);
  }

  // Each covered lane exports one fragment: a store to the OM aperture. A lane the
  // primitive misses is a helper — it shaded so its neighbours could take
  // derivatives, and it simply does not export.
  if (vx_frag_covered(p)) {
    vx_om_export_both(
        VX_OM_APERTURE_ADDR(arg->aperture_xbits, arg->aperture_ybits,
                            arg->aperture_record_shift, x, y, 0),
        out_color.value, DEPTH_WORD(z));
  }
}
