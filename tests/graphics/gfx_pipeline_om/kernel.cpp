#include <vx_spawn2.h>
#include <vx_graphics.h>
#include <cocogfx/include/color.hpp>
#include <cocogfx/include/math.hpp>
#include "common.h"
#include <gfx_frontend_k.h>   // setup_k + binning_k (-I gfx_setup_kernel)


// Device front end + fragment-interpolation + OM in one module: the
// shared pipeline (setup_k / binning_k) produces RASTER's tilebuf + primbuf;
// this fragment kernel interpolates the primitive's colour from the
// device-produced primbuf and writes it through the OM.

#define INTERPOLATE(dst, src) { \
    auto tmp = src.x * dx + src.z; dst = src.y * dy + tmp; }

#define TO_RGBA(dst, sr, sg, sb, sa) \
    dst.r = static_cast<uint8_t>(sr * 255); \
    dst.g = static_cast<uint8_t>(sg * 255); \
    dst.b = static_cast<uint8_t>(sb * 255); \
    dst.a = static_cast<uint8_t>(sa * 255)

// Depth word: the screen-space plane MAC is a Q7.24 z value; write it saturated
// to the 24-bit zbuf range so edge extrapolation clamps to near / far.
#define DEPTH_WORD(d) \
    ((d).data() < 0 ? 0u \
      : ((uint32_t)(d).data() > (uint32_t)OM_DEPTH_MASK ? (uint32_t)OM_DEPTH_MASK \
                                                           : (uint32_t)(d).data()))

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

__kernel void kernel_main(frag_arg_t* __UNIFORM__ arg) {
  using namespace vortex::graphics;
  FloatA z(0.0f), r(1.0f), g(1.0f), b(1.0f), a(1.0f), dx, dy;
  cocogfx::ColorARGB out_color;
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
  // Perspective divide: colour planes carry a*(1/w) (see gfx_draw3d/kernel.cpp).
  // Recover a by dividing the interpolated a*(1/w) by the interpolated 1/w.
  if (arg->color_enabled) {
    FloatA w_i;
    INTERPOLATE(w_i, attribs.rhw);
    INTERPOLATE(r, attribs.r); INTERPOLATE(g, attribs.g);
    INTERPOLATE(b, attribs.b); INTERPOLATE(a, attribs.a);
    float iw = (w_i.data() != 0) ? (1.0f / static_cast<float>(w_i)) : 0.0f;
    r = FloatA(static_cast<float>(r) * iw); g = FloatA(static_cast<float>(g) * iw);
    b = FloatA(static_cast<float>(b) * iw); a = FloatA(static_cast<float>(a) * iw);
  }
  TO_RGBA(out_color, r, g, b, a);

  // Each covered lane exports one fragment: a store to the OM aperture. No window
  // staging, no OM bus. A lane the primitive misses is a helper — it shaded so its
  // neighbours could take derivatives, and it simply does not export.
  if (vx_frag_covered(p)) {
    vx_om_export_both(
        VX_OM_APERTURE_ADDR(arg->aperture_xbits, arg->aperture_ybits,
                            arg->aperture_record_shift, x, y, 0),
        out_color.value, DEPTH_WORD(z));
  }
}
