#include <vx_spawn2.h>
#include <vx_graphics.h>
#include <cocogfx/include/color.hpp>
#include <cocogfx/include/math.hpp>
#include "common.h"
#include <pipe_frontend.h>   // setup_k + binning_k (-I gfx_setup_kernel)

// gfx_v2 device front end + fragment-interpolation + OM in one module: the
// shared pipeline (setup_k / binning_k) produces RASTER's tilebuf + primbuf;
// this fragment kernel (gfx_draw3d's) interpolates the primitive's colour from
// the device-produced primbuf (RASTER bcoord CSRs + PID) and writes via OM.

#define BCOORD_AS_FLOAT(csr) \
    static_cast<float>(fixed16_t::make(static_cast<int32_t>(csr_read(csr))))

#define GRADIENTS_HW_i(i) { \
    auto F0 = BCOORD_AS_FLOAT(VX_CSR_RASTER_BCOORD_X##i); \
    auto F1 = BCOORD_AS_FLOAT(VX_CSR_RASTER_BCOORD_Y##i); \
    auto F2 = BCOORD_AS_FLOAT(VX_CSR_RASTER_BCOORD_Z##i); \
    auto recip = 1.0f / (F0 + F1 + F2); \
    dx[i] = FloatA(recip * F0); dy[i] = FloatA(recip * F1); }

#define INTERPOLATE_i(i, dst, src) { \
    auto tmp = src.x * dx[i] + src.z; dst[i] = src.y * dy[i] + tmp; }

#define TO_RGBA_i(i, dst, sr, sg, sb, sa) \
    dst[i].r = static_cast<uint8_t>(sr[i] * 255); \
    dst[i].g = static_cast<uint8_t>(sg[i] * 255); \
    dst[i].b = static_cast<uint8_t>(sb[i] * 255); \
    dst[i].a = static_cast<uint8_t>(sa[i] * 255)

#define OUTPUT_i(i, mask, x, y, face, color, depth) \
    if (mask & (1 << i)) { \
        auto pos_x = (x << 1) + (i & 1); auto pos_y = (y << 1) + (i >> 1); \
        auto pos_z = static_cast<uint32_t>(depth[i] * 65336); \
        vx_om(pos_x, pos_y, face, color[i].value, pos_z); }

#define GRADIENTS_HW   GRADIENTS_HW_i(0) GRADIENTS_HW_i(1) GRADIENTS_HW_i(2) GRADIENTS_HW_i(3)
#define INTERPOLATE(d, s) INTERPOLATE_i(0,d,s); INTERPOLATE_i(1,d,s); INTERPOLATE_i(2,d,s); INTERPOLATE_i(3,d,s)
#define TO_RGBA(d, r, g, b, a) TO_RGBA_i(0,d,r,g,b,a); TO_RGBA_i(1,d,r,g,b,a); TO_RGBA_i(2,d,r,g,b,a); TO_RGBA_i(3,d,r,g,b,a)
#define OUTPUT_QUAD(pos_mask, face, color, depth) \
    auto mask = (pos_mask >> 0) & 0xf; \
    auto x = (pos_mask >> 4) & ((1 << (VX_RASTER_DIM_BITS-1))-1); \
    auto y = (pos_mask >> (4 + (VX_RASTER_DIM_BITS-1))) & ((1 << (VX_RASTER_DIM_BITS-1))-1); \
    OUTPUT_i(0,mask,x,y,face,color,depth) OUTPUT_i(1,mask,x,y,face,color,depth) \
    OUTPUT_i(2,mask,x,y,face,color,depth) OUTPUT_i(3,mask,x,y,face,color,depth)

__kernel void kernel_main(frag_arg_t* __UNIFORM__ arg) {
  using namespace vortex::graphics;
  FloatA z[4], r[4], g[4], b[4], a[4], dx[4], dy[4];
  cocogfx::ColorARGB out_color[4];
  for (int i = 0; i < 4; ++i) {
    z[i] = FloatA(0.0f); r[i] = FloatA(1.0f); g[i] = FloatA(1.0f);
    b[i] = FloatA(1.0f); a[i] = FloatA(1.0f);
  }
  auto prim_ptr = reinterpret_cast<rast_prim_t*>(arg->prim_addr);

  vx_rast_begin();
  for (;;) {
    uint32_t pos_mask = vx_rast();
    if (pos_mask == 0) return;
    uint32_t pid = csr_read(VX_CSR_RASTER_PID);
    auto& attribs = prim_ptr[pid].attribs;
    GRADIENTS_HW
    if (arg->depth_enabled) { INTERPOLATE(z, attribs.z); }
    if (arg->color_enabled) {
      INTERPOLATE(r, attribs.r); INTERPOLATE(g, attribs.g);
      INTERPOLATE(b, attribs.b); INTERPOLATE(a, attribs.a);
    }
    TO_RGBA(out_color, r, g, b, a);
    OUTPUT_QUAD(pos_mask, 0, out_color, z);
  }
}
