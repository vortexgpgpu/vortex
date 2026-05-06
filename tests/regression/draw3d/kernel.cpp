// vortex2 KMU port of the skybox draw3d (full-pipeline) test.
//
// Each thread polls vx_rast() to pop quads from the cluster-shared
// raster_core, looks up vertex attributes for the popped pid, computes
// barycentric-interpolated colour/uv/depth from raster bcoord CSRs,
// optionally samples a texture, and writes the result through vx_om.
//
// vx_rast() returns the pos_mask word directly (matches raster_smoke /
// raster kernels). The per-thread pid + bcoords for the popped quad are
// read back through VX_CSR_RASTER_PID + VX_CSR_RASTER_BCOORD_X/Y/Z[0..3],
// which simx latches into per-warp+thread CSR storage on each pop.

#include <vx_spawn2.h>
#include <vx_intrinsics.h>
#include <cocogfx/include/color.hpp>
#include <cocogfx/include/math.hpp>
#include <graphics.h>
#include "common.h"

using namespace graphics;

using fixeduv_t = cocogfx::TFixed<VX_TEX_FXD_FRAC>;

#define DEFAULTS_i(i) \
    z[i] = FloatA(0.0f); \
    r[i] = FloatA(1.0f); \
    g[i] = FloatA(1.0f); \
    b[i] = FloatA(1.0f); \
    a[i] = FloatA(1.0f); \
    u[i] = FloatA(0.0f); \
    v[i] = FloatA(0.0f)

// CSR-bcoord helper: bcoord CSRs hold Q15.16 fixed-point bits (FloatE
// `.data()` from the simx side / VX_raster_edge multiplier output on RTL).
// Reinterpret via `fixed16_t::make()` then static_cast to float for the
// reciprocal computation.
#define BCOORD_AS_FLOAT(csr_addr) \
    static_cast<float>(fixed16_t::make(static_cast<int32_t>(csr_read(csr_addr))))

#define GRADIENTS_HW_i(i) { \
    auto F0 = BCOORD_AS_FLOAT(VX_CSR_RASTER_BCOORD_X##i); \
    auto F1 = BCOORD_AS_FLOAT(VX_CSR_RASTER_BCOORD_Y##i); \
    auto F2 = BCOORD_AS_FLOAT(VX_CSR_RASTER_BCOORD_Z##i); \
    auto recip = 1.0f / (F0 + F1 + F2); \
    dx[i] = FloatA(recip * F0); \
    dy[i] = FloatA(recip * F1); \
}

#ifdef FIXEDPOINT_RASTERIZER

inline int32_t imadd(int32_t a, int32_t b, int32_t c, int32_t s) {
    int32_t p = ((int64_t)a * (int64_t)b) >> (s << 3);
    return p + c;
}

#define multadd_fx(a, b, c) \
    fixed24_t::make(imadd(a.data(), b.data(), c.data(), 3))

#define INTERPOLATE_i(i, dst, src) { \
    auto tmp = multadd_fx(src.x, dx[i], src.z);  \
    dst[i]   = multadd_fx(src.y, dy[i], tmp);    \
}

#define MODULATE_i(i, dst, s1r, s1g, s1b, s1a, s2) \
    dst[i].r = (s1r[i].data() * s2[i].r) >> fixed24_t::FRAC; \
    dst[i].g = (s1g[i].data() * s2[i].g) >> fixed24_t::FRAC; \
    dst[i].b = (s1b[i].data() * s2[i].b) >> fixed24_t::FRAC; \
    dst[i].a = (s1a[i].data() * s2[i].a) >> fixed24_t::FRAC

#define TO_RGBA_i(i, dst, sr, sg, sb, sa) \
    dst[i].r = static_cast<uint8_t>((sr[i].data() * 255) >> fixed24_t::FRAC); \
    dst[i].g = static_cast<uint8_t>((sg[i].data() * 255) >> fixed24_t::FRAC); \
    dst[i].b = static_cast<uint8_t>((sb[i].data() * 255) >> fixed24_t::FRAC); \
    dst[i].a = static_cast<uint8_t>((sa[i].data() * 255) >> fixed24_t::FRAC)

#define OUTPUT_i(i, mask, x, y, face, color, depth) \
    if (mask & (1 << i)) { \
        auto pos_x = (x << 1) + (i & 1); \
        auto pos_y = (y << 1) + (i >> 1); \
        auto pos_z = depth[i].data(); \
        vx_om(pos_x, pos_y, face, color[i].value, pos_z); \
    }

#else

#define INTERPOLATE_i(i, dst, src) { \
    auto tmp = src.x * dx[i] + src.z;  \
    dst[i]   = src.y * dy[i] + tmp;    \
}

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

#define OUTPUT_i(i, mask, x, y, face, color, depth) \
    if (mask & (1 << i)) { \
        auto pos_x = (x << 1) + (i & 1); \
        auto pos_y = (y << 1) + (i >> 1); \
        auto pos_z = static_cast<uint32_t>(depth[i] * 65336); \
        vx_om(pos_x, pos_y, face, color[i].value, pos_z); \
    }

#endif

#define DEFAULTS \
    DEFAULTS_i(0); DEFAULTS_i(1); DEFAULTS_i(2); DEFAULTS_i(3)

#define GRADIENTS_HW \
    GRADIENTS_HW_i(0) GRADIENTS_HW_i(1) GRADIENTS_HW_i(2) GRADIENTS_HW_i(3)

#define INTERPOLATE(dst, src) \
    INTERPOLATE_i(0, dst, src); INTERPOLATE_i(1, dst, src); \
    INTERPOLATE_i(2, dst, src); INTERPOLATE_i(3, dst, src)

#define MODULATE(dst, s1r, s1g, s1b, s1a, s2) \
    MODULATE_i(0, dst, s1r, s1g, s1b, s1a, s2); \
    MODULATE_i(1, dst, s1r, s1g, s1b, s1a, s2); \
    MODULATE_i(2, dst, s1r, s1g, s1b, s1a, s2); \
    MODULATE_i(3, dst, s1r, s1g, s1b, s1a, s2)

#define REPLACE(dst, src) \
    dst[0] = src[0]; dst[1] = src[1]; dst[2] = src[2]; dst[3] = src[3]

#define TO_RGBA(dst, sr, sg, sb, sa) \
    TO_RGBA_i(0, dst, sr, sg, sb, sa); TO_RGBA_i(1, dst, sr, sg, sb, sa); \
    TO_RGBA_i(2, dst, sr, sg, sb, sa); TO_RGBA_i(3, dst, sr, sg, sb, sa)

#define TEXTURING(dst, u, v) \
    dst[0] = vx_tex(0, fixeduv_t(u[0]).data(), fixeduv_t(v[0]).data(), 0); \
    dst[1] = vx_tex(0, fixeduv_t(u[1]).data(), fixeduv_t(v[1]).data(), 0); \
    dst[2] = vx_tex(0, fixeduv_t(u[2]).data(), fixeduv_t(v[2]).data(), 0); \
    dst[3] = vx_tex(0, fixeduv_t(u[3]).data(), fixeduv_t(v[3]).data(), 0)

#define OUTPUT_QUAD(pos_mask, face, color, depth) \
    auto mask = (pos_mask >> 0) & 0xf; \
    auto x    = (pos_mask >> 4) & ((1 << (VX_RASTER_DIM_BITS-1))-1); \
    auto y    = (pos_mask >> (4 + (VX_RASTER_DIM_BITS-1))) & ((1 << (VX_RASTER_DIM_BITS-1))-1); \
    OUTPUT_i(0, mask, x, y, face, color, depth) \
    OUTPUT_i(1, mask, x, y, face, color, depth) \
    OUTPUT_i(2, mask, x, y, face, color, depth) \
    OUTPUT_i(3, mask, x, y, face, color, depth)

__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
    FloatA z[4], r[4], g[4], b[4], a[4], u[4], v[4];
    FloatA dx[4], dy[4];
    cocogfx::ColorARGB tex_color[4], out_color[4];
    DEFAULTS;

    auto prim_ptr = (rast_prim_t*)arg->prim_addr;

    for (;;) {
        uint32_t pos_mask = vx_rast();
        if (pos_mask == 0) return;          // queue drained
        uint32_t pid = csr_read(VX_CSR_RASTER_PID);
        auto& attribs = prim_ptr[pid].attribs;

        GRADIENTS_HW

        if (arg->depth_enabled) {
            INTERPOLATE(z, attribs.z);
        }
        if (arg->color_enabled) {
            INTERPOLATE(r, attribs.r);
            INTERPOLATE(g, attribs.g);
            INTERPOLATE(b, attribs.b);
            INTERPOLATE(a, attribs.a);
        }
        if (arg->tex_enabled) {
            INTERPOLATE(u, attribs.u);
            INTERPOLATE(v, attribs.v);
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

        OUTPUT_QUAD(pos_mask, 0, out_color, z);
    }
}
