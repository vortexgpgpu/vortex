// Full-pipeline draw3d kernel (RASTER dispatch v2 / FWD).
//
// Persistent fragment worker: each warp arms the producer (idempotent), then
// loops calling vx_rast_fetch, which stages the next covered-quad wave's per-lane
// frag_payload_t (pos_mask + pid + bcoords) into this warp's own LMEM band and
// returns a drained flag. The worker looks up vertex attributes for the popped
// pid, computes barycentric-interpolated colour/uv/depth from the payload
// bcoords, optionally samples a texture, and writes the result through vx_om4.

#include <vx_spawn2.h>
#include <vx_graphics.h>
#include <vx_raytrace.h>   // vx_gfx_set (SETW) — stage the vx_om4 payload window
#include <cocogfx/include/color.hpp>
#include <cocogfx/include/math.hpp>
#include "common.h"

using namespace vortex::graphics;

// vx_om4 payload window: slots 0..3 = colour[0..3], 4..7 = depth[0..3].
static const unsigned OM_WIN = 0;

using fixeduv_t = vortex::graphics::fixed_t<VX_TEX_FXD_FRAC>;

#define DEFAULTS_i(i) \
    z[i] = FloatA(0.0f); \
    r[i] = FloatA(1.0f); \
    g[i] = FloatA(1.0f); \
    b[i] = FloatA(1.0f); \
    a[i] = FloatA(1.0f); \
    u[i] = FloatA(0.0f); \
    v[i] = FloatA(0.0f)

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

#define STAGE_i(i, color, depth) \
    vx_gfx_set(OM_WIN + (i),     color[i].value); \
    vx_gfx_set(OM_WIN + 4 + (i), depth[i].data())

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

#define STAGE_i(i, color, depth) \
    vx_gfx_set(OM_WIN + (i),     color[i].value); \
    vx_gfx_set(OM_WIN + 4 + (i), static_cast<uint32_t>(depth[i] * 65336))

#endif

#define DEFAULTS \
    DEFAULTS_i(0); DEFAULTS_i(1); DEFAULTS_i(2); DEFAULTS_i(3)

// bcoords come from the per-lane window payload (`p`) staged by vx_rast_fetch.
// p.bcoord[axis][corner] holds the raw Q15.16 bit pattern.
#define BCOORD_PL_AS_FLOAT(axis, i) \
    static_cast<float>(fixed16_t::make(static_cast<int32_t>(p.bcoord[axis][i])))

#define GRADIENTS_PL_i(i) { \
    auto F0 = BCOORD_PL_AS_FLOAT(0, i); \
    auto F1 = BCOORD_PL_AS_FLOAT(1, i); \
    auto F2 = BCOORD_PL_AS_FLOAT(2, i); \
    auto recip = 1.0f / (F0 + F1 + F2); \
    dx[i] = FloatA(recip * F0); \
    dy[i] = FloatA(recip * F1); \
}

#define GRADIENTS_PL \
    GRADIENTS_PL_i(0) GRADIENTS_PL_i(1) GRADIENTS_PL_i(2) GRADIENTS_PL_i(3)

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

// Stage the quad's four colours/depths into the window (uncovered sub-pixels are
// masked off by cov_mask in the descriptor) and submit one vx_om4. rs1 = the
// raster pos_mask (cov_mask + quad origin) with face in bit 31.
#define OUTPUT_QUAD(pos_mask, face, color, depth) \
    STAGE_i(0, color, depth); STAGE_i(1, color, depth); \
    STAGE_i(2, color, depth); STAGE_i(3, color, depth); \
    vx_om4((pos_mask) | ((unsigned)(face) << 31), OM_WIN)

// RASTER dispatch v2 (FWD) — persistent fragment worker (self-pull). Host launches
// a normal fragment grid; each warp arms the producer (idempotent) then loops:
// vx_rast_fetch stages the next covered-quad wave's per-lane frag_payload_t into
// this warp's own LMEM and returns a drained flag. No bcoord CSRs, no pos_mask
// sentinel — doctrine-clean single-issue scoreboarded handoff from a single-owner
// producer.
__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
    FloatA z[4], r[4], g[4], b[4], a[4], u[4], v[4];
    FloatA dx[4], dy[4];
    cocogfx::ColorARGB tex_color[4], out_color[4];
    DEFAULTS;

    auto prim_ptr = (rast_prim_t*)arg->prim_addr;

    vx_rast_begin();  // arm the producer (idempotent across workers)

    for (;;) {
        unsigned drained = vx_rast_fetch();
        if (drained) return;            // producer drained → worker exits

        // This lane's quad, staged by the op into the gfx window (FWD-5).
        frag_payload_t p;
        vx_frag_load(p, drained);
        uint32_t pos_mask = p.pos_mask;
        uint32_t pid = p.pid;
        auto& attribs = prim_ptr[pid].attribs;

        GRADIENTS_PL

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

        OUTPUT_QUAD(pos_mask, 0, out_color, z);  // pos_mask=0 lanes are masked off
    }
}
