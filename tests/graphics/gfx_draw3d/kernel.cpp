// Full-pipeline draw3d kernel: RASTER dispatch, TEX sampling, OM export.
//
// Each of the three stages runs on its fixed-function unit when that unit is in
// the build, and on its gfx_sw software emitter when it is not (NO_TEX / NO_OM /
// NO_RASTER). The routing is keyed off VX_CFG_EXT_*_ENABLED so a stage can never
// issue an FF instruction on a device that has no FF unit, and every mix — down
// to all-software — renders against the same golden image.
//
// The shader body (shade_quad) is common to both routings: it looks up the
// primitive's planes by pid, recomputes the per-corner edge values from the quad
// origin, interpolates colour/uv/depth, optionally samples a texture, and exports
// the covered sub-pixels.

#include <vx_spawn2.h>
#include <vx_graphics.h>
#include <gfx_sw.h>             // gfx_sw::tex_sample_sw / om_fragment — SW TEX/OM
#if !VX_CFG_EXT_RASTER_ENABLED
#include <gfx_frag_rast.h>      // gfx_rast::rast_walk_primitive — SW fine-rasterizer
#endif
#include <cocogfx/include/color.hpp>
#include <cocogfx/include/math.hpp>
#include "common.h"

using namespace vortex::graphics;


using fixeduv_t = vortex::graphics::fixed_t<TEX_FXD_FRAC>;

// One texture sample at (u, v, lod=0) on stage 0.
//
// FF: u/v/lod ride registers and the texel comes back in rd -- TEX does not touch
// registers. SW: the same sampler math through the LSU, which is
// bit-identical to the unit (gfx_frag_tex.h is shared by both).
static inline uint32_t tex_sample(const kernel_arg_t* arg, unsigned u, unsigned v) {
#if VX_CFG_EXT_TEX_ENABLED
    (void)arg;
    return vx_tex(0, u, v, 0);
#else
    return gfx_sw::tex_sample_sw(arg->tex, (int32_t)u, (int32_t)v, 0);
#endif
}

// One fragment out. FF: a store to the OM aperture, which the cluster's OM steer
// peels off the L1->L2 trunk. SW: the same depth/blend/ROP merge done in-thread
// through the LSU.
static inline void export_frag(const kernel_arg_t* arg, uint32_t x, uint32_t y,
                               uint32_t face, uint32_t color, uint32_t depth) {
#if VX_CFG_EXT_OM_ENABLED
    vx_om_export_both(
        VX_OM_APERTURE_ADDR(arg->aperture_xbits, arg->aperture_ybits,
                            arg->aperture_record_shift, x, y, face),
        color, depth);
#else
    gfx_sw::om_fragment(arg->om, x, y, face, color, depth);
#endif
}

#define DEFAULTS \
    z = FloatA(0.0f); \
    r = FloatA(1.0f); \
    g = FloatA(1.0f); \
    b = FloatA(1.0f); \
    a = FloatA(1.0f); \
    u = FloatA(0.0f); \
    v = FloatA(0.0f)

#ifdef FIXEDPOINT_RASTERIZER

inline int32_t imadd(int32_t a, int32_t b, int32_t c, int32_t s) {
    int32_t p = ((int64_t)a * (int64_t)b) >> (s << 3);
    return p + c;
}

#define multadd_fx(a, b, c) \
    fixed24_t::make(imadd(a.data(), b.data(), c.data(), 3))

#define INTERPOLATE(dst, src) { \
    auto tmp = multadd_fx(src.x, dx, src.z);  \
    dst      = multadd_fx(src.y, dy, tmp);    \
}

#define MODULATE(dst, s1r, s1g, s1b, s1a, s2) \
    dst.r = (s1r.data() * s2.r) >> fixed24_t::FRAC; \
    dst.g = (s1g.data() * s2.g) >> fixed24_t::FRAC; \
    dst.b = (s1b.data() * s2.b) >> fixed24_t::FRAC; \
    dst.a = (s1a.data() * s2.a) >> fixed24_t::FRAC

#define TO_RGBA(dst, sr, sg, sb, sa) \
    dst.r = static_cast<uint8_t>((sr.data() * 255) >> fixed24_t::FRAC); \
    dst.g = static_cast<uint8_t>((sg.data() * 255) >> fixed24_t::FRAC); \
    dst.b = static_cast<uint8_t>((sb.data() * 255) >> fixed24_t::FRAC); \
    dst.a = static_cast<uint8_t>((sa.data() * 255) >> fixed24_t::FRAC)

// Depth word for the aperture record (this branch's z is already in range).
#define DEPTH_WORD(d) ((uint32_t)(d).data())

#else

#define INTERPOLATE(dst, src) { \
    auto tmp = src.x * dx + src.z;  \
    dst      = src.y * dy + tmp;    \
}

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

// Depth word: the screen-space plane MAC is a Q7.24 z value; write it saturated
// to the 24-bit zbuf range. Interior z in [0,1) maps through unchanged; edge
// extrapolation below 0 / above 1 clamps to near / far, keeping depth monotonic.
#define DEPTH_WORD(d) \
    ((d).data() < 0 ? 0u \
      : ((uint32_t)(d).data() > (uint32_t)OM_DEPTH_MASK ? (uint32_t)OM_DEPTH_MASK \
                                                           : (uint32_t)(d).data()))
#endif

// The edge value F_axis is recomputed in the shader from the primitive's edge
// coefficients instead of carried in the payload. F = a*X + b*Y + c in Q15.16 is
// bit-identical to the raster HW's bcoord (the HW evaluates the same edges at the
// same absolute pixel). `edges`, `x`, `y` are bound in the shader body.
#define BCOORD_PL_AS_FLOAT(axis) \
    static_cast<float>(fixed16_t::make( \
        edges[axis].x.data() * (int32_t)x \
      + edges[axis].y.data() * (int32_t)y \
      + edges[axis].z.data()))

#define GRADIENTS_PL { \
    auto F0 = BCOORD_PL_AS_FLOAT(0); \
    auto F1 = BCOORD_PL_AS_FLOAT(1); \
    auto F2 = BCOORD_PL_AS_FLOAT(2); \
    auto recip = 1.0f / (F0 + F1 + F2); \
    dx = FloatA(recip * F0); \
    dy = FloatA(recip * F1); \
}

// Depth is a fixed-function screen-space plane Z = A'*X + B'*Y + C' (coeffs
// in attribs.z as {x:A', y:B', z:C'}, Q7.24). The integer MAC is bit-identical
// to the raster early-Z (and the SW reference), so early-Z and late-Z agree.
#define PLANE_Z(dst) dst = fixed24_t::make((int32_t)( \
      (int64_t)attribs.z.x.data() * (int32_t)x \
    + (int64_t)attribs.z.y.data() * (int32_t)y \
    + (int64_t)attribs.z.z.data()))

#define REPLACE(dst, src) dst = src

#define TEXTURING(dst, u, v) \
    dst = tex_sample(arg, fixeduv_t(u).data(), fixeduv_t(v).data())

// Shade this lane's pixel: interpolate colour/uv/depth from the primitive's
// planes, optionally sample a texture, and export. The body is identical on both
// routings; only tex_sample() and export_frag() differ, and they resolve at
// compile time.
//
// A lane the primitive misses still runs the whole shader — it is a helper, there
// so its covered neighbours in the quad can shuffle a value out of it for a
// derivative — and `covered` is what withholds its export.
static void shade_fragment(const kernel_arg_t* arg, uint32_t x, uint32_t y,
                           uint32_t covered, uint32_t pid) {
    FloatA z, r, g, b, a, u, v;
    FloatA dx, dy;
    cocogfx::ColorARGB tex_color, out_color;
    DEFAULTS;

    auto prim_ptr = (rast_prim_t*)arg->prim_addr;
    auto& attribs = prim_ptr[pid].attribs;
    auto& edges = prim_ptr[pid].edges;

    GRADIENTS_PL

    if (arg->depth_enabled) {
        PLANE_Z(z);
    }
    // Perspective divide: the colour/uv planes carry a*(1/w); recover the
    // attribute by dividing the affinely-interpolated a*(1/w) by the affinely-
    // interpolated 1/w. Float divide is bit-exact SimX<->RTL (same path as
    // GRADIENTS_PL).
    float inv_w = 1.0f;
    if (arg->color_enabled || arg->tex_enabled) {
        FloatA w_i;
        INTERPOLATE(w_i, attribs.rhw);
        // Guard the perspective divide: when interpolated 1/w underflows to ~0
        // (a near-plane / w->0 fragment) clamp the divisor to a tiny epsilon so
        // the recovered attribute degrades gracefully instead of collapsing to 0
        // (black / uv 0). Preserve sign so the reciprocal stays well-defined.
        const float kMinRhw = 1e-8f;
        float rhw = static_cast<float>(w_i);
        if (rhw < kMinRhw && rhw > -kMinRhw) {
            rhw = (rhw < 0.0f) ? -kMinRhw : kMinRhw;
        }
        inv_w = 1.0f / rhw;
    }
    if (arg->color_enabled) {
        INTERPOLATE(r, attribs.r);
        INTERPOLATE(g, attribs.g);
        INTERPOLATE(b, attribs.b);
        INTERPOLATE(a, attribs.a);
        r = FloatA(static_cast<float>(r) * inv_w);
        g = FloatA(static_cast<float>(g) * inv_w);
        b = FloatA(static_cast<float>(b) * inv_w);
        a = FloatA(static_cast<float>(a) * inv_w);
    }
    if (arg->tex_enabled) {
        INTERPOLATE(u, attribs.u);
        INTERPOLATE(v, attribs.v);
        u = FloatA(static_cast<float>(u) * inv_w);
        v = FloatA(static_cast<float>(v) * inv_w);
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

    if (covered) {
        export_frag(arg, x, y, 0, out_color.value, DEPTH_WORD(z));
    }
}

#if VX_CFG_EXT_RASTER_ENABLED

// RASTER dispatch (push) — straight-line fragment shader. The raster engine's work
// distributor launches this kernel once per packed fragment wave, with this lane's
// stamp already landed in the warp's launch registers, so the shader just reads its
// own pixel and shades it. No worker loop, no pull op.
__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
    frag_payload_t p;
    vx_frag_load(p);
    shade_fragment(arg, vx_frag_x(p), vx_frag_y(p), vx_frag_covered(p), p.pid);
}

#else

// Software fine-rasterizer: with no RASTER unit there is no work distributor, so
// the host launches an ordinary grid and the threads walk the screen themselves
// with the same coverage core the FF model uses (gfx_rast::rast_walk_primitive),
// shading every quad the walk emits. Coverage is binning-independent, so the image
// matches the FF golden bit-for-bit.
//
// A quad's four pixels must land on four adjacent lanes here too, so a group of
// VX_FRAG_QUAD_LANES lanes shares one primitive and walks it in lockstep (same
// edges, same tiles, hence uniform control flow across the group); each lane then
// shades its own sub-pixel of every quad the walk emits.
__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
    // A warp must hold whole quad groups, or a group straddles two warps and the
    // derivative SHFL silently reads outside it.
    static_assert(VX_CFG_NUM_THREADS >= VX_FRAG_QUAD_LANES
               && (VX_CFG_NUM_THREADS % VX_FRAG_QUAD_LANES) == 0,
                  "a pixel quad occupies four adjacent lanes, so a warp must hold whole quads");
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t pid = gid / VX_FRAG_QUAD_LANES;
    uint32_t sub = gid % VX_FRAG_QUAD_LANES;
    if (pid >= arg->num_prims) return;

    const rast_prim_t* prim = reinterpret_cast<const rast_prim_t*>(
        (uintptr_t)(arg->prim_addr + (uint64_t)pid * sizeof(rast_prim_t)));

    uint32_t tile = 1u << arg->tile_logsize;
    gfx_rast::RastConfig cfg{ arg->tile_logsize, 0, 0, arg->dst_width, arg->dst_height };

    for (uint32_t ty = 0; ty < arg->dst_height; ty += tile) {
        for (uint32_t tx = 0; tx < arg->dst_width; tx += tile) {
            gfx_rast::rast_walk_primitive(cfg, tx, ty, pid, prim->edges,
                [&](uint32_t pos_mask, const vec3e_t*, uint32_t) {
                    uint32_t qx = (pos_mask >> 4) & ((1u << (VX_RASTER_DIM_BITS - 1)) - 1);
                    uint32_t qy = (pos_mask >> (4 + (VX_RASTER_DIM_BITS - 1)))
                                & ((1u << (VX_RASTER_DIM_BITS - 1)) - 1);
                    shade_fragment(arg, (qx << 1) | (sub & 1), (qy << 1) | (sub >> 1),
                                   (pos_mask >> sub) & 1, pid);
                });
        }
    }
}

#endif
