#include <vx_spawn2.h>
#include <vx_graphics.h>
#include <vx_raytrace.h>   // vx_rt_set (SETW) / vx_rt_get_after (handle-chained GETW)
#include <vx_tex_lod.h>    // vx_tex_quad_lod — the shared HW-LOD formula
#include "common.h"

using namespace vortex::graphics;

// Quad self-check: each thread owns a 2x2 quad of texture coords (the pixel + its
// right/down/diagonal neighbours, one output-pixel uv step apart). vx_tex4_quad
// computes ONE integer mip LOD from the quad derivatives and samples the four
// fragments at that LOD into the window. The kernel independently computes the
// same LOD with vx_tex_quad_lod and samples each fragment with the legacy vx_tex
// at that LOD, then XORs the two — every output word must be 0, proving the HW
// LOD + quad path == vx_tex per fragment, on both SimX and rtlsim.
//
// Window scratch (no ray tracing here): u[0..3]@0..3, v[0..3]@4..7, texel[0..3]@8..11.
static const unsigned IN_SLOT  = 0;
static const unsigned OUT_SLOT = 8;

__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
    uint32_t gx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t gy = blockIdx.y * blockDim.y + threadIdx.y;
    if (gx >= arg->dst_width || gy >= arg->dst_height) return;

    uint32_t fu = (arg->deltaX >> 1) + arg->deltaX * gx;
    uint32_t fv = (arg->deltaY >> 1) + arg->deltaY * gy;
    int32_t u[4] = { (int32_t)fu, (int32_t)(fu + arg->deltaX), (int32_t)fu,                 (int32_t)(fu + arg->deltaX) };
    int32_t v[4] = { (int32_t)fv, (int32_t)fv,                 (int32_t)(fv + arg->deltaY), (int32_t)(fv + arg->deltaY) };

    for (int F = 0; F < 4; ++F) {
        vx_rt_set(IN_SLOT + F,     (unsigned)u[F]);
        vx_rt_set(IN_SLOT + 4 + F, (unsigned)v[F]);
    }
    unsigned handle = vx_tex4_quad(0, arg->logw, arg->logh, IN_SLOT, OUT_SLOT);

    uint32_t sw_lod = vx_tex_quad_lod(u, v, arg->logw, arg->logh);
    uint32_t diff = 0;
    for (int F = 0; F < 4; ++F) {
        uint32_t got = vx_rt_get_after(OUT_SLOT + F, handle);       // texel from the window
        uint32_t ref = vx_tex(0, (unsigned)u[F], (unsigned)v[F], sw_lod);
        diff |= (got ^ ref);
    }

    auto dst_row = reinterpret_cast<uint32_t*>(arg->dst_addr + gy * arg->dst_pitch);
    dst_row[gx] = diff;   // 0 iff vx_tex4_quad == vx_tex at the hardware LOD
}
