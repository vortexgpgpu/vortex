#include <vx_spawn2.h>
#include <vx_graphics.h>
#include <vx_raytrace.h>   // vx_gfx_set (SETW) / vx_gfx_get_after (handle-chained GETW)
#include <gfx_sw.h>        // gfx_sw::tex_sample_sw — software-sampler path (§5)
#include "common.h"

using namespace vortex::graphics;

// One thread per output pixel. Maps (gx, gy) to (u, v) in fixed-point texture
// coords and samples via vx_tex4 on the shared graphics window: stage u,v into
// the window with SETW, issue vx_tex4 (which reads u,v from the window and lands
// the texel back in the window output slot + rd handle), then read the texel
// from the window with a handle-chained GETW. The result must equal the legacy
// vx_tex point sample — validated against the gfx_tex g0 golden image.
//
// Window scratch slots (free here — no ray tracing in this kernel): u@0, v@1,
// texel@4.
static const unsigned IN_SLOT  = 0;
static const unsigned OUT_SLOT = 4;

__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
    uint32_t gx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t gy = blockIdx.y * blockDim.y + threadIdx.y;
    if (gx >= arg->dst_width || gy >= arg->dst_height) return;

    uint32_t fu = (arg->deltaX >> 1) + arg->deltaX * gx;
    uint32_t fv = (arg->deltaY >> 1) + arg->deltaY * gy;

    uint32_t color;
    if (arg->sw_path) {
        // §5 software-sampler routing: sample the resident texture via the LSU
        // (gfx_sw::tex_sample_sw) instead of the FF TEX unit. Must match vx_tex4
        // (and the golden) bit-for-bit — same tex_sample.h math (§7).
        gfx_sw::TexState st{};
        st.base   = arg->tex_addr;
        st.logdim = arg->tex_logdim;
        st.format = arg->tex_format;
        st.filter = arg->tex_filter;
        st.wrap   = arg->tex_wrap;
        for (uint32_t i = 0; i <= (uint32_t)VX_TEX_LOD_MAX; ++i)
            st.mip_off[i] = arg->tex_mipoff[i];
        // Trilinear consumes a fixed-point LOD; point/bilinear an integer LOD.
        uint32_t lod = (arg->tex_filter & VX_TEX_FILTER_MIP_LINEAR)
                     ? ((arg->lod << VX_TEX_LOD_FRAC_BITS) | (arg->frac & 0xff))
                     : arg->lod;
        color = gfx_sw::tex_sample_sw(st, (int32_t)fu, (int32_t)fv, lod);
    } else {
        vx_gfx_set(IN_SLOT,     fu);
        vx_gfx_set(IN_SLOT + 1, fv);
        uint32_t handle = vx_tex4_single(0, arg->lod, IN_SLOT, OUT_SLOT);
        color           = vx_gfx_get_after(OUT_SLOT, handle);
    }

    auto dst_row = reinterpret_cast<uint32_t*>(arg->dst_addr + gy * arg->dst_pitch);
    dst_row[gx] = color;
}
