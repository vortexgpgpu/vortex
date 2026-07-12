#include <vx_spawn2.h>
#include <vx_graphics.h>
#include <gfx_sw.h>        // gfx_sw::tex_sample_sw — software-sampler path (§5)
#include "common.h"

using namespace vortex::graphics;

// One thread per output pixel. Maps (gx, gy) to (u, v) in fixed-point texture
// coords and samples with vx_tex (u/v/lod in registers, texel in rd), or with the
// software sampler when the SW path is selected. Validated against the gfx_tex g0
// golden image.

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
        // (and the golden) bit-for-bit — same gfx_frag_tex.h math (§7).
        gfx_sw::TexState st{};
        st.base   = arg->tex_addr;
        st.logdim = arg->tex_logdim;
        st.format = arg->tex_format;
        st.filter = arg->tex_filter;
        st.wrap   = arg->tex_wrap;
        for (uint32_t i = 0; i <= (uint32_t)VX_TEX_LOD_MAX; ++i)
            st.mip_off[i] = arg->tex_mipoff[i];
        color = gfx_sw::tex_sample_sw(st, (int32_t)fu, (int32_t)fv, arg->lod);
    } else {
        color = vx_tex(0, fu, fv, arg->lod);
    }

    auto dst_row = reinterpret_cast<uint32_t*>(arg->dst_addr + gy * arg->dst_pitch);
    dst_row[gx] = color;
}
