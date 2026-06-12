#include <vx_spawn2.h>
#include <vx_graphics.h>
#include <vx_raytrace.h>   // vx_rt_set (SETW) / vx_rt_get_after (handle-chained GETW)
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

    vx_rt_set(IN_SLOT,     fu);
    vx_rt_set(IN_SLOT + 1, fv);
    uint32_t handle = vx_tex4_single(0, arg->lod, IN_SLOT, OUT_SLOT);
    uint32_t color  = vx_rt_get_after(OUT_SLOT, handle);

    auto dst_row = reinterpret_cast<uint32_t*>(arg->dst_addr + gy * arg->dst_pitch);
    dst_row[gx] = color;
}
