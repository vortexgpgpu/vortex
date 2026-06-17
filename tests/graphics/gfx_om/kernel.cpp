#include <vx_spawn2.h>
#include <vx_graphics.h>
#include <vx_raytrace.h>   // vx_rt_set (SETW) — stage the OM payload window
#include "common.h"

using namespace vortex::graphics;

// One thread per output pixel. Each pixel (x,y) is one covered sub-pixel of the
// quad at origin (x>>1, y>>1): the thread stages its colour/depth into that
// sub-pixel's window slot and issues vx_om4 with a 1-bit coverage mask. The OM
// unit blends it into the host-configured cbuf per the DCR-set blend/depth state
// — identical OM math to the legacy vx_om, validated against the same images.
static const unsigned OM_WIN = 0;   // window slots 0..7 (colour[0..3], depth[0..3])

__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= arg->dst_width || y >= arg->dst_height) return;

    uint32_t alpha = arg->blend_enable
                       ? ((y * arg->a_scale_q16) >> 16)
                       : 0xff;
    uint32_t red   = (x * arg->r_scale_q16) >> 16;
    uint32_t green = (y * arg->g_scale_q16) >> 16;
    uint32_t blue  = ((x + y) * arg->b_scale_q16) >> 16;

    uint32_t color = (alpha << 24) | (red << 16) | (green << 8) | blue;

    // The window slot is a funct7 immediate (compile-time), but this pixel's
    // sub-pixel index s is runtime — so stage the colour/depth into all four
    // quad slots (constant indices) and let the runtime cov_mask=1<<s in the
    // descriptor select the covered sub-pixel. The OM reads only slot s.
    uint32_t s = ((y & 1) << 1) | (x & 1);
    vx_rt_set(OM_WIN + 0, color);     vx_rt_set(OM_WIN + 1, color);
    vx_rt_set(OM_WIN + 2, color);     vx_rt_set(OM_WIN + 3, color);
    vx_rt_set(OM_WIN + 4, arg->depth); vx_rt_set(OM_WIN + 5, arg->depth);
    vx_rt_set(OM_WIN + 6, arg->depth); vx_rt_set(OM_WIN + 7, arg->depth);

    uint32_t desc = (1u << s)                          // cov_mask
                  | ((x >> 1) << 4)                    // quad x  @ [4 +: 14]
                  | ((y >> 1) << (4 + (VX_RASTER_DIM_BITS - 1)))  // quad y @ [18 +: 13]
                  | (arg->backface << 31);             // face
    vx_om4(desc, OM_WIN);
}
