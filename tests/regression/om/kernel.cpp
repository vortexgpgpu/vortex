#include <vx_spawn2.h>
#include <vx_intrinsics.h>
#include "common.h"

// vortex2 v2 kernel: one thread per output pixel. Each thread issues
// one vx_om(x, y, face, color, depth) which the OM unit blends into
// the host-configured cbuf according to DCR-set blend/depth state.

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
    vx_om(x, y, arg->backface, color, arg->depth);
}
