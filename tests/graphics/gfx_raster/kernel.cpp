#include <vx_spawn2.h>
#include <vx_graphics.h>
#include "common.h"

using namespace vortex::graphics;

const uint32_t out_color = 0xffffffff;

__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
    // Trigger raster fetch.
    vx_rast_begin();

    for (;;) {
        uint32_t pos_mask = vx_rast();
        if (pos_mask == 0) return;  // raster unit drained

        uint32_t mask = (pos_mask >> 0) & 0xf;
        uint32_t x    = (pos_mask >> 4) & ((1u << (VX_RASTER_DIM_BITS - 1)) - 1);
        uint32_t y    = (pos_mask >> (4 + (VX_RASTER_DIM_BITS - 1))) & ((1u << (VX_RASTER_DIM_BITS - 1)) - 1);

        for (uint32_t i = 0; i < 4; ++i) {
            if (mask & (1u << i)) {
                uint32_t px = (x << 1) + (i & 1);
                uint32_t py = (y << 1) + (i >> 1);
                auto dst_ptr = reinterpret_cast<uint32_t*>(
                    arg->cbuf_addr + px * arg->cbuf_stride + py * arg->cbuf_pitch);
                *dst_ptr = out_color;
            }
        }
    }
}
