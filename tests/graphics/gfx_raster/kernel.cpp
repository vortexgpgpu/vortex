#include <vx_spawn2.h>
#include <vx_graphics.h>
#include "common.h"

using namespace vortex::graphics;

const uint32_t out_color = 0xffffffff;

// RASTER dispatch v2 (FWD): persistent fragment worker. vx_frag_fetch pops the
// next covered-quad wave from the single-owner raster producer and stages this
// lane's frag_payload_t into the gfx window; only pos_mask is used here (pure
// coverage write, no shading).
__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
    vx_rast_begin();  // arm the producer (idempotent across workers)

    for (;;) {
        unsigned drained = vx_frag_fetch();
        if (drained) return;  // producer drained → worker exits

        uint32_t pos_mask = vx_frag_payload(0, drained);
        if (pos_mask == 0) continue;

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
