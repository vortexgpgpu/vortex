#include <vx_spawn2.h>
#include <vx_intrinsics.h>
#include <cocogfx/include/color.hpp>
#include "common.h"

// vortex2 v2 kernel: every thread polls vx_rast() in a loop until the
// raster unit reports done. On each pop, the per-warp
// VX_CSR_RASTER_POS_MASK CSR holds {y, x, mask} for the active quad,
// and we write white to every covered sub-pixel. The cluster-shared
// VX_raster_core fans descriptors out via VX_raster_arb so multiple
// threads racing for vx_rast each get distinct quads (or done=0 when
// the queue drains).

static inline void output_quad(kernel_arg_t* arg, uint32_t pos_mask) {
    const cocogfx::ColorARGB out_color[4] = {0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff};
    uint32_t mask = (pos_mask >> 0) & 0xf;
    uint32_t x    = (pos_mask >> 4) & ((1u << (VX_RASTER_DIM_BITS - 1)) - 1);
    uint32_t y    = (pos_mask >> (4 + (VX_RASTER_DIM_BITS - 1))) & ((1u << (VX_RASTER_DIM_BITS - 1)) - 1);
    for (uint32_t i = 0; i < 4; ++i) {
        if (mask & (1u << i)) {
            uint32_t px = (x << 1) + (i & 1);
            uint32_t py = (y << 1) + (i >> 1);
            auto dst_ptr = reinterpret_cast<uint32_t*>(
                arg->cbuf_addr + px * arg->cbuf_stride + py * arg->cbuf_pitch);
            *dst_ptr = out_color[i].value;
        }
    }
}

__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
    for (;;) {
        uint32_t pos_mask = vx_rast();
        if (pos_mask == 0) return;  // raster unit drained
        output_quad(arg, pos_mask);
    }
}
