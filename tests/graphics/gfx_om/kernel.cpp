#include <vx_spawn2.h>
#include <vx_graphics.h>
#include "common.h"

using namespace vortex::graphics;

// One thread per output pixel. Each pixel (x,y) is one covered sub-pixel of the
// quad at origin (x>>1, y>>1): the thread stages its colour/depth into that
// fragment to the OM aperture (one store). The OM
// unit blends it into the host-configured cbuf per the DCR-set blend/depth state,
// validated against the golden images.

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

    if (arg->sw_path) {
        // Software output-merger routing: merge this fragment via the LSU
        // (gfx_sw::om_fragment) instead of the FF OM unit. Same depth/blend/ROP
        // math (gfx_sw ops) → bit-exact vs the FF OM and the golden.
        gfx_sw::om_fragment(arg->om, x, y, arg->backface, color, arg->depth);
        return;
    }

    // One fragment, one export: a store to the OM aperture. The old path had to
    // stage this colour into ALL FOUR quad slots and select the live one with a
    // runtime cov_mask, because the window slot was a funct7 immediate while the
    // sub-pixel index is runtime. Addressing a pixel directly makes that whole
    // contortion disappear.
    vx_om_export_both(
        VX_OM_APERTURE_ADDR(arg->aperture_xbits, arg->aperture_ybits,
                            arg->aperture_record_shift, x, y, arg->backface),
        color, arg->depth);
}
