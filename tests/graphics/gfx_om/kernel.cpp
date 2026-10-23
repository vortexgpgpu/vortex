#include <vx_spawn2.h>
#include <vx_graphics.h>
#include "common.h"

using namespace vortex::graphics;

// One thread per output pixel. Each pixel (x,y) is one covered sub-pixel of the
// quad at origin (x>>1, y>>1): the thread stages its colour/depth into that
// fragment to the OM aperture (one store). The OM
// unit blends it into the host-configured cbuf per the DCR-set blend/depth state,
// validated against the golden images.

// The fragment colour both entries below export. Shared so the two cannot drift
// apart: the host recomputes this same value to check the second attachment, and
// a divergence would be read as a merge bug rather than a test bug.
inline uint32_t frag_color(const kernel_arg_t* arg, uint32_t x, uint32_t y) {
    uint32_t alpha = arg->blend_enable
                       ? ((y * arg->a_scale_q16) >> 16)
                       : 0xff;
    uint32_t red   = (x * arg->r_scale_q16) >> 16;
    uint32_t green = (y * arg->g_scale_q16) >> 16;
    uint32_t blue  = ((x + y) * arg->b_scale_q16) >> 16;
    return (alpha << 24) | (red << 16) | (green << 8) | blue;
}

__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= arg->dst_width || y >= arg->dst_height) {
        return;
    }

    uint32_t color = frag_color(arg, x, y);

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
        VX_OM_APERTURE_ADDR_PAIR(arg->aperture_xbits, arg->aperture_record_shift, x, y),
        color, VX_OM_DEPTH_WORD(arg->depth, 0, arg->backface));
}

// The same fragment on two colour attachments. A separate entry rather than a
// flag on the one above, so the single-attachment shader keeps the exact
// instruction sequence its perf baseline was recorded against.
//
// Both exports carry the same two words: the record shape is a property of the
// aperture, not of the attachment, and the depth word is inert because the
// attachments share one depth buffer and the host leaves its writemask clear.
__kernel void kernel_main_mrt(kernel_arg_t* __UNIFORM__ arg) {
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= arg->dst_width || y >= arg->dst_height) {
        return;
    }

    uint32_t color = frag_color(arg, x, y);

    const uint32_t addr =
        VX_OM_APERTURE_ADDR_PAIR(arg->aperture_xbits, arg->aperture_record_shift, x, y);
    vx_om_export_both(addr, color, VX_OM_DEPTH_WORD(arg->depth, 0, arg->backface));
    vx_om_export_both(addr, color, VX_OM_DEPTH_WORD(arg->depth, arg->rt2, arg->backface));
}

// A float depth target: the depth is the whole word, so the face and the
// attachment ride the export address. Separate entries for the same reason as
// kernel_main_mrt. Under -S the software merger reads the key as is.
__kernel void kernel_main_float(kernel_arg_t* __UNIFORM__ arg) {
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= arg->dst_width || y >= arg->dst_height) {
        return;
    }

    uint32_t color = frag_color(arg, x, y);

    if (arg->sw_path) {
        gfx_sw::om_fragment(arg->om, x, y, arg->backface, color, arg->depth);
        return;
    }
    vx_om_export_both(
        VX_OM_APERTURE_ADDR_FLOAT_PAIR(arg->aperture_xbits, arg->aperture_ybits,
                                       arg->aperture_record_shift, x, y, arg->backface, 0),
        color, arg->depth);
}

__kernel void kernel_main_mrt_float(kernel_arg_t* __UNIFORM__ arg) {
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= arg->dst_width || y >= arg->dst_height) {
        return;
    }

    uint32_t color = frag_color(arg, x, y);

    vx_om_export_both(
        VX_OM_APERTURE_ADDR_FLOAT_PAIR(arg->aperture_xbits, arg->aperture_ybits,
                                       arg->aperture_record_shift, x, y, arg->backface, 0),
        color, arg->depth);
    vx_om_export_both(
        VX_OM_APERTURE_ADDR_FLOAT_PAIR(arg->aperture_xbits, arg->aperture_ybits,
                                       arg->aperture_record_shift, x, y, arg->backface, arg->rt2),
        color, arg->depth);
}

// One fragment per pixel, exported once for each sample it covers, with that
// sample's depth: under 4x the samples msaa_frag_mask names, at one sample the
// pixel. A separate entry for the same reason as kernel_main_mrt.
__kernel void kernel_main_msaa(kernel_arg_t* __UNIFORM__ arg) {
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= arg->dst_width || y >= arg->dst_height) {
        return;
    }

    const uint32_t color = frag_color(arg, x, y);
    const bool     ms    = arg->samples > 1;
    const uint32_t mask  = ms ? msaa_frag_mask(x, y) : 1u;
    const uint32_t xb = arg->aperture_xbits, yb = arg->aperture_ybits;
    const uint32_t sh = arg->aperture_record_shift;
    for (uint32_t k = 0; k < 4; ++k) {
        if (!((mask >> k) & 1)) {
            continue;
        }
        // A float target stores the depth's ordered key.
        uint32_t depth = msaa_frag_depth(x, y, k);
        if (arg->depth_float) {
            float f = msaa_frag_depth_f(x, y, k);
            uint32_t bits;
            __builtin_memcpy(&bits, &f, 4);
            depth = gfx_sw::om_float_depth_key(bits);
        }
        if (arg->depth_only) {
            vx_om_export_depth(ms ? VX_OM_APERTURE_ADDR_SAMPLE(xb, yb, sh, x, y, arg->backface, k)
                                  : VX_OM_APERTURE_ADDR(xb, yb, sh, x, y, arg->backface),
                               depth);
        } else if (arg->depth_float) {
            vx_om_export_both(
                ms ? VX_OM_APERTURE_ADDR_FLOAT_PAIR_SAMPLE(xb, yb, sh, x, y, arg->backface, 0, k)
                   : VX_OM_APERTURE_ADDR_FLOAT_PAIR(xb, yb, sh, x, y, arg->backface, 0),
                color, depth);
        } else {
            const uint32_t addr = VX_OM_APERTURE_ADDR_PAIR(xb, sh, x, y);
            vx_om_export_both(addr, color,
                VX_OM_DEPTH_WORD_SAMPLE(msaa_frag_depth(x, y, k), 0, arg->backface, k));
            if (arg->mrt) {
                vx_om_export_both(addr, color,
                    VX_OM_DEPTH_WORD_SAMPLE(msaa_frag_depth(x, y, k), arg->rt2, arg->backface, k));
            }
        }
    }
}
