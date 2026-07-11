#include <vx_spawn2.h>
#include <vx_graphics.h>
#include <gfx_frag_rast.h>      // gfx_rast::rast_walk_primitive — software raster path (§5)
#include "common.h"

using namespace vortex::graphics;

const uint32_t out_color = 0xffffffff;

// §5 software fine-rasterizer: one thread per resident primitive walks the
// screen with gfx_rast::rast_walk_primitive (the same coverage core the FF model
// uses) and writes out_color at every covered sub-pixel — no FF RASTER. Coverage
// is binning-independent, so the image matches the FF RASTER golden (§7).
static void sw_raster_main(kernel_arg_t* arg) {
    uint32_t pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= arg->num_prims) return;

    const rast_prim_t* prim =
        reinterpret_cast<const rast_prim_t*>(
            (uintptr_t)(arg->prim_addr + (uint64_t)pid * sizeof(rast_prim_t)));

    uint32_t tile = 1u << arg->tile_logsize;
    gfx_rast::RastConfig cfg{ arg->tile_logsize, 0, 0, arg->dst_width, arg->dst_height };

    for (uint32_t ty = 0; ty < arg->dst_height; ty += tile) {
        for (uint32_t tx = 0; tx < arg->dst_width; tx += tile) {
            gfx_rast::rast_walk_primitive(cfg, tx, ty, pid, prim->edges,
                [&](uint32_t pos_mask, const vec3e_t*, uint32_t) {
                    uint32_t mask = pos_mask & 0xf;
                    uint32_t qx = (pos_mask >> 4) & ((1u << (VX_RASTER_DIM_BITS - 1)) - 1);
                    uint32_t qy =  pos_mask >> (4 + (VX_RASTER_DIM_BITS - 1));
                    for (uint32_t i = 0; i < 4; ++i) {
                        if (!(mask & (1u << i))) continue;
                        uint32_t px = (qx << 1) + (i & 1);
                        uint32_t py = (qy << 1) + (i >> 1);
                        if (px >= arg->dst_width || py >= arg->dst_height) continue;
                        *reinterpret_cast<uint32_t*>(
                            (uintptr_t)(arg->cbuf_addr + px * arg->cbuf_stride
                                                       + py * arg->cbuf_pitch)) = out_color;
                    }
                });
        }
    }
}

// RASTER dispatch v2 (push): straight-line fragment shader. The raster engine's
// work distributor launches this kernel once per covered-quad wave, with this
// lane's frag_payload_t already staged in the gfx window; only pos_mask is used
// here (pure coverage write, no shading).
__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {

    if (arg->sw_path) { sw_raster_main(arg); return; }

    uint32_t pos_mask = vx_frag_posmask();
    if (pos_mask == 0) return;

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
