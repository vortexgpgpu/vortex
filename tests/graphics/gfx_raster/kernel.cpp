#include <vx_spawn2.h>
#include <vx_graphics.h>
#include <gfx_frag_rast.h>      // gfx_rast::rast_walk_primitive — software raster path
#include "common.h"

using namespace vortex::graphics;

const uint32_t out_color = 0xffffffff;

// Software fine-rasterizer: walks the screen with gfx_rast::rast_walk_primitive
// (the same coverage core the FF model uses) and writes out_color at every covered
// sub-pixel — no FF RASTER. Coverage is binning-independent, so the image matches
// the FF RASTER golden.
//
// A quad's four pixels must land on four adjacent lanes here too, so a group of
// VX_FRAG_QUAD_LANES lanes shares one primitive and walks it in lockstep (same
// edges, same tiles, hence uniform control flow across the group); each lane then
// owns its own sub-pixel of every quad the walk emits.
static void sw_raster_main(kernel_arg_t* arg) {
    // A warp must hold whole quad groups, or a group straddles two warps and the
    // derivative SHFL silently reads outside it.
    static_assert(VX_CFG_NUM_THREADS >= VX_FRAG_QUAD_LANES
               && (VX_CFG_NUM_THREADS % VX_FRAG_QUAD_LANES) == 0,
                  "a pixel quad occupies four adjacent lanes, so a warp must hold whole quads");
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t pid = gid / VX_FRAG_QUAD_LANES;
    uint32_t sub = gid % VX_FRAG_QUAD_LANES;
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
                    uint32_t qx = (pos_mask >> 4) & ((1u << (VX_RASTER_DIM_BITS - 1)) - 1);
                    uint32_t qy =  pos_mask >> (4 + (VX_RASTER_DIM_BITS - 1));
                    uint32_t px = (qx << 1) + (sub & 1);
                    uint32_t py = (qy << 1) + (sub >> 1);
                    if (!((pos_mask >> sub) & 1)) return;
                    if (px >= arg->dst_width || py >= arg->dst_height) return;
                    *reinterpret_cast<uint32_t*>(
                        (uintptr_t)(arg->cbuf_addr + px * arg->cbuf_stride
                                                   + py * arg->cbuf_pitch)) = out_color;
                });
        }
    }
}

// RASTER dispatch (push): straight-line fragment shader. The raster engine's work
// distributor launches this kernel once per packed fragment wave, with this lane's
// pixel already in its launch registers (pure coverage write, no shading).
__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {

    if (arg->sw_path) { sw_raster_main(arg); return; }

    frag_payload_t p;
    vx_frag_load(p);
    // A lane the primitive misses is a helper: it must not branch out of the
    // shader (a covered neighbour may still need to shuffle a value out of it),
    // so coverage gates the export, not the control flow.
    if (vx_frag_covered(p)) {
        auto dst_ptr = reinterpret_cast<uint32_t*>(
            arg->cbuf_addr + vx_frag_x(p) * arg->cbuf_stride
                           + vx_frag_y(p) * arg->cbuf_pitch);
        *dst_ptr = out_color;
    }
}
