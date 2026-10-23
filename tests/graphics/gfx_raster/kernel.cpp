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
    if (pid >= arg->num_prims) {
        return;
    }

    const rast_prim_t* prim =
        reinterpret_cast<const rast_prim_t*>(
            (uintptr_t)(arg->prim_addr + (uint64_t)pid * sizeof(rast_prim_t)));

    uint32_t tile = 1u << arg->tile_logsize;
    const gfx_rast::RastConfig cfg = gfx_rast::rast_prim_config(
        { arg->tile_logsize, RAST_SCISSOR_LO(arg->scissor_x), RAST_SCISSOR_LO(arg->scissor_y),
          RAST_SCISSOR_HI(arg->scissor_x), RAST_SCISSOR_HI(arg->scissor_y) },
        prim->scissor_x, prim->scissor_y);

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
                                                   + py * arg->cbuf_pitch)) =
                        arg->pid_color ? prim_id_color(arg->pid_base + pid) : out_color;
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
        *dst_ptr = arg->pid_color ? prim_id_color(arg->pid_base + p.pid) : out_color;
    }
}

// 4x multisampling: the colour buffer is sample-interleaved and every covered
// sample of a covered pixel takes the colour. A separate entry, so the
// single-sample shader's instruction sequence is unchanged.
static void sw_raster_msaa(kernel_arg_t* arg) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t pid = gid / VX_FRAG_QUAD_LANES;
    uint32_t sub = gid % VX_FRAG_QUAD_LANES;
    if (pid >= arg->num_prims) {
        return;
    }

    const rast_prim_t* prim =
        reinterpret_cast<const rast_prim_t*>(
            (uintptr_t)(arg->prim_addr + (uint64_t)pid * sizeof(rast_prim_t)));

    uint32_t tile = 1u << arg->tile_logsize;
    const gfx_rast::RastConfig cfg = gfx_rast::rast_prim_config(
        { arg->tile_logsize, RAST_SCISSOR_LO(arg->scissor_x), RAST_SCISSOR_LO(arg->scissor_y),
          RAST_SCISSOR_HI(arg->scissor_x), RAST_SCISSOR_HI(arg->scissor_y) },
        prim->scissor_x, prim->scissor_y);
    const uint32_t color = arg->pid_color ? prim_id_color(arg->pid_base + pid) : out_color;

    for (uint32_t ty = 0; ty < arg->dst_height; ty += tile) {
        for (uint32_t tx = 0; tx < arg->dst_width; tx += tile) {
            gfx_rast::rast_walk_primitive_msaa(cfg, tx, ty, pid, prim->edges,
                [&](uint32_t pos_mask, const vec3e_t*, const uint32_t* sample_masks, uint32_t) {
                    uint32_t qx = (pos_mask >> 4) & ((1u << (VX_RASTER_DIM_BITS - 1)) - 1);
                    uint32_t qy =  pos_mask >> (4 + (VX_RASTER_DIM_BITS - 1));
                    uint32_t px = (qx << 1) + (sub & 1);
                    uint32_t py = (qy << 1) + (sub >> 1);
                    if (px >= arg->dst_width || py >= arg->dst_height) {
                        return;
                    }
                    for (uint32_t k = 0; k < arg->samples; ++k) {
                        if ((sample_masks[sub] >> k) & 1) {
                            *reinterpret_cast<uint32_t*>(
                                (uintptr_t)(arg->cbuf_addr + px * arg->cbuf_stride + k * 4
                                                           + py * arg->cbuf_pitch)) = color;
                        }
                    }
                });
        }
    }
}

__kernel void kernel_main_msaa(kernel_arg_t* __UNIFORM__ arg) {

    if (arg->sw_path) {
        sw_raster_msaa(arg);
        return;
    }

    frag_payload_t p;
    vx_frag_load(p);
    const uint32_t mask  = vx_frag_mask();
    const uint32_t color = arg->pid_color ? prim_id_color(arg->pid_base + p.pid) : out_color;
    for (uint32_t k = 0; k < arg->samples; ++k) {
        if ((mask >> k) & 1) {
            *reinterpret_cast<uint32_t*>(
                arg->cbuf_addr + vx_frag_x(p) * arg->cbuf_stride + k * 4
                               + vx_frag_y(p) * arg->cbuf_pitch) = color;
        }
    }
}
