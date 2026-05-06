#include <vx_spawn2.h>
#include <vx_intrinsics.h>
#include "common.h"

// vortex2 v2 kernel: one thread per output pixel. Maps (gx, gy) to
// (u, v) in fixed-point texture coords and issues a vx_tex sample.

__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
    uint32_t gx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t gy = blockIdx.y * blockDim.y + threadIdx.y;
    if (gx >= arg->dst_width || gy >= arg->dst_height) return;

    // Pixel center in fixed-point u/v space:
    //   fu = deltaX/2 + deltaX * gx
    //   fv = deltaY/2 + deltaY * gy
    uint32_t fu = (arg->deltaX >> 1) + arg->deltaX * gx;
    uint32_t fv = (arg->deltaY >> 1) + arg->deltaY * gy;

    uint32_t color;
    if (arg->use_trilinear) {
        uint32_t lod0 = arg->lod;
        uint32_t lod1 = (lod0 + 1 < (uint32_t)VX_TEX_LOD_MAX) ? (lod0 + 1) : (uint32_t)VX_TEX_LOD_MAX;
        uint32_t t0 = vx_tex(0, fu, fv, lod0);
        uint32_t t1 = vx_tex(0, fu, fv, lod1);
        // Per-channel lerp by frac/256 between two LODs (manual unpack/pack).
        uint32_t frac = arg->frac & 0xff;
        uint32_t inv  = 256 - frac;
        uint32_t r = ((t0 & 0xff) * inv + (t1 & 0xff) * frac) >> 8;
        uint32_t g = (((t0 >> 8) & 0xff) * inv + ((t1 >> 8) & 0xff) * frac) >> 8;
        uint32_t b = (((t0 >> 16) & 0xff) * inv + ((t1 >> 16) & 0xff) * frac) >> 8;
        uint32_t a = (((t0 >> 24) & 0xff) * inv + ((t1 >> 24) & 0xff) * frac) >> 8;
        color = (a << 24) | (b << 16) | (g << 8) | r;
    } else {
        color = vx_tex(0, fu, fv, arg->lod);
    }

    auto dst_row = reinterpret_cast<uint32_t*>(arg->dst_addr + gy * arg->dst_pitch);
    dst_row[gx] = color;
}
