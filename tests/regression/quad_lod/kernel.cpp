#include <vx_spawn2.h>
#include <vx_intrinsics.h>
#include <vx_tex_lod.h>
#include "common.h"

// The cross-lane LOD must equal the single-owner LOD, bit for bit.
//
// A quad is four adjacent lanes, so lane l holds the pixel
//   (2*qx + (l&1), 2*qy + ((l>>1)&1))   where qx,qy address the quad.
// vx_tex_auto_lod derives the texture gradients by reading its quad neighbours;
// vx_tex_quad_lod derives them from four coords one caller already holds. Each
// lane computes both and XORs them, so a nonzero output word is a mismatch.
//
// The warp index scales the gradient, sweeping the LOD across its whole range.

__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
    auto dst_ptr = reinterpret_cast<uint32_t*>(arg->dst_addr);

    uint32_t tid = threadIdx.x;  // lane index within warp
    uint32_t wid = blockIdx.x;   // warp index
    uint32_t threads_per_warp = blockDim.x;

    uint32_t quad = tid >> 2;
    uint32_t sub  = tid & 3;

    // A quad's pixel origin, and a per-warp gradient that walks the mip range.
    // The gradient is scaled past the fixed-point bias so the LOD is actually
    // nonzero: a coord step of 1 << (TEX_FXD_FRAC - logw + k) selects mip k.
    uint32_t qx = quad;
    uint32_t qy = wid;
    uint32_t shift = (uint32_t)TEX_FXD_FRAC - arg->logw + (wid & 7);
    int32_t  step = (int32_t)((uint32_t)1 << shift);

    // this lane's own texel coords
    int32_t px = (int32_t)(2 * qx) + (int32_t)(sub & 1);
    int32_t py = (int32_t)(2 * qy) + (int32_t)((sub >> 1) & 1);
    int32_t u = step * px;
    int32_t v = step * py;

    uint32_t lod_lane = vx_tex_auto_lod(u, v, arg->logw, arg->logh);

    // the same four coords, held by this lane alone
    int32_t uu[4], vv[4];
    for (int f = 0; f < 4; ++f) {
        int32_t fx = (int32_t)(2 * qx) + (int32_t)(f & 1);
        int32_t fy = (int32_t)(2 * qy) + (int32_t)((f >> 1) & 1);
        uu[f] = step * fx;
        vv[f] = step * fy;
    }
    uint32_t lod_quad = vx_tex_quad_lod(uu, vv, arg->logw, arg->logh);

    // Low half: the mismatch (must be zero). High half: the LOD this lane saw, so
    // the host can prove the sweep actually reached nonzero mips -- a test whose
    // LOD is always 0 would pass even with the cross-lane read broken.
    dst_ptr[wid * threads_per_warp + tid] = ((lod_lane & 0xffff) << 16)
                                          | ((lod_lane ^ lod_quad) & 0xffff);
}
