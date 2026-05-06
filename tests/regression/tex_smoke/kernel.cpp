#include <vx_spawn2.h>
#include <vx_intrinsics.h>
#include "common.h"

// Tiny TEX smoke: each thread reads one texel via vx_tex with a fixed
// (u,v,lod=0) and stores it into dst_ptr[tid]. Host configures the
// texture unit to point at a small texture before launch.

__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= arg->dst_count) return;

    auto dst_ptr = reinterpret_cast<uint32_t*>(arg->dst_addr);

    // Sample at (u=0, v=0, lod=0) from texture stage 0.
    // Returns the single texel of the 1x1 texture.
    uint32_t color = vx_tex(0, 0, 0, 0);
    dst_ptr[tid] = color;
}
