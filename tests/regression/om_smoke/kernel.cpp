#include <vx_spawn2.h>
#include <vx_intrinsics.h>
#include "common.h"

// Tiny OM smoke: each thread issues one vx_om() call (fire-and-forget).
// The OM unit writes the color to a host-configured cbuf at (x, y).
// Host then reads cbuf back and verifies the writes landed.

__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= arg->dst_count) return;

    // Each thread writes a unique color to (x=tid, y=0, face=0).
    uint32_t color = 0x10000000u | tid;
    uint32_t depth = 0;
    vx_om(tid, 0, 0, color, depth);
}
