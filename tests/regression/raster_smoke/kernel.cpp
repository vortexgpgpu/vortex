#include <vx_spawn2.h>
#include <vx_intrinsics.h>
#include "common.h"

// Tiny RASTER smoke: each thread issues one vx_rast() to pop a quad
// descriptor from the cluster-shared raster unit. With raster
// tile_count=0, the unit immediately returns done=1 in bit 0 of the
// result. Thread writes the descriptor word into dst_ptr[tid].

__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= arg->dst_count) return;

    auto dst_ptr = reinterpret_cast<uint32_t*>(arg->dst_addr);
    dst_ptr[tid] = vx_rast();
}
