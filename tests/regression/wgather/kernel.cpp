#include <vx_spawn2.h>
#include <vx_intrinsics.h>
#include "common.h"

// wgather operates in groups of 4 lanes. Within each group, src_offset (funct2)
// selects the source lane; that lane's rs1/rs2/rs3 are broadcast to the other
// three lanes of the same group. The source lane keeps its own rd unchanged.
//
// With src_offset=0 and groups of 4:
//   group g (base = g*4):
//     lane base+0 (src):  rd unchanged  → self_val = base+0
//     lane base+1:        rd ← rs1[base+0]  = (base+0)*10 + 1
//     lane base+2:        rd ← rs2[base+0]  = (base+0)*10 + 2
//     lane base+3:        rd ← rs3[base+0]  = (base+0)*10 + 3

__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
    auto dst_ptr = reinterpret_cast<uint32_t*>(arg->dst_addr);
    auto tp_ptr  = reinterpret_cast<uint32_t*>(arg->tp_addr);

    uint32_t tid = threadIdx.x; // lane index within warp
    uint32_t wid = blockIdx.x;  // warp index
    uint32_t threads_per_warp = blockDim.x;

    // Compute which group of 4 this thread belongs to, and its position within it.
    uint32_t group      = tid >> 2;        // tid / 4
    uint32_t group_base = group << 2;      // tid rounded down to multiple of 4

    // ---- Basic wgather test (src_offset=0) ----
    // Each thread carries independent rs1/rs2/rs3 values.
    // The source lane (offset=0 within each group) provides:
    //   rs1 = group_base*10 + 1
    //   rs2 = group_base*10 + 2
    //   rs3 = group_base*10 + 3
    size_t self_val = (size_t)wid * 1000 + (size_t)group_base;
    size_t v1       = (size_t)group_base * 10 + 1;
    size_t v2       = (size_t)group_base * 10 + 2;
    size_t v3       = (size_t)group_base * 10 + 3;

    size_t result = vx_wgather(self_val, v1, v2, v3);

    dst_ptr[wid * threads_per_warp + tid] = (uint32_t)result;

    // ---- Transpose test ----
    // Each lane holds one row of a 4x4 matrix.
    // Row values for lane_in_group i (= tid - group_base):
    //   M[i][j] = group_base_val + j*4 + i + 1,  j = 0..3
    // where group_base_val = wid*100 + group*16
    //
    // After vx_transpose4, lane i holds column i of the original:
    //   T[i][j] = M[j][i] = group_base_val + i*4 + j + 1
    size_t group_base_val = (size_t)wid * 100 + (size_t)group * 16;
    uint32_t i            = tid - group_base; // lane within group (0-3)

    size_t a0 = group_base_val + 0 * 4 + i + 1; // row 0, col i
    size_t a1 = group_base_val + 1 * 4 + i + 1; // row 1, col i
    size_t a2 = group_base_val + 2 * 4 + i + 1; // row 2, col i
    size_t a3 = group_base_val + 3 * 4 + i + 1; // row 3, col i

    size_t t0, t1, t2, t3;
    vx_transpose4(a0, a1, a2, a3, t0, t1, t2, t3);

    // tp_ptr layout: 4 values per lane, ordered by warp then lane then column j=0..3
    uint32_t* lane_out = tp_ptr + (wid * threads_per_warp + tid) * 4;
    lane_out[0] = (uint32_t)t0;
    lane_out[1] = (uint32_t)t1;
    lane_out[2] = (uint32_t)t2;
    lane_out[3] = (uint32_t)t3;
}
