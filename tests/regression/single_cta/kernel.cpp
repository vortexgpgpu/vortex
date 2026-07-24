#include "common.h"
#include <vx_spawn2.h>

// One global thread writes one element, stamped with its global index so a
// missing CTA shows up as an unwritten slot rather than a plausible value.
//
// Thread 0 also publishes the CTA dimension CSRs past the end of the data. A
// kernel that sizes its own work from gridDim (as most SPIR-V compute shaders
// do) silently does nothing when they read back wrong, so the launch shape is
// checked as data rather than inferred from the result.
__kernel void kernel_main(kernel_arg_t *__UNIFORM__ arg) {
  auto dst = reinterpret_cast<uint32_t *>(arg->dst_addr);
  uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;

  // Optional thread-divergent branch that no active thread takes. The bound
  // comes from memory so it cannot be folded away, and it varies per thread so
  // the compiler emits a real divergence (split/join) rather than a uniform
  // branch. In a single-thread CTA the taken side has an empty thread mask,
  // which is the degenerate case a reconvergence stack has to represent
  // explicitly.
  if (arg->diverge) {
    if (threadIdx.x > arg->count + 1000) {
      dst[0] = 0xbadbad;
    }
  }

  if (gid < arg->count)
    dst[gid] = gid + 1;

  if (gid == 0) {
    dst[arg->count + 0] = gridDim.x;
    dst[arg->count + 1] = blockDim.x;
    dst[arg->count + 2] = gridDim.y;
    dst[arg->count + 3] = blockDim.y;
    // Trailing markers. Which of these survive tells a warp that stopped
    // executing (a contiguous tail is missing from a fixed point) apart from
    // stores that never reached memory (only the final few are missing).
    for (uint32_t i = 0; i < NUM_MARKERS; ++i)
      dst[arg->count + NUM_DIMS + i] = MARKER_BASE + i;
  }
}
