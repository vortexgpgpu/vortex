#include <vx_spawn2.h>
#include "common.h"

// A divergent early `continue` out of a loop body whose other path carries its
// own loop, both feeding one accumulator that is live around the outer loop:
// the reconvergence shape of a ray-tracing raygen shader (a missed ray adds the
// sky and continues, a hit one first attenuates by its shadow rays).
__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  auto t_ptr     = reinterpret_cast<const float*>(arg->t_addr);
  auto color_ptr = reinterpret_cast<const float*>(arg->color_addr);
  auto occl_ptr  = reinterpret_cast<const float*>(arg->occl_addr);
  auto dst_ptr   = reinterpret_cast<float*>(arg->dst_addr);
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= arg->num_points)
    return;

  float acc = 0.0f;
  for (uint32_t s = 0; s < arg->num_samples; ++s) {
    float t = t_ptr[idx * arg->num_samples + s];
    float h = color_ptr[idx];
    if (t < 0.0f) {
      acc += h;
      continue;
    }
    for (uint32_t j = 0; j < arg->num_shadows; ++j) {
      float occluded = 1.0f;
      if (occl_ptr[idx] > 0.5f)
        occluded = t - occl_ptr[idx];
      if (occluded > 0.0f)
        h *= 0.3f;
    }
    acc += h;
  }
  dst_ptr[idx] = acc;
}
