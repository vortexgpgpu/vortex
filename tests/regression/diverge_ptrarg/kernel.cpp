#include <vx_spawn2.h>
#include "common.h"

struct record_t {
  float v[24];
};

static __attribute__((noinline)) void store(int32_t* out, int32_t value) {
  *out = value;
}

// Branches on per-thread data read through a pointer into the caller's stack:
// the pointer is the same in every thread, the data behind it is not. Enough
// fields are read that the argument stays a pointer rather than being promoted
// to values, and only added, since a contracted multiply-add is a call that
// the analysis takes as divergent anyway.
static __attribute__((noinline)) void classify(const record_t& r, int32_t* out) {
  if (r.v[0] + r.v[5] + r.v[9] + r.v[13] - r.v[17] - r.v[2] > 0.0f) {
    store(out, 1);
  } else {
    *out = 0;
  }
}

__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  auto src_ptr = reinterpret_cast<const float*>(arg->src_addr);
  auto dst_ptr = reinterpret_cast<int32_t*>(arg->dst_addr);
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= arg->num_points)
    return;
  record_t r;
  for (int i = 0; i < 24; ++i) {
    r.v[i] = (i == 0) ? src_ptr[idx] : 0.0f;
  }
  // Two calls keep classify out of line.
  classify(r, dst_ptr + idx);
  classify(r, dst_ptr + idx);
}
