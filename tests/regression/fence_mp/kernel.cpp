#include <vx_spawn2.h>
#include <vx_intrinsics.h>
#include "common.h"

// Message passing across cores through plain memory: the producer (the first
// thread) writes data, fences, then raises a flag; the consumer (the last
// thread, on the last core) reads the data once beforehand so stale copies sit
// in its caches, polls the flag behind a fence, fences again and checks the
// data. A memory fence must make the data visible across every cache the two
// cores do not share.
__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  auto data = reinterpret_cast<volatile uint32_t*>(arg->data_addr);
  auto flag = reinterpret_cast<volatile uint32_t*>(arg->flag_addr);
  auto out  = reinterpret_cast<volatile uint32_t*>(arg->out_addr);
  if (gid == 0) {
    for (uint32_t k = 0; k < arg->count; ++k) {
      data[k] = FENCE_MP_MAGIC + k;
    }
    vx_fence();
    *flag = 1;
    vx_fence();
  } else if (gid == arg->num_tasks - 1) {
    uint32_t sum = 0;
    for (uint32_t k = 0; k < arg->count; ++k) {
      sum += data[k];
    }
    out[2] = sum;
    uint32_t timeout = 1;
    for (uint32_t s = 0; s < arg->spin_limit; ++s) {
      vx_fence();
      if (*flag != 0) {
        timeout = 0;
        break;
      }
    }
    vx_fence();
    uint32_t errors = 0;
    for (uint32_t k = 0; k < arg->count; ++k) {
      errors += (data[k] != FENCE_MP_MAGIC + k);
    }
    out[0] = errors;
    out[1] = timeout;
  }
}
