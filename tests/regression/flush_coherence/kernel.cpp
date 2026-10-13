#include <vx_spawn2.h>
#include "common.h"

__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= arg->count) {
    return;
  }
  // The rotation moves each element, and so each line, to another thread
  // and core from one launch to the next.
  uint32_t i = tid + arg->rotate;
  if (i >= arg->count) {
    i -= arg->count;
  }
  auto src = reinterpret_cast<const uint32_t*>(arg->src_addr);
  auto dst = reinterpret_cast<uint32_t*>(arg->dst_addr);
  auto acc = reinterpret_cast<uint32_t*>(arg->acc_addr);
  uint32_t v = src[i];
  acc[i] += v ^ arg->round;
  // A store burst saturates the shared levels, so the final stores are
  // still queued in front of them when the flushes start.
  auto tail = reinterpret_cast<uint32_t*>(arg->tail_addr);
  for (uint32_t k = 0; k < arg->burst; ++k) {
    tail[k * arg->count + i] = tail_value(i, k, arg->round);
  }
  // Ending on a store leaves the kernel's last writes in flight at exit.
  dst[i] = v * 3u + arg->round;
}
