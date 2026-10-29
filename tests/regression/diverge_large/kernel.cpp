#include <vx_spawn2.h>
#include "common.h"

__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  auto out = reinterpret_cast<uint32_t*>(arg->out_addr);
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= arg->num_points)
    return;
  out[i] = large_cfg(i);
}
