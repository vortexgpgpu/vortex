#include "common.h"
#include <vx_spawn2.h>

__kernel void kernel_main(kernel_arg_t *__UNIFORM__ arg) {
  auto dst_f = reinterpret_cast<float *>(arg->dst_f_addr);
  auto dst_i = reinterpret_cast<uint32_t *>(arg->dst_i_addr);
  int size = arg->size;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y;
  if (col >= size)
    return;
  intfp_mix_compute(row, col, size, &dst_f[row * size + col], &dst_i[row * size + col]);
}
