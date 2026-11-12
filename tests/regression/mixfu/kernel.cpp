#include "common.h"
#include <vx_spawn2.h>

__kernel void kernel_main(kernel_arg_t *__UNIFORM__ arg) {
  auto dst = reinterpret_cast<mixfu_result_t *>(arg->dst_addr);
  int size = arg->size;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y;
  if (col >= size)
    return;
  dst[row * size + col] = mixfu_compute(row, col, size, arg->iters);
}
