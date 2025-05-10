#include "common.h"
#include <vx_spawn.h>

void kernel_body(kernel_arg_t *__UNIFORM__ arg) {
  auto dst = reinterpret_cast<float *>(arg->dst_addr);
  int size = arg->size;
  int col = blockIdx.x;
  int row = blockIdx.y;
  dst[row * size + col] = madmax_compute(row, col, size);
}

int main() {
  kernel_arg_t *arg = (kernel_arg_t *)csr_read(VX_CSR_MSCRATCH);
  return vx_spawn_threads(2, arg->grid_dim, nullptr, (vx_kernel_func_cb)kernel_body, arg);
}