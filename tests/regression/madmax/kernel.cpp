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
    if (arg->enable_dfv_test) {
        csr_write(VX_CSR_DFV_CTRL, 0);
        csr_write(VX_CSR_DFV_RANDOM_SEED, 0xABCDEF00);
        csr_write(VX_CSR_DFV_SET_THRESHOLD, 240);
        csr_write(VX_CSR_DFV_RELEASE_THRESHOLD, 65518);  // release when lfsr2[15:0] >= 65504 (~0.04%)
        csr_write(VX_CSR_DFV_RELEASE_DELAY, 0);
        csr_write(VX_CSR_DFV_RELEASE_FOREVER, 1);
        csr_write(VX_CSR_DFV_THROTTLE_THRESHOLD, 6144);
        csr_write(VX_CSR_DFV_ICACHE_STALL, 0);
        csr_write(VX_CSR_DFV_DCACHE_STALL, 0);
        csr_write(VX_CSR_DFV_WRITEBACK_STALL, 0);
        csr_write(VX_CSR_DFV_FILL_STALL, 1);
    }
  int __ret = vx_spawn_threads(2, arg->grid_dim, nullptr, (vx_kernel_func_cb)kernel_body, arg);
  if (arg->enable_dfv_test) {
      csr_write(VX_CSR_DFV_CTRL, 0);
  }
  return __ret;
}
