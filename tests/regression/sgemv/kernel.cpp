#include <vx_spawn.h>
#include "common.h"
#include "float4.h"

void kernel_body(kernel_arg_t *__UNIFORM__ arg) {
  auto A = reinterpret_cast<float *>(arg->A_addr); // Matrix (M x N)
  auto x = reinterpret_cast<float *>(arg->x_addr); // Vector (N x 1)
  auto y = reinterpret_cast<float *>(arg->y_addr); // Output (M x 1)
  uint32_t M = arg->M, N = arg->N;

  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= M)
    return;

  float sum = 0.0f;
  for (int col = 0; col < N; col += 4) {
    float4 a = *(float4 *)(&A[row * N + col]);
    float4 b = *(float4 *)(&x[col]);
    sum += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
  }
  y[row] = sum;
}

int main() {
  kernel_arg_t *arg = (kernel_arg_t *)csr_read(VX_CSR_MSCRATCH);
    if (arg->enable_dfv_test) {
        csr_write(VX_CSR_DFV_CTRL, 1);
        csr_write(VX_CSR_DFV_RANDOM_SEED, 0xABCDEF00);
        csr_write(VX_CSR_DFV_SET_THRESHOLD, 240);
        csr_write(VX_CSR_DFV_RELEASE_THRESHOLD, 65504);  // release when lfsr2[15:0] >= 65504 (~0.04%)
        csr_write(VX_CSR_DFV_RELEASE_DELAY, 4096);
        csr_write(VX_CSR_DFV_RELEASE_FOREVER, 0);
        csr_write(VX_CSR_DFV_THROTTLE_THRESHOLD, 6144);
        csr_write(VX_CSR_DFV_FILL_BANK_MASK, 65535);
        csr_write(VX_CSR_DFV_DCACHE_CORE_REQ_STALL, 0);
        csr_write(VX_CSR_DFV_ICACHE_FILL_REQ_STALL, 0);
        csr_write(VX_CSR_DFV_DCACHE_FILL_REQ_STALL, 0);
        csr_write(VX_CSR_DFV_WRITEBACK_STALL, 0);
        csr_write(VX_CSR_DFV_DCACHE_FILL_RSP_STALL, 1);
    }
  uint32_t blockDim(1); // 1 thread per row (adjust for vector width)
  int __ret = vx_spawn_threads(1, arg->grid_dim, &blockDim, (vx_kernel_func_cb)kernel_body, arg);
  if (arg->enable_dfv_test) {
      csr_write(VX_CSR_DFV_CTRL, 0);
  }
  return __ret;
}