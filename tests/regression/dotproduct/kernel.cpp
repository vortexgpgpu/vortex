#include <vx_spawn2.h>
#include "common.h"

extern "C" void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  auto* __restrict src0 = reinterpret_cast<TYPE*>(arg->src0_addr);
  auto* __restrict src1 = reinterpret_cast<TYPE*>(arg->src1_addr);
  auto* __restrict dst  = reinterpret_cast<TYPE*>(arg->dst_addr);
  uint32_t n = arg->num_points;

  auto tbuf = reinterpret_cast<TYPE*>(__local_mem());

  int tid = threadIdx.x;
  int gid = blockIdx.x * blockDim.x + tid;

  TYPE acc(0);
  uint32_t stride = blockDim.x * gridDim.x;
  for (uint32_t i = gid; i < n; i += stride) {
    acc += src0[i] * src1[i];
  }

  tbuf[tid] = acc;

  for (int d = blockDim.x / 2; d > 0; d /= 2) {
    __syncthreads();
    if (tid < d) {
      tbuf[tid] += tbuf[tid + d];
    }
  }
  if (tid == 0) {
    dst[blockIdx.x] = tbuf[0];
  }
}
