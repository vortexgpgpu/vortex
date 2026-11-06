#include <vx_spawn2.h>
#include <vx_mufu.h>
#include "common.h"

template <typename F>
static void apply(const float* src, float* dst, uint32_t n, uint32_t repeat, F f) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  float x = src[i];
  float y = f(x);
  // Extra rounds recompute f(x) to lengthen the run; the empty asm statements
  // hide x and keep each result live so the rounds cannot be folded.
  for (uint32_t r = 1; r < repeat; ++r) {
    __asm__ volatile ("" : "+f"(x));
    y = f(x);
    __asm__ volatile ("" :: "f"(y));
  }
  dst[i] = y;
}

__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  auto src = reinterpret_cast<const float*>(arg->src_addr);
  auto dst = reinterpret_cast<float*>(arg->dst_addr);
  uint32_t n = arg->num_points;
  uint32_t k = arg->repeat;
  switch (arg->op) {
  case OP_RCP:     apply(src, dst, n, k, vx_rcpf); break;
  case OP_RSQRT:   apply(src, dst, n, k, vx_rsqrtf); break;
  case OP_EX2:     apply(src, dst, n, k, vx_ex2f); break;
  case OP_LG2:     apply(src, dst, n, k, vx_lg2f); break;
  case OP_SINPI:   apply(src, dst, n, k, vx_sinpif); break;
  case OP_COSPI:   apply(src, dst, n, k, vx_cospif); break;
  case OP_TANH:    apply(src, dst, n, k, vx_tanhf); break;
  case OP_EXPF:    apply(src, dst, n, k, vx_expf_fast); break;
  case OP_LOGF:    apply(src, dst, n, k, vx_logf_fast); break;
  case OP_SINF:    apply(src, dst, n, k, vx_sinf_fast); break;
  case OP_COSF:    apply(src, dst, n, k, vx_cosf_fast); break;
  case OP_SQRTF:   apply(src, dst, n, k, vx_sqrtf_fast); break;
  case OP_SIGMOID: apply(src, dst, n, k, vx_sigmoidf_fast); break;
  case OP_SILU:    apply(src, dst, n, k, vx_siluf_fast); break;
  case OP_GELU:    apply(src, dst, n, k, vx_geluf_fast); break;
  default: break;
  }
}
