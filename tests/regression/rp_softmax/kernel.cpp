#include <vx_spawn2.h>
#include "common.h"

// FlashAttention-style online softmax, as a softmax warp runs it: each thread
// owns one score row and walks it CHUNK columns at a time, keeping the chunk in
// registers. Each chunk updates the running max m and sum l, writes the
// unnormalized probabilities, and rescales the thread's output row (held in
// memory, as tensor memory would hold it) by exp(m_old - m_new). Larger chunks
// mean fewer rescales.
__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= arg->num_rows)
    return;
  uint32_t n = arg->num_cols;
  float scale = arg->scale;
  auto s  = reinterpret_cast<const float*>(arg->s_addr) + row * n;
  auto p  = reinterpret_cast<float*>(arg->p_addr) + row * n;
  auto o  = reinterpret_cast<float*>(arg->o_addr) + row * HEAD;
  auto ml = reinterpret_cast<float*>(arg->ml_addr) + row * 2;

  float m = s[0] * scale;
  float l = 0.0f;
  for (uint32_t c = 0; c < n; c += CHUNK) {
    float v[CHUNK];
    #pragma clang loop unroll(full)
    for (uint32_t i = 0; i < CHUNK; ++i)
      v[i] = s[c + i] * scale;
    float mt = v[0];
    #pragma clang loop unroll(full)
    for (uint32_t i = 1; i < CHUNK; ++i)
      mt = rp_max(mt, v[i]);
    float mn = rp_max(m, mt);
    float corr = rp_exp(m - mn);
    float sum = 0.0f;
    #pragma clang loop unroll(full)
    for (uint32_t i = 0; i < CHUNK; ++i) {
      v[i] = rp_exp(v[i] - mn);
      sum += v[i];
    }
    #pragma clang loop unroll(full)
    for (uint32_t i = 0; i < CHUNK; ++i)
      p[c + i] = v[i];
    l = l * corr + sum;
    // the rescale walks the output row in unrolled column groups, as a
    // correction warp would load them from tensor memory
    for (uint32_t k = 0; k < HEAD; k += OGROUP) {
      #pragma clang loop unroll(full)
      for (uint32_t j = 0; j < OGROUP; ++j)
        o[k + j] *= corr;
    }
    m = mn;
  }
  float r = 1.0f / l;
  for (uint32_t k = 0; k < HEAD; k += OGROUP) {
    #pragma clang loop unroll(full)
    for (uint32_t j = 0; j < OGROUP; ++j)
      o[k + j] *= r;
  }
  ml[0] = m;
  ml[1] = l;
}
