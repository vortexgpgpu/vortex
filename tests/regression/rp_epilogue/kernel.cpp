#include <vx_spawn2.h>
#include "common.h"

// GEMM epilogue on one accumulator row per thread: bias, GELU, per-row absmax
// and int8 quantization. The scale needs the whole row's absmax first, so a
// row held in registers (CHUNK == ROW) takes one pass; otherwise the kernel
// walks the row twice in CHUNK-column groups, reloading and recomputing it.
__kernel void kernel_main(kernel_arg_t* __UNIFORM__ arg) {
  uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= arg->num_rows)
    return;
  auto acc  = reinterpret_cast<const float*>(arg->acc_addr) + row * ROW;
  auto bias = reinterpret_cast<const float*>(arg->bias_addr);
  auto q    = reinterpret_cast<uint32_t*>(arg->q_addr) + row * (ROW / 4);
  float as  = arg->act_scale;

  float amax = 1e-12f;
  float y[CHUNK];
  for (uint32_t c = 0; c < ROW; c += CHUNK) {
    #pragma clang loop unroll(full)
    for (uint32_t i = 0; i < CHUNK; ++i) {
      y[i] = rp_gelu((acc[c + i] + bias[c + i]) * as);
      amax = __builtin_fmaxf(amax, __builtin_fabsf(y[i]));
    }
  }
  float s = 127.0f / amax;
  for (uint32_t c = 0; c < ROW; c += CHUNK) {
    if (CHUNK != ROW) {
      #pragma clang loop unroll(full)
      for (uint32_t i = 0; i < CHUNK; ++i)
        y[i] = rp_gelu((acc[c + i] + bias[c + i]) * as);
    }
    #pragma clang loop unroll(full)
    for (uint32_t i = 0; i < CHUNK; i += 4) {
      uint32_t w = (uint32_t(rp_quant(y[i], s)) & 0xff)
                 | ((uint32_t(rp_quant(y[i + 1], s)) & 0xff) << 8)
                 | ((uint32_t(rp_quant(y[i + 2], s)) & 0xff) << 16)
                 | (uint32_t(rp_quant(y[i + 3], s)) << 24);
      q[(c + i) / 4] = w;
    }
  }
  reinterpret_cast<float*>(arg->scale_addr)[row] = amax / 127.0f;
}
