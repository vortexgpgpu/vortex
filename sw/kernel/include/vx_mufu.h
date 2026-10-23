// Copyright © 2019-2026
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// MUFU special-function instructions (custom-2, requires VX_CFG_EXT_MUFU_ENABLE).
// Results are approximate: subnormals
// flush to zero, frm is ignored and fflags are never raised.
//
// Encoding: .insn r CUSTOM2, fmt=funct3, 0=funct7, fd, fs1, x<fn>
// The function select rides the rs2 field, so it is written as a literal xN.

#ifndef __VX_MUFU_H__
#define __VX_MUFU_H__

#include <vx_intrinsics.h>

#ifdef __cplusplus
extern "C" {
#endif

#define __VX_MUFU_S(name, fn)                               \
__attribute__((always_inline))                              \
inline float name(float x) {                                \
  float r;                                                  \
  __asm__ (".insn r %1, 0, 0, %0, %2, x" #fn                \
           : "=f"(r) : "i"(RISCV_CUSTOM2), "f"(x));         \
  return r;                                                 \
}

__VX_MUFU_S(vx_rcpf,   0)  // 1/x
__VX_MUFU_S(vx_rsqrtf, 1)  // 1/sqrt(x)
__VX_MUFU_S(vx_ex2f,   2)  // 2^x
__VX_MUFU_S(vx_lg2f,   3)  // log2(x)
__VX_MUFU_S(vx_sinpif, 4)  // sin(pi*x)
__VX_MUFU_S(vx_cospif, 5)  // cos(pi*x)
__VX_MUFU_S(vx_tanhf,  6)  // tanh(x)

#undef __VX_MUFU_S

// Fast-math compositions, as CUDA's __expf/__logf/__sinf/__cosf build on MUFU.

__attribute__((always_inline))
inline float vx_expf_fast(float x) {
  return vx_ex2f(x * 1.44269504088896341f);
}

__attribute__((always_inline))
inline float vx_logf_fast(float x) {
  return vx_lg2f(x) * 0.693147180559945309f;
}

__attribute__((always_inline))
inline float vx_sinf_fast(float x) {
  return vx_sinpif(x * 0.318309886183790672f);
}

__attribute__((always_inline))
inline float vx_cosf_fast(float x) {
  return vx_cospif(x * 0.318309886183790672f);
}

__attribute__((always_inline))
inline float vx_sqrtf_fast(float x) {
  // x * rsqrt(x) is 0 * inf at both ends, and MUFU flushes subnormal x to zero.
  // The fixups use integer selects: a float select lowers to a branch, which
  // is not divergence-safe.
  float r = x * vx_rsqrtf(x);
  uint32_t xb, rb;
  __builtin_memcpy(&xb, &x, sizeof(xb));
  __builtin_memcpy(&rb, &r, sizeof(rb));
  rb = ((xb & 0x7f800000) == 0) ? (xb & 0x80000000) : rb;
  rb = (xb == 0x7f800000) ? xb : rb;
  __builtin_memcpy(&r, &rb, sizeof(r));
  return r;
}

__attribute__((always_inline))
inline float vx_divf_fast(float a, float b) {
  return a * vx_rcpf(b);
}

// Activations.

__attribute__((always_inline))
inline float vx_sigmoidf_fast(float x) {
  return 0.5f + 0.5f * vx_tanhf(0.5f * x);
}

__attribute__((always_inline))
inline float vx_siluf_fast(float x) {
  return x * vx_sigmoidf_fast(x);
}

// GELU, tanh form: 0.5x(1 + tanh(sqrt(2/pi)(x + 0.044715x^3))).
__attribute__((always_inline))
inline float vx_geluf_fast(float x) {
  float u = 0.797884560802865356f * x * (1.0f + 0.044715f * x * x);
  return 0.5f * x * (1.0f + vx_tanhf(u));
}

#ifdef __cplusplus
}
#endif

#endif // __VX_MUFU_H__
