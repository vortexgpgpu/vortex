#pragma once

#include <stdint.h>

typedef struct {
  uint32_t size;
  uint32_t iters;
  uint64_t dst_addr;
} kernel_arg_t;

typedef struct {
  float    f;
  uint32_t i;
} mixfu_result_t;

inline float mixfu_fma(float a, float b, float c) {
#ifdef __riscv
  float r;
  __asm__("fmadd.s %0, %1, %2, %3" : "=f"(r) : "f"(a), "f"(b), "f"(c));
  return r;
#else
  return __builtin_fmaf(a, b, c);
#endif
}

// Eight independent FMA chains interleaved with eight independent integer
// chains, so the FPU and the ALU each receive every other instruction. Two
// multiplies, a minimum and a conversion take their units' other pipelines,
// whose results return between those of the chains.
inline mixfu_result_t mixfu_compute(uint32_t row, uint32_t col, uint32_t size, uint32_t iters) {
  uint32_t x0 = row * size + col;
  uint32_t x1 = col * size + row;
  uint32_t x2 = x0 ^ 0x9e3779b9u;
  uint32_t x3 = x1 + 0x7f4a7c15u;
  uint32_t x4 = x2 ^ x3;
  uint32_t x5 = x2 + x3;
  uint32_t x6 = x4 ^ 0x85ebca6bu;
  uint32_t x7 = x5 + 0xc2b2ae35u;
  float a0 = (float)(x0 & 0xff) * 0.0078125f;
  float a1 = (float)(x1 & 0xff) * 0.0078125f;
  float a2 = a0 + 0.25f;
  float a3 = a1 + 0.25f;
  float a4 = a0 - 0.5f;
  float a5 = a1 - 0.5f;
  float a6 = a2 * 0.5f;
  float a7 = a3 * 0.5f;
  for (uint32_t i = 0; i < iters; ++i) {
    a0 = mixfu_fma(a0, 0.25f, a1);
    x0 += x1;
    a1 = mixfu_fma(a1, 0.25f, a2);
    x1 ^= x2;
    a2 = mixfu_fma(a2, 0.25f, a3);
    x2 += x3;
    a3 = mixfu_fma(a3, 0.25f, a4);
    x3 ^= x4;
    a4 = mixfu_fma(a4, 0.25f, a5);
    x4 += x5;
    a5 = mixfu_fma(a5, 0.25f, a6);
    x5 ^= x6;
    a6 = mixfu_fma(a6, 0.25f, a7);
    x6 += x7;
    a7 = mixfu_fma(a7, 0.25f, a0);
    x7 ^= x0;
    x3 += x4 * x5;
    x6 ^= (uint32_t)(((uint64_t)x7 * x1) >> 32);
    a3 = mixfu_fma(a3, 0.5f, __builtin_fminf(a5, 1.0f));
    a6 = mixfu_fma((float)(x2 & 0xf), 0.0625f, a6);
  }
  mixfu_result_t r;
  r.f = ((a0 + a1) + (a2 + a3)) + ((a4 + a5) + (a6 + a7));
  r.i = ((x0 ^ x1) + (x2 ^ x3)) ^ ((x4 + x5) ^ (x6 + x7));
  return r;
}
