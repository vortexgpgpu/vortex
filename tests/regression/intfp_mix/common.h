#pragma once

#include <stdint.h>

// A mixed integer / FP32 throughput kernel: every iteration issues 16 FP32 FMAs
// and 16 integer ALU operations, each on its own chain, so every instruction of
// an iteration is independent of the others. A core that executes INT and FP
// concurrently retires both streams in parallel; one whose writeback serialises
// them does not.

#define MIX_ITERS 256

typedef struct {
  uint32_t size;
  uint64_t dst_f_addr;
  uint64_t dst_i_addr;
} kernel_arg_t;

#ifdef __riscv
// One fmadd.s, never a libcall (-nostdlib kernels lower fma builtins to fmaf()).
inline float mix_fma(float a, float b, float c) {
  float r;
  __asm__("fmadd.s %0, %1, %2, %3" : "=f"(r) : "f"(a), "f"(b), "f"(c));
  return r;
}
#define MIX_FMA(a, b, c) mix_fma(a, b, c)
#else
#include <cmath>
#define MIX_FMA(a, b, c) std::fmaf(a, b, c)
#endif

// FP32: 16 chains f_k = f_k * m + c_(k mod 8) (m < 1, so values stay bounded); fused on
// both sides, so the result is bit-exact.
// INT: 8 chains x_k = (x_k ^ s_k) + k, two dependent ALU ops each.
inline void intfp_mix_compute(uint32_t row, uint32_t col, uint32_t size, float *out_f, uint32_t *out_i) {
  const uint32_t id = row * size + col;
  const float m = 0.9990234375f;
  float f[16];
  float c[8];
  for (int k = 0; k < 16; ++k)
    f[k] = float(id + k) * 0.25f;
  for (int k = 0; k < 8; ++k)
    c[k] = float(k + 1) * 0.0078125f;
  uint32_t x[8];
  uint32_t s[8];
  for (int k = 0; k < 8; ++k) {
    x[k] = id * 2654435761u + k;
    s[k] = (id << k) | 1u;
  }
  for (int i = 0; i < MIX_ITERS; ++i) {
    f[0]  = MIX_FMA(f[0],  m, c[0]);   x[0] = (x[0] ^ s[0]) + 1u;
    f[1]  = MIX_FMA(f[1],  m, c[1]);   x[1] = (x[1] ^ s[1]) + 2u;
    f[2]  = MIX_FMA(f[2],  m, c[2]);
    f[3]  = MIX_FMA(f[3],  m, c[3]);   x[2] = (x[2] ^ s[2]) + 3u;
    f[4]  = MIX_FMA(f[4],  m, c[4]);   x[3] = (x[3] ^ s[3]) + 4u;
    f[5]  = MIX_FMA(f[5],  m, c[5]);
    f[6]  = MIX_FMA(f[6],  m, c[6]);   x[4] = (x[4] ^ s[4]) + 5u;
    f[7]  = MIX_FMA(f[7],  m, c[7]);   x[5] = (x[5] ^ s[5]) + 6u;
    f[8]  = MIX_FMA(f[8],  m, c[0]);
    f[9]  = MIX_FMA(f[9],  m, c[1]);   x[6] = (x[6] ^ s[6]) + 7u;
    f[10] = MIX_FMA(f[10], m, c[2]);  x[7] = (x[7] ^ s[7]) + 8u;
    f[11] = MIX_FMA(f[11], m, c[3]);
    f[12] = MIX_FMA(f[12], m, c[4]);
    f[13] = MIX_FMA(f[13], m, c[5]);
    f[14] = MIX_FMA(f[14], m, c[6]);
    f[15] = MIX_FMA(f[15], m, c[7]);
  }
  float sf = 0.0f;
  for (int k = 0; k < 16; ++k)
    sf += f[k];
  uint32_t si = 0;
  for (int k = 0; k < 8; ++k)
    si ^= x[k];
  *out_f = sf;
  *out_i = si;
}
