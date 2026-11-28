// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0

// Device side shared by the flash_attn kernels: warp reductions over a row,
// and the softmax exponential.

#ifndef FLASH_ATTN_DEVICE_H
#define FLASH_ATTN_DEVICE_H

#include <vx_intrinsics.h>
#ifdef VX_CFG_EXT_TFU_ENABLE
#include <vx_tfu.h>
#endif

namespace flash_attn {

inline size_t bits_from(float x) {
  size_t u = 0;
  __builtin_memcpy(&u, &x, sizeof(x));
  return u;
}

inline float from_bits(size_t u) {
  float x;
  __builtin_memcpy(&x, &u, sizeof(x));
  return x;
}

// Butterfly reductions across groups of warp_size lanes (a power of two, up to
// the warp); every lane of a group gets its group's result.
// The max uses fmax.s: a (y > x) ? y : x select compiles to a per-lane branch,
// so every step would pay a divergent split and join.
inline float warp_reduce_max(float x, uint32_t warp_size) {
  int clamp = warp_size - 1, segmask = ~clamp & 0x3f;
  for (uint32_t off = warp_size >> 1; off > 0; off >>= 1) {
    float y = from_bits(vx_shfl_bfly(bits_from(x), off, clamp, segmask));
    x = __builtin_fmaxf(x, y);
  }
  return x;
}

inline float warp_reduce_sum(float x, uint32_t warp_size) {
  int clamp = warp_size - 1, segmask = ~clamp & 0x3f;
  for (uint32_t off = warp_size >> 1; off > 0; off >>= 1) {
    x += from_bits(vx_shfl_bfly(bits_from(x), off, clamp, segmask));
  }
  return x;
}

// exp(x) for the softmax.
//
// The C library's expf computes in double precision. Vortex cores have no
// double-precision FPU at XLEN 32, so each of those operations is emulated in
// integer code, and expf alone took most of the attention kernels' time. With
// the transcendental unit (TFU) the hardware 2^x is used. Without it, a
// float-only version: x = k*ln2 + r with |r| <= ln2/2, e^r from a degree-6
// polynomial (about 1 ulp), and 2^k placed in the exponent bits. Results below
// FLT_MIN flush to 0, so exp(-inf) is 0.
inline float expf_fast(float x) {
#ifdef VX_CFG_EXT_TFU_ENABLE
  return vx_expf_fast(x);
#else
  const uint32_t keep = (x >= -87.0f);
  x = __builtin_fminf(__builtin_fmaxf(x, -87.0f), 88.0f);
  // k = round(x * log2(e)), by the 1.5 * 2^23 rounding trick.
  const float shifter = 12582912.0f;
  const float t = __builtin_fmaf(x, 1.44269504088896341f, shifter);
  uint32_t tb;
  __builtin_memcpy(&tb, &t, 4);
  const int32_t k = (int32_t)(tb - 0x4B400000u);
  const float kf = t - shifter;
  // r = x - k*ln2, with ln2 split in two (Cody-Waite).
  float r = __builtin_fmaf(kf, -0.693145751953125f, x);
  r = __builtin_fmaf(kf, -1.428606765330187e-06f, r);
  float p = 1.388888888888889e-03f;
  p = __builtin_fmaf(p, r, 8.333333333333333e-03f);
  p = __builtin_fmaf(p, r, 4.166666666666667e-02f);
  p = __builtin_fmaf(p, r, 1.666666666666667e-01f);
  p = __builtin_fmaf(p, r, 0.5f);
  p = __builtin_fmaf(p, r, 1.0f);
  p = __builtin_fmaf(p, r, 1.0f);
  // 2^k for k in [-126, 127], then flush the clamped low end to 0.
  const uint32_t sb = (uint32_t)(k + 127) << 23;
  float s;
  __builtin_memcpy(&s, &sb, 4);
  uint32_t rb;
  const float res = p * s;
  __builtin_memcpy(&rb, &res, 4);
  rb &= 0u - keep;
  float out;
  __builtin_memcpy(&out, &rb, 4);
  return out;
#endif
}

} // namespace flash_attn

#endif // FLASH_ATTN_DEVICE_H
