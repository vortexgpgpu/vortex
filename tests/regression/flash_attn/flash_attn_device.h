// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0

// Device side shared by the flash_attn kernels: warp reductions over a row.

#ifndef FLASH_ATTN_DEVICE_H
#define FLASH_ATTN_DEVICE_H

#include <vx_intrinsics.h>

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

// Butterfly reductions across the warp's lanes; every lane gets the result.
inline float warp_reduce_max(float x, uint32_t warp_size) {
  int clamp = warp_size - 1, segmask = ~clamp & 0x3f;
  for (uint32_t off = warp_size >> 1; off > 0; off >>= 1) {
    float y = from_bits(vx_shfl_bfly(bits_from(x), off, clamp, segmask));
    x = (y > x) ? y : x;
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

} // namespace flash_attn

#endif // FLASH_ATTN_DEVICE_H
