#pragma once

// Masked texel fill — device kernel entry. Device-only (pulls vx_spawn2.h).

#include <vx_spawn2.h>
#include <gfx_frontend_abi.h>   // clear_arg_t

__kernel void clear_k(clear_arg_t* __UNIFORM__ arg) {
  const uint32_t w = arg->w, wh = w * arg->h, n = wh * arg->d;
  const uint32_t tb = arg->texel_bytes;
  // A texel of whole words starts word-aligned: the span starts at texel 0 and
  // the row and slice pitches are word multiples.
  const bool words = (tb & 3u) == 0;
  const auto* vb = reinterpret_cast<const uint8_t*>(arg->value);
  const auto* mb = reinterpret_cast<const uint8_t*>(arg->mask);

  uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t gstride = gridDim.x * blockDim.x;
  for (uint32_t i = gid; i < n; i += gstride) {
    uint32_t z = i / wh, r = i % wh, y = r / w, x = r % w;
    uintptr_t addr = (uintptr_t)(arg->dst_addr + (uint64_t)z * arg->slice_pitch
                                 + (uint64_t)y * arg->row_pitch + (uint64_t)x * tb);
    if (words) {
      auto* t = reinterpret_cast<uint32_t*>(addr);
      for (uint32_t k = 0; k < tb / 4u; ++k) {
        uint32_t m = arg->mask[k];
        t[k] = (m == 0xffffffffu) ? arg->value[k] : ((t[k] & ~m) | (arg->value[k] & m));
      }
    } else {
      auto* t = reinterpret_cast<uint8_t*>(addr);
      for (uint32_t k = 0; k < tb; ++k) {
        t[k] = (uint8_t)((t[k] & ~mb[k]) | (vb[k] & mb[k]));
      }
    }
  }
}
