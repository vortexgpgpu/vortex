#pragma once

// Texel copy between boxes of different texel layouts — device kernel entry.
// Device-only (pulls vx_spawn2.h).

#include <vx_spawn2.h>
#include <gfx_frontend_abi.h>   // texel_copy_arg_t

__kernel void texel_copy_k(texel_copy_arg_t* __UNIFORM__ arg) {
  const uint32_t w = arg->w, wh = w * arg->h, n = wh * arg->d;
  const auto* mb = reinterpret_cast<const uint8_t*>(arg->mask);

  uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t gstride = gridDim.x * blockDim.x;
  for (uint32_t i = gid; i < n; i += gstride) {
    uint32_t z = i / wh, r = i % wh, y = r / w, x = r % w;
    const auto* s = reinterpret_cast<const uint8_t*>(static_cast<uintptr_t>(
        arg->src_addr + (uint64_t)z * arg->src_slice + (uint64_t)y * arg->src_row
        + (uint64_t)x * arg->src_texel + arg->src_byte));
    auto* d = reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(
        arg->dst_addr + (uint64_t)z * arg->dst_slice + (uint64_t)y * arg->dst_row
        + (uint64_t)x * arg->dst_texel + arg->dst_byte));
    for (uint32_t k = 0; k < arg->bytes; ++k) {
      d[k] = (uint8_t)((d[k] & ~mb[k]) | (s[k] & mb[k]));
    }
  }
}
