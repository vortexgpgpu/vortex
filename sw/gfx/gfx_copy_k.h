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
    uint8_t v[16];
    for (uint32_t k = 0; k < arg->bytes; ++k) {
      v[k] = s[k];
    }
    if (arg->samples > 1) {
      // The depth field of the first word, averaged over the sample planes:
      // as floats, or as unorm codes rounded to nearest.
      const uint32_t fmask = (arg->depth_bits >= 32) ? ~0u : ((1u << arg->depth_bits) - 1u);
      float fsum = 0.0f;
      uint64_t usum = 0;
      for (uint32_t k = 0; k < arg->samples; ++k) {
        const uint8_t* p = s + (uint64_t)k * arg->sample_stride;
        uint32_t word = 0;
        for (uint32_t b = 0; b < 4 && b < arg->bytes; ++b) {
          word |= (uint32_t)p[b] << (8 * b);
        }
        const uint32_t field = (word >> arg->depth_shift) & fmask;
        if (arg->depth_float) {
          float f;
          __builtin_memcpy(&f, &field, 4);
          fsum += f;
        } else {
          usum += field;
        }
      }
      uint32_t avg;
      if (arg->depth_float) {
        const float f = fsum / (float)arg->samples;
        __builtin_memcpy(&avg, &f, 4);
      } else {
        avg = (uint32_t)((usum + arg->samples / 2) / arg->samples);
      }
      uint32_t word = 0;
      for (uint32_t b = 0; b < 4 && b < arg->bytes; ++b) {
        word |= (uint32_t)v[b] << (8 * b);
      }
      word = (word & ~(fmask << arg->depth_shift)) | ((avg & fmask) << arg->depth_shift);
      for (uint32_t b = 0; b < 4 && b < arg->bytes; ++b) {
        v[b] = (uint8_t)(word >> (8 * b));
      }
    }
    for (uint32_t k = 0; k < arg->bytes; ++k) {
      d[k] = (uint8_t)((d[k] & ~mb[k]) | (v[k] & mb[k]));
    }
  }
}
