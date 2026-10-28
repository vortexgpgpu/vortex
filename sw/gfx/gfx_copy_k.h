#pragma once

// Texel copy between boxes of different texel layouts — device kernel entry.
// Device-only (pulls vx_spawn2.h).

#include <vx_spawn2.h>
#include <gfx_frontend_abi.h>   // texel_copy_arg_t

// One texel: `bytes` bytes from `s` to `d` under the mask, the depth of a
// multisample source averaged over its sample planes.
static inline void texel_copy_one(const texel_copy_arg_t* arg, const uint8_t* mb,
                                  uintptr_t sa, uintptr_t da) {
  const auto* s = reinterpret_cast<const uint8_t*>(sa);
  auto* d = reinterpret_cast<uint8_t*>(da);
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

__kernel void texel_copy_k(texel_copy_arg_t* __UNIFORM__ arg) {
  // The fields the loops use, read once: the stores below could alias the
  // argument block, which would otherwise reload them after every store.
  const texel_copy_arg_t& a = *arg;
  const auto* mb = reinterpret_cast<const uint8_t*>(a.mask);

  // Whole texels of whole words, every bit written: a word copy. The pitches,
  // tile offsets and byte offsets are then all word multiples.
  bool words = a.samples <= 1 && (a.bytes & 3u) == 0 && (a.src_byte & 3u) == 0 &&
               (a.dst_byte & 3u) == 0 && ((a.src_addr | a.dst_addr) & 3u) == 0 &&
               ((a.src_texel | a.dst_texel | a.src_row | a.dst_row |
                 a.src_slice | a.dst_slice) & 3u) == 0;
  for (uint32_t k = 0; words && k < a.bytes / 4u; ++k) {
    words = a.mask[k] == 0xffffffffu;
  }
  const uint32_t nwords = a.bytes / 4u;

  // A thread copies a run of TEXEL_COPY_RUN texels along a row, in segments
  // that end at the next tile edge of either tiled side: within a segment both
  // addresses step a texel at a time.
  const uint32_t runs = (a.w + TEXEL_COPY_RUN - 1u) / TEXEL_COPY_RUN;
  const uint32_t rh = runs * a.h, n = rh * a.d;
  const uint32_t src_tw = a.src_tiles.shape ? 1u << GFX_SPARSE_LW(a.src_tiles.shape) : 0u;
  const uint32_t dst_tw = a.dst_tiles.shape ? 1u << GFX_SPARSE_LW(a.dst_tiles.shape) : 0u;
  const uint32_t src_step = a.src_tiles.shape ? GFX_SPARSE_BPP(a.src_tiles.shape) : a.src_texel;
  const uint32_t dst_step = a.dst_tiles.shape ? GFX_SPARSE_BPP(a.dst_tiles.shape) : a.dst_texel;
  const uintptr_t src_base = static_cast<uintptr_t>(a.src_addr) + a.src_byte;
  const uintptr_t dst_base = static_cast<uintptr_t>(a.dst_addr) + a.dst_byte;

  const gfx_sparse_box_t src_tiles = a.src_tiles, dst_tiles = a.dst_tiles;
  const uint32_t src_texel = a.src_texel, src_row = a.src_row, src_slice = a.src_slice;
  const uint32_t dst_texel = a.dst_texel, dst_row = a.dst_row, dst_slice = a.dst_slice;
  const uint32_t w = a.w;

  uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t gstride = gridDim.x * blockDim.x;
  for (uint32_t i = gid; i < n; i += gstride) {
    const uint32_t z = i / rh, r = i % rh, y = r / runs, x0 = (r % runs) * TEXEL_COPY_RUN;
    const uint32_t x1 = (x0 + TEXEL_COPY_RUN < w) ? x0 + TEXEL_COPY_RUN : w;
    for (uint32_t x = x0; x < x1;) {
      uint32_t len = x1 - x;
      if (src_tw) {
        const uint32_t edge = src_tw - ((src_tiles.x + x) & (src_tw - 1u));
        len = edge < len ? edge : len;
      }
      if (dst_tw) {
        const uint32_t edge = dst_tw - ((dst_tiles.x + x) & (dst_tw - 1u));
        len = edge < len ? edge : len;
      }
      uintptr_t sa = src_base + static_cast<uintptr_t>(gfx_sparse_box_off(
          &src_tiles, src_texel, src_row, src_slice, x, y, z));
      uintptr_t da = dst_base + static_cast<uintptr_t>(gfx_sparse_box_off(
          &dst_tiles, dst_texel, dst_row, dst_slice, x, y, z));
      if (words && nwords == 1) {
        for (uint32_t k = 0; k < len; ++k, sa += src_step, da += dst_step) {
          *reinterpret_cast<uint32_t*>(da) = *reinterpret_cast<const uint32_t*>(sa);
        }
      } else if (words) {
        for (uint32_t k = 0; k < len; ++k, sa += src_step, da += dst_step) {
          const auto* sw = reinterpret_cast<const uint32_t*>(sa);
          auto* dw = reinterpret_cast<uint32_t*>(da);
          for (uint32_t j = 0; j < nwords; ++j) {
            dw[j] = sw[j];
          }
        }
      } else {
        for (uint32_t k = 0; k < len; ++k, sa += src_step, da += dst_step) {
          texel_copy_one(arg, mb, sa, da);
        }
      }
      x += len;
    }
  }
}
