#pragma once

// Masked texel fill — device kernel entry. Device-only (pulls vx_spawn2.h).

#include <vx_spawn2.h>
#include <gfx_frontend_abi.h>   // clear_arg_t

__kernel void clear_k(clear_arg_t* __UNIFORM__ arg) {
  // The fields the loops use, read once: the stores below could alias the
  // argument block, which would otherwise reload them after every store.
  const clear_arg_t& a = *arg;
  const uint32_t w = a.w, tb = a.texel_bytes;
  const uint32_t row_pitch = a.row_pitch, slice_pitch = a.slice_pitch;
  const gfx_sparse_box_t tiles = a.dst_tiles;
  const uintptr_t base = static_cast<uintptr_t>(a.dst_addr);
  const auto* vb = reinterpret_cast<const uint8_t*>(a.value);
  const auto* mb = reinterpret_cast<const uint8_t*>(a.mask);

  // A texel of whole words starts word-aligned: the span starts at texel 0 and
  // the row and slice pitches are word multiples.
  const bool words = (tb & 3u) == 0;
  const uint32_t nwords = tb / 4u;
  const bool one_word = words && nwords == 1 && a.mask[0] == 0xffffffffu;
  const uint32_t v0 = a.value[0];

  // A thread fills a run of CLEAR_RUN texels along a row, in segments that end
  // at the next tile edge of a tiled destination: within a segment the address
  // steps a texel at a time.
  const uint32_t runs = (w + CLEAR_RUN - 1u) / CLEAR_RUN;
  const uint32_t rh = runs * a.h, n = rh * a.d;
  const uint32_t tw = tiles.shape ? 1u << GFX_SPARSE_LW(tiles.shape) : 0u;
  const uint32_t step = tiles.shape ? GFX_SPARSE_BPP(tiles.shape) : tb;

  uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t gstride = gridDim.x * blockDim.x;
  for (uint32_t i = gid; i < n; i += gstride) {
    const uint32_t z = i / rh, r = i % rh, y = r / runs, x0 = (r % runs) * CLEAR_RUN;
    const uint32_t x1 = (x0 + CLEAR_RUN < w) ? x0 + CLEAR_RUN : w;
    for (uint32_t x = x0; x < x1;) {
      uint32_t len = x1 - x;
      if (tw) {
        const uint32_t edge = tw - ((tiles.x + x) & (tw - 1u));
        len = edge < len ? edge : len;
      }
      uintptr_t addr = base + static_cast<uintptr_t>(gfx_sparse_box_off(
          &tiles, tb, row_pitch, slice_pitch, x, y, z));
      if (one_word) {
        for (uint32_t k = 0; k < len; ++k, addr += step) {
          *reinterpret_cast<uint32_t*>(addr) = v0;
        }
      } else if (words) {
        for (uint32_t k = 0; k < len; ++k, addr += step) {
          auto* t = reinterpret_cast<uint32_t*>(addr);
          for (uint32_t j = 0; j < nwords; ++j) {
            const uint32_t m = a.mask[j], v = a.value[j];
            t[j] = (m == 0xffffffffu) ? v : ((t[j] & ~m) | (v & m));
          }
        }
      } else {
        for (uint32_t k = 0; k < len; ++k, addr += step) {
          auto* t = reinterpret_cast<uint8_t*>(addr);
          for (uint32_t j = 0; j < tb; ++j) {
            t[j] = (uint8_t)((t[j] & ~mb[j]) | (vb[j] & mb[j]));
          }
        }
      }
      x += len;
    }
  }
}
