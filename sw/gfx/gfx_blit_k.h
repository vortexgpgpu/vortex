#pragma once

// Image blit — device kernel entry. Kept out of gfx_frontend_k.h for the same
// reason as gfx_resolve_k.h: it pulls in gfx_sw.h. Device-only (pulls vx_spawn2.h).

#include <vx_spawn2.h>
#include <gfx_frontend_abi.h>   // blit_arg_t, blit_raw_arg_t
#include <gfx_sw.h>             // gfx_sw::{TexState, tex_sample_f32_layer, om_*}

static inline uint32_t blit_unorm8(float f) {
  return (uint32_t)(gfx_sw::om_clamp01(f) * 255.0f + 0.5f);
}

// The source is sampled in float, so the destination may be any format the
// merger encodes: a wide format takes the four channels as lanes, an 8-bit one
// the packed word the fragment path would hand the merger.
__kernel void blit_k(blit_arg_t* __UNIFORM__ arg) {
  const auto& ts = *reinterpret_cast<const gfx_sw::TexState*>(
      static_cast<uintptr_t>(arg->texstate_addr));
  const uint32_t w = arg->dst_w, h = arg->dst_h, n = w * h;
  const uint32_t fmt = arg->dst_format;
  const uint32_t bpp = gfx_sw::om_color_bpp(fmt);
  const float one = (float)(1 << TEX_FXD_FRAC);

  gfx_sw::om_color_t c{};
  c.color_format = fmt;
  c.cbuf_writemask4 = arg->dst_mask;
  c.color_write = 1;

  // Byte mask of the packed 8-bit word, in the order the word is packed.
  const uint32_t m = arg->dst_mask;
  const uint32_t mr = arg->dst_bgra ? 0x00ff0000u : 0x000000ffu;
  const uint32_t mb = arg->dst_bgra ? 0x000000ffu : 0x00ff0000u;
  const uint32_t bytemask = ((m & 1u) ? mr : 0u) | ((m & 2u) ? 0x0000ff00u : 0u)
                          | ((m & 4u) ? mb : 0u) | ((m & 8u) ? 0xff000000u : 0u);

  uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t gstride = gridDim.x * blockDim.x;
  for (uint32_t i = gid; i < n; i += gstride) {
    uint32_t x = i % w, y = i / w;
    float sx = arg->src_x0 + ((float)x + 0.5f) * arg->scale_x;
    float sy = arg->src_y0 + ((float)y + 0.5f) * arg->scale_y;
    int32_t u = (int32_t)(sx * arg->inv_src_w * one);
    int32_t v = (int32_t)(sy * arg->inv_src_h * one);
    float col[4];
    if (arg->src_3d) {
      gfx_sw::tex_sample_f32_3d(ts, u, v, arg->src_r, 0, arg->filter, col);
    } else {
      gfx_sw::tex_sample_f32_layer(ts, u, v, 0, arg->layer, arg->filter, col);
    }
    uintptr_t addr = (uintptr_t)(arg->dst_addr + (uint64_t)y * arg->dst_pitch + (uint64_t)x * bpp);
    if (gfx_sw::om_is_wide(fmt)) {
      uint32_t lanes[4];
      for (uint32_t k = 0; k < 4; ++k) {
        lanes[k] = gfx_sw::om_f_bits(col[k]);
      }
      gfx_sw::om_wide_color_rmw(c, addr, lanes);
      continue;
    }
    uint32_t r = blit_unorm8(col[0]), g = blit_unorm8(col[1]);
    uint32_t b = blit_unorm8(col[2]), a = blit_unorm8(col[3]);
    uint32_t word = (a << 24) | (g << 8)
                  | (arg->dst_bgra ? ((r << 16) | b) : ((b << 16) | r));
    uint32_t raw = gfx_sw::om_encode_color(fmt, word);
    if (bytemask != 0xffffffffu) {
      // The encoders map 0x00 and 0xff bytes to themselves, so the encoded byte
      // mask selects the stored texel's bytes.
      uint32_t keep = gfx_sw::om_encode_color(fmt, bytemask);
      raw = (gfx_sw::om_load(addr, bpp) & ~keep) | (raw & keep);
    }
    gfx_sw::om_store(addr, bpp, raw);
  }
}

// floor(p), clamped to [0, extent).
static inline uint32_t blit_raw_coord(float p, uint32_t extent) {
  int32_t i = (int32_t)p;
  if ((float)i > p) {
    --i;
  }
  return i < 0 ? 0u : ((uint32_t)i >= extent ? extent - 1u : (uint32_t)i);
}

__kernel void blit_raw_k(blit_raw_arg_t* __UNIFORM__ arg) {
  const uint32_t w = arg->dst_w, n = w * arg->dst_h, tb = arg->texel_bytes;
  const bool words = (tb & 3u) == 0;
  const auto* mb = reinterpret_cast<const uint8_t*>(arg->mask);

  uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t gstride = gridDim.x * blockDim.x;
  for (uint32_t i = gid; i < n; i += gstride) {
    uint32_t x = i % w, y = i / w;
    uint32_t sx = blit_raw_coord(arg->src_x0 + ((float)x + 0.5f) * arg->scale_x, arg->src_w);
    uint32_t sy = blit_raw_coord(arg->src_y0 + ((float)y + 0.5f) * arg->scale_y, arg->src_h);
    uintptr_t s = (uintptr_t)(arg->src_addr + (uint64_t)sy * arg->src_pitch + (uint64_t)sx * tb);
    uintptr_t d = (uintptr_t)(arg->dst_addr + (uint64_t)y * arg->dst_pitch + (uint64_t)x * tb);
    if (words) {
      const auto* sw = reinterpret_cast<const uint32_t*>(s);
      auto* dw = reinterpret_cast<uint32_t*>(d);
      for (uint32_t k = 0; k < tb / 4u; ++k) {
        uint32_t m = arg->mask[k];
        dw[k] = (dw[k] & ~m) | (sw[k] & m);
      }
    } else {
      const auto* sb = reinterpret_cast<const uint8_t*>(s);
      auto* db = reinterpret_cast<uint8_t*>(d);
      for (uint32_t k = 0; k < tb; ++k) {
        db[k] = (uint8_t)((db[k] & ~mb[k]) | (sb[k] & mb[k]));
      }
    }
  }
}
