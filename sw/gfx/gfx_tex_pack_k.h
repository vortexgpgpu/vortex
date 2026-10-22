#pragma once

// Texture pack for the TEX unit — device kernel entry. Device-only (pulls
// vx_spawn2.h).

#include <vx_spawn2.h>
#include <gfx_frontend_abi.h>   // tex_pack_arg_t
#include <gfx_sw.h>             // gfx_sw::{TexState, tex_level_base}, gfx_tex decode

__kernel void tex_pack_k(tex_pack_arg_t* __UNIFORM__ arg) {
  const auto& ts = *reinterpret_cast<const gfx_sw::TexState*>(
      static_cast<uintptr_t>(arg->texstate_addr));
  const bool is_3d = arg->depth != 0u;
  const uint32_t stride = gfx_tex::FormatStride(ts.format);
  const uint32_t out = arg->mode == TEX_PACK_COPY16 ? 2u : 4u;   // bytes a packed texel
  // A power-of-two texture may leave width/height 0 and carry its size in logdim.
  const uint32_t w0 = ts.width  ? ts.width  : (1u << (ts.logdim & 0xffff));
  const uint32_t h0 = ts.height ? ts.height : (1u << (ts.logdim >> 16));
  // One layer's chain, in texels: each level's first texel (first[l]) and the total.
  uint32_t first[VX_TEX_LOD_MAX + 2];
  uint32_t chain = 0;
  for (uint32_t l = 0; l < arg->levels; ++l) {
    const uint32_t w = (uint32_t)gfx_tex::tex_imax((int32_t)(w0 >> l), 1);
    const uint32_t h = (uint32_t)gfx_tex::tex_imax((int32_t)(h0 >> l), 1);
    const uint32_t d = is_3d ? (uint32_t)gfx_tex::tex_imax((int32_t)(arg->depth >> l), 1) : 1u;
    first[l] = chain;
    chain += w * h * d;
  }
  first[arg->levels] = chain;
  const uint32_t n = chain * arg->layers;

  uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t gstride = gridDim.x * blockDim.x;
  for (uint32_t i = gid; i < n; i += gstride) {
    const uint32_t layer = i / chain;
    uint32_t t = i % chain;
    uint32_t l = 0;
    while (l + 1 < arg->levels && t >= first[l + 1]) {
      ++l;
    }
    t -= first[l];
    const uint32_t w = (uint32_t)gfx_tex::tex_imax((int32_t)(w0 >> l), 1);
    const uint32_t h = (uint32_t)gfx_tex::tex_imax((int32_t)(h0 >> l), 1);
    const uint32_t z = t / (w * h), r = t % (w * h), y = r / w, x = r % w;
    uint32_t sub;
    const uint64_t src = gfx_sw::tex_level_base(ts, is_3d ? z : layer, l)
                       + gfx_tex::TexTexelOffset(ts.format, w, x, y, &sub, ts.row_pitch[l]);
    const void* sp = reinterpret_cast<const void*>(static_cast<uintptr_t>(src));
    const uintptr_t dp = static_cast<uintptr_t>(
        arg->dst_addr + (uint64_t)layer * arg->dst_layer_stride + arg->dst_mip_off[l]
        + (uint64_t)t * out);
    if (arg->mode == TEX_PACK_COPY16) {
      *reinterpret_cast<uint16_t*>(dp) = *reinterpret_cast<const uint16_t*>(sp);
    } else if (arg->mode == TEX_PACK_COPY32) {
      *reinterpret_cast<uint32_t*>(dp) = *reinterpret_cast<const uint32_t*>(sp);
    } else if (arg->mode == TEX_PACK_D24_TO_F32) {
      const uint32_t d = *reinterpret_cast<const uint32_t*>(sp) & 0xffffffu;
      *reinterpret_cast<float*>(dp) = (float)d * (1.0f / 16777215.0f);
    } else {
      *reinterpret_cast<uint32_t*>(dp) = gfx_tex::TexDecodeArgb8At(ts.format, sp, sub, stride);
    }
  }
}
