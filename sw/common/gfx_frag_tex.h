// Copyright © 2019-2023
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// gfx_v2 texture-sampling math — the single source of truth shared by the host
// fixed-function tex model (sw/common/gfx_ff_model.cpp TextureSampler) and the
// on-device SIMT software fallback (sw/common/gfx_sw.h tex_sample_sw). The math
// here is byte-identical on both sides, so the SW path matches the FF unit (and
// the cocogfx oracle) bit-for-bit because it IS the same code
// (gfx_v2_software_fallback.md §4.2 / §7).
//
// Freestanding: cocogfx is header-only and Lerp8888/Pack8888 live in
// vx_gfx_abi.h, so this compiles for the baremetal device (no <algorithm>/
// <cmath>/<cassert>).

#pragma once

#include <stdint.h>
#include <VX_types.h>
#include <vx_gfx_abi.h>                 // Pack8888 / Lerp8888
#include <cocogfx/include/fixed.hpp>    // cocogfx::TFixed

namespace gfx_tex {

using cocogfx::TFixed;
using vortex::graphics::Lerp8888;   // vx_gfx_abi.h
using vortex::graphics::Pack8888;

// Freestanding max (device baremetal has no <algorithm>); identical to
// std::max<int32_t> for the use here.
static inline __attribute__((always_inline)) int32_t tex_imax(int32_t a, int32_t b) {
  return a > b ? a : b;
}

// Address/filter descriptor for one (u, v, lod) sample: per-tap byte addresses,
// stride, blend fractions, and format/filter selectors.
struct TexelRequest {
  uint64_t addr[4];   // [0] always populated; [1..3] only for BILINEAR
  uint32_t stride;    // bytes per texel (1, 2, or 4)
  uint32_t format;    // VX_TEX_FORMAT_*
  uint32_t filter;    // VX_TEX_FILTER_POINT or _BILINEAR
  uint32_t alpha;     // u-fraction (BILINEAR only)
  uint32_t beta;      // v-fraction (BILINEAR only)
};

template <uint32_t F>
static inline int32_t TextureWrap(TFixed<F> fx, uint32_t wrap) {
  int32_t ret;
  switch (wrap) {
  default:
  case VX_TEX_WRAP_CLAMP:
    ret = fx.data() & -(fx.data() >= 0);
    ret |= ((TFixed<F>::MASK - ret) >> 31);
    break;
  case VX_TEX_WRAP_REPEAT:
    ret = fx.data();
    break;
  case VX_TEX_WRAP_MIRROR:
    ret = fx.data() ^ ((fx.data() << (31-F)) >> 31);
    break;
  }
  return ret & TFixed<F>::MASK;
}

static inline uint32_t FormatStride(uint32_t format) {
  switch (format) {
  default:
  case VX_TEX_FORMAT_A8R8G8B8:
    return 4;
  case VX_TEX_FORMAT_R5G6B5:
  case VX_TEX_FORMAT_A1R5G5B5:
  case VX_TEX_FORMAT_A4R4G4B4:
  case VX_TEX_FORMAT_A8L8:
    return 2;
  case VX_TEX_FORMAT_L8:
  case VX_TEX_FORMAT_A8:
    return 1;
  }
}

static inline void Unpack8888(uint32_t format, uint32_t texel, uint32_t* lo, uint32_t* hi) {
  uint32_t r, g, b, a;
  switch (format) {
  default:
  case VX_TEX_FORMAT_A8R8G8B8:
    r = (texel >> 16) & 0xff;
    g = (texel >> 8) & 0xff;
    b = texel & 0xff;
    a = texel >> 24;
    break;
  case VX_TEX_FORMAT_R5G6B5:
    r = ((texel >> 8) & 0xf8) | ((texel >> 13) & 0x07);
    g = ((texel >> 3) & 0xfc) | ((texel >> 9) & 0x03);
    b = ((texel << 3) & 0xf8) | ((texel >> 2) & 0x07);
    a = 0xff;
    break;
  case VX_TEX_FORMAT_A1R5G5B5:
    r = ((texel >> 7) & 0xf8) | ((texel >> 12) & 0x07);
    g = ((texel >> 2) & 0xf8) | ((texel >> 7)  & 0x07);
    b = ((texel << 3) & 0xf8) | ((texel >> 2)  & 0x07);
    a = (((int32_t)texel << 16) >> 31) & 0xff;
    break;
  case VX_TEX_FORMAT_A4R4G4B4:
    r = ((texel >> 4) & 0xf0) | ((texel >> 8)  & 0x0f);
    g = ((texel >> 0) & 0xf0) | ((texel >> 4)  & 0x0f);
    b = ((texel << 4) & 0xf0) | ((texel >> 0)  & 0x0f);
    a = ((texel >> 8) & 0xf0) | ((texel >> 12) & 0x0f);
    break;
  case VX_TEX_FORMAT_A8L8:
    r = texel & 0xff;
    g = r;
    b = r;
    a = (texel >> 8) & 0xff;
    break;
  case VX_TEX_FORMAT_L8:
    r = texel & 0xff;
    g = r;
    b = r;
    a = 0xff;
    break;
  case VX_TEX_FORMAT_A8:
    r = 0xff;
    g = 0xff;
    b = 0xff;
    a = texel & 0xff;
    break;
  }
  *lo = (r << 16) + b;
  *hi = (a << 16) + g;
}

template <uint32_t F, typename T = int32_t>
static inline void TexAddressLinear(TFixed<F,T> fu,
                                    TFixed<F,T> fv,
                                    uint32_t    log_width,
                                    uint32_t    log_height,
                                    uint32_t    wrapu,
                                    uint32_t    wrapv,
                                    uint32_t*   addr00,
                                    uint32_t*   addr01,
                                    uint32_t*   addr10,
                                    uint32_t*   addr11,
                                    uint32_t*   alpha,
                                    uint32_t*   beta) {
  auto delta_x = TFixed<F,T>::make(TFixed<F,T>::HALF >> log_width);
  auto delta_y = TFixed<F,T>::make(TFixed<F,T>::HALF >> log_height);

  uint32_t u0 = TextureWrap(fu - delta_x, wrapu);
  uint32_t u1 = TextureWrap(fu + delta_x, wrapu);
  uint32_t v0 = TextureWrap(fv - delta_y, wrapv);
  uint32_t v1 = TextureWrap(fv + delta_y, wrapv);

  uint32_t shift_u = (TFixed<F,T>::FRAC - log_width);
  uint32_t shift_v = (TFixed<F,T>::FRAC - log_height);

  uint32_t x0s = (u0 << 8) >> shift_u;
  uint32_t y0s = (v0 << 8) >> shift_v;

  uint32_t x0 = x0s >> 8;
  uint32_t y0 = y0s >> 8;
  uint32_t x1 = u1 >> shift_u;
  uint32_t y1 = v1 >> shift_v;

  *addr00 = x0 + (y0 << log_width);
  *addr01 = x1 + (y0 << log_width);
  *addr10 = x0 + (y1 << log_width);
  *addr11 = x1 + (y1 << log_width);

  *alpha = x0s & 0xff;
  *beta  = y0s & 0xff;
}

template <uint32_t F, typename T = int32_t>
static inline void TexAddressPoint(TFixed<F,T> fu,
                                   TFixed<F,T> fv,
                                   uint32_t    log_width,
                                   uint32_t    log_height,
                                   int         wrapu,
                                   int         wrapv,
                                   uint32_t*   addr) {
  uint32_t u = TextureWrap(fu, wrapu);
  uint32_t v = TextureWrap(fv, wrapv);

  uint32_t x = u >> (TFixed<F,T>::FRAC - log_width);
  uint32_t y = v >> (TFixed<F,T>::FRAC - log_height);

  *addr = x + (y << log_width);
}

static inline uint32_t TexFilterLinear(uint32_t format,
                                       uint32_t texel00,
                                       uint32_t texel01,
                                       uint32_t texel10,
                                       uint32_t texel11,
                                       uint32_t alpha,
                                       uint32_t beta) {
  uint32_t c01l, c01h;
  {
    uint32_t c0l, c0h, c1l, c1h;
    Unpack8888(format, texel00, &c0l, &c0h);
    Unpack8888(format, texel01, &c1l, &c1h);
    c01l = Lerp8888(c0l, c1l, alpha);
    c01h = Lerp8888(c0h, c1h, alpha);
  }

  uint32_t c23l, c23h;
  {
    uint32_t c2l, c2h, c3l, c3h;
    Unpack8888(format, texel10, &c2l, &c2h);
    Unpack8888(format, texel11, &c3l, &c3h);
    c23l = Lerp8888(c2l, c3l, alpha);
    c23h = Lerp8888(c2h, c3h, alpha);
  }

  uint32_t color;
  {
    uint32_t cl = Lerp8888(c01l, c23l, beta);
    uint32_t ch = Lerp8888(c01h, c23h, beta);
    color = Pack8888(cl, ch);
  }

  return color;
}

static inline uint32_t TexFilterPoint(int format, uint32_t texel) {
  uint32_t cl, ch;
  Unpack8888(format, texel, &cl, &ch);
  return Pack8888(cl, ch);
}

// Free per-sample request: produce the TexelRequest from already-resolved tex
// state (no DCR object), so the host TextureSampler and the device SW path share
// it. `base_addr` is the mip's base (mip_base + mip_off for this lod); `logdim`
// is {log_h<<16 | log_w} of mip 0; `u`/`v` are VX_TEX_FXD_FRAC fixed-point.
static inline TexelRequest tex_compute_request(uint64_t base_addr,
                                               uint32_t logdim,
                                               uint32_t format,
                                               uint32_t filter,
                                               uint32_t wrap,
                                               int32_t  u,
                                               int32_t  v,
                                               uint32_t lod) {
  uint32_t log_width  = (uint32_t)tex_imax((int32_t)(logdim & 0xffff) - (int32_t)lod, 0);
  uint32_t log_height = (uint32_t)tex_imax((int32_t)(logdim >> 16) - (int32_t)lod, 0);

  uint32_t wrapu = wrap & 0xffff;
  uint32_t wrapv = wrap >> 16;

  uint32_t stride = FormatStride(format);

  auto xu = TFixed<VX_TEX_FXD_FRAC>::make(u);
  auto xv = TFixed<VX_TEX_FXD_FRAC>::make(v);

  TexelRequest req{};
  req.stride = stride;
  req.format = format;
  req.filter = filter;

  if (filter == VX_TEX_FILTER_BILINEAR) {
    uint32_t offset00, offset01, offset10, offset11;
    uint32_t alpha, beta;
    TexAddressLinear(xu, xv, log_width, log_height, wrapu, wrapv,
      &offset00, &offset01, &offset10, &offset11, &alpha, &beta);
    req.addr[0] = base_addr + offset00 * stride;
    req.addr[1] = base_addr + offset01 * stride;
    req.addr[2] = base_addr + offset10 * stride;
    req.addr[3] = base_addr + offset11 * stride;
    req.alpha   = alpha;
    req.beta    = beta;
  } else { // VX_TEX_FILTER_POINT
    uint32_t offset;
    TexAddressPoint(xu, xv, log_width, log_height, wrapu, wrapv, &offset);
    req.addr[0] = base_addr + offset * stride;
  }
  return req;
}

// Free filter: format-decode + bilinear/point combine of fetched texels.
static inline uint32_t tex_apply_filter(const TexelRequest& req, const uint32_t texels[4]) {
  if (req.filter == VX_TEX_FILTER_BILINEAR)
    return TexFilterLinear(req.format, texels[0], texels[1], texels[2], texels[3],
                           req.alpha, req.beta);
  return TexFilterPoint(req.format, texels[0]);
}

// Trilinear LOD blend: per-channel lerp of two filtered texels by frac/256
// (frac in [0,255], inv = 256 - frac). Bit-identical on the FF model and the
// device SW path, so the trilinear sample matches the gfx_tex oracle.
static inline uint32_t TexLodLerp(uint32_t c0, uint32_t c1, uint32_t frac) {
  frac &= 0xff;
  uint32_t inv = 256 - frac, out = 0;
  for (uint32_t s = 0; s < 32; s += 8) {
    uint32_t a = (c0 >> s) & 0xff, b = (c1 >> s) & 0xff;
    out |= (((a * inv + b * frac) >> 8) & 0xff) << s;
  }
  return out;
}

} // namespace gfx_tex
