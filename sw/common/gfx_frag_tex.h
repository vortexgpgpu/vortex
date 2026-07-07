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
// fixed-function tex model (sw/common/gfx_ff_model.cpp) and the
// on-device SIMT software fallback (sw/common/gfx_sw.h tex_sample_sw). The math
// here is byte-identical on both sides, so the SW path matches the FF unit (and
// the cocogfx oracle) bit-for-bit because it IS the same code.
//
// Freestanding: cocogfx is header-only and Lerp8888/Pack8888 live in
// vx_gfx_abi.h, so this compiles for the baremetal device (no <algorithm>/
// <cmath>/<cassert>).

#pragma once

#include <stdint.h>
#include <VX_types.h>
#include "gfx_tex_const.h"
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
  case VX_TEX_FORMAT_SRGB8:      // R8G8B8(X8) sRGB, 4 bytes
  case VX_TEX_FORMAT_SRGB8A8:    // R8G8B8A8 sRGB
  case VX_TEX_FORMAT_R32F:       // single float
  case VX_TEX_FORMAT_D32F:       // float depth
    return 4;
  case VX_TEX_FORMAT_R5G6B5:
  case VX_TEX_FORMAT_A1R5G5B5:
  case VX_TEX_FORMAT_A4R4G4B4:
  case VX_TEX_FORMAT_A8L8:
  case VX_TEX_FORMAT_RG8:        // two 8-bit channels
  case VX_TEX_FORMAT_R16F:       // single half
  case VX_TEX_FORMAT_D16:        // 16-bit depth
    return 2;
  case VX_TEX_FORMAT_L8:
  case VX_TEX_FORMAT_A8:
  case VX_TEX_FORMAT_R8:         // single 8-bit channel
    return 1;
  case VX_TEX_FORMAT_RG16F:      // 2x half
  case VX_TEX_FORMAT_RGBA16F:    // 4x half (8 bytes)
  case VX_TEX_FORMAT_RG32F:      // 2x float (8 bytes)
    return (format == VX_TEX_FORMAT_RG16F) ? 4 : 8;
  case VX_TEX_FORMAT_RGBA32F:    // 4x float (16 bytes)
    return 16;
  }
}

// FF vx_tex4 handles only formats 0..VX_TEX_FORMAT_FF_MAX with POT dims and no
// border wrap; everything else is the on-device software sampler.
static inline bool TexIsFFFormat(uint32_t format) {
  return format <= (uint32_t)VX_TEX_FORMAT_FF_MAX;
}

// ── Extended-format decode → 8-bit ARGB working colour ───────────────────────
// The software sampler keeps its filter/blend in the existing 8-bit ARGB space,
// so every extended format is decoded to ARGB8888 here (sRGB linearised, float
// clamped to [0,1], depth read as luminance). SW-path only — never reached by
// the FF model, so there is no FF golden to match.

// sRGB EOTF, 8-bit in → 8-bit linear out (IEC 61966-2-1, quantised to 8-bit to
// stay in the sampler's working space). Table keeps the header freestanding — no
// powf/<cmath> dependency on the baremetal device.
static inline uint32_t SrgbToLinear8(uint32_t c) {
  static const uint8_t kSrgbLut[256] = {
      0,  0,  0,  0,  0,  0,  0,  1,  1,  1,  1,  1,  1,  1,  1,  1,
      1,  1,  2,  2,  2,  2,  2,  2,  2,  2,  3,  3,  3,  3,  3,  3,
      4,  4,  4,  4,  4,  5,  5,  5,  5,  6,  6,  6,  6,  7,  7,  7,
      8,  8,  8,  8,  9,  9,  9, 10, 10, 10, 11, 11, 12, 12, 12, 13,
     13, 13, 14, 14, 15, 15, 16, 16, 17, 17, 17, 18, 18, 19, 19, 20,
     20, 21, 22, 22, 23, 23, 24, 24, 25, 25, 26, 27, 27, 28, 29, 29,
     30, 30, 31, 32, 32, 33, 34, 35, 35, 36, 37, 37, 38, 39, 40, 41,
     41, 42, 43, 44, 45, 45, 46, 47, 48, 49, 50, 51, 51, 52, 53, 54,
     55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70,
     71, 72, 73, 74, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88,
     90, 91, 92, 93, 95, 96, 97, 99,100,101,103,104,105,107,108,109,
    111,112,114,115,116,118,119,121,122,124,125,127,128,130,131,133,
    134,136,138,139,141,142,144,146,147,149,151,152,154,156,157,159,
    161,163,164,166,168,170,171,173,175,177,179,181,183,184,186,188,
    190,192,194,196,198,200,202,204,206,208,210,212,214,216,218,220,
    222,224,226,229,231,233,235,237,239,242,244,246,248,250,253,255};
  return kSrgbLut[c & 0xff];
}

// IEEE half → float (freestanding bit manipulation; handles subnormals/inf/nan).
static inline float HalfToFloat(uint16_t h) {
  uint32_t sign = (uint32_t)(h & 0x8000) << 16;
  uint32_t exp  = (h >> 10) & 0x1f;
  uint32_t man  = h & 0x3ff;
  uint32_t bits;
  if (exp == 0) {
    if (man == 0) {
      bits = sign;                              // +/- zero
    } else {
      // subnormal: normalise
      exp = 127 - 15 + 1;
      while (!(man & 0x400)) { man <<= 1; --exp; }
      man &= 0x3ff;
      bits = sign | (exp << 23) | (man << 13);
    }
  } else if (exp == 0x1f) {
    bits = sign | 0x7f800000u | (man << 13);    // inf / nan
  } else {
    bits = sign | ((exp - 15 + 127) << 23) | (man << 13);
  }
  float f;
  __builtin_memcpy(&f, &bits, 4);
  return f;
}

// float in [0,1] → 8-bit unorm (clamped, round-to-nearest).
static inline uint32_t FloatToUnorm8(float f) {
  if (f <= 0.0f) return 0;
  if (f >= 1.0f) return 0xff;
  return (uint32_t)(f * 255.0f + 0.5f);
}

// Decode one extended-format texel at `p` to ARGB8888. `p` points at the raw
// texel bytes in resident memory.
static inline uint32_t TexDecodeExtended(uint32_t format, const void* p) {
  const uint8_t* b = (const uint8_t*)p;
  uint32_t r = 0, g = 0, bch = 0, a = 0xff;
  switch (format) {
  case VX_TEX_FORMAT_SRGB8:
  case VX_TEX_FORMAT_SRGB8A8: {
    // stored R,G,B,A little-endian (matches cocogfx ARGB packing R@16,G@8,B@0)
    uint32_t t; __builtin_memcpy(&t, b, 4);
    r   = SrgbToLinear8((t >> 16) & 0xff);
    g   = SrgbToLinear8((t >> 8) & 0xff);
    bch = SrgbToLinear8(t & 0xff);
    a   = (format == VX_TEX_FORMAT_SRGB8A8) ? (t >> 24) : 0xff;   // alpha is linear
    break;
  }
  case VX_TEX_FORMAT_R8:  r = b[0]; g = 0; bch = 0; a = 0xff; break;
  case VX_TEX_FORMAT_RG8: r = b[0]; g = b[1]; bch = 0; a = 0xff; break;
  case VX_TEX_FORMAT_R16F: {
    uint16_t h; __builtin_memcpy(&h, b, 2);
    r = FloatToUnorm8(HalfToFloat(h)); g = 0; bch = 0; a = 0xff; break;
  }
  case VX_TEX_FORMAT_RG16F: {
    uint16_t h0, h1; __builtin_memcpy(&h0, b, 2); __builtin_memcpy(&h1, b + 2, 2);
    r = FloatToUnorm8(HalfToFloat(h0)); g = FloatToUnorm8(HalfToFloat(h1)); bch = 0; a = 0xff; break;
  }
  case VX_TEX_FORMAT_RGBA16F: {
    uint16_t h[4]; __builtin_memcpy(h, b, 8);
    r   = FloatToUnorm8(HalfToFloat(h[0])); g = FloatToUnorm8(HalfToFloat(h[1]));
    bch = FloatToUnorm8(HalfToFloat(h[2])); a = FloatToUnorm8(HalfToFloat(h[3])); break;
  }
  case VX_TEX_FORMAT_R32F: {
    float f; __builtin_memcpy(&f, b, 4);
    r = FloatToUnorm8(f); g = 0; bch = 0; a = 0xff; break;
  }
  case VX_TEX_FORMAT_RG32F: {
    float f0, f1; __builtin_memcpy(&f0, b, 4); __builtin_memcpy(&f1, b + 4, 4);
    r = FloatToUnorm8(f0); g = FloatToUnorm8(f1); bch = 0; a = 0xff; break;
  }
  case VX_TEX_FORMAT_RGBA32F: {
    float f[4]; __builtin_memcpy(f, b, 16);
    r = FloatToUnorm8(f[0]); g = FloatToUnorm8(f[1]); bch = FloatToUnorm8(f[2]); a = FloatToUnorm8(f[3]); break;
  }
  case VX_TEX_FORMAT_D16: {
    uint16_t d; __builtin_memcpy(&d, b, 2);
    uint32_t l = d >> 8;                         // 16-bit unorm depth → luminance
    r = g = bch = l; a = 0xff; break;
  }
  case VX_TEX_FORMAT_D32F: {
    float f; __builtin_memcpy(&f, b, 4);
    uint32_t l = FloatToUnorm8(f);
    r = g = bch = l; a = 0xff; break;
  }
  default:
    break;
  }
  return (a << 24) | (r << 16) | (g << 8) | bch;
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

// Decode one texel of ANY format at `p` (stride bytes) to ARGB8888. FF formats
// go through Unpack8888/Pack8888 (bit-identical to the FF path); extended formats
// through TexDecodeExtended. Used by the border-aware extended sampler so an
// FF-format texture with a CLAMP_TO_BORDER wrap still decodes its real taps.
static inline uint32_t TexDecodeArgb8(uint32_t format, const void* p, uint32_t stride) {
  if (TexIsFFFormat(format)) {
    const uint8_t* b = (const uint8_t*)p;
    uint32_t texel = 0;
    for (uint32_t i = 0; i < stride; ++i) texel |= (uint32_t)b[i] << (i * 8);
    uint32_t lo, hi;
    Unpack8888(format, texel, &lo, &hi);
    return Pack8888(lo, hi);
  }
  return TexDecodeExtended(format, p);
}

// NPOT flag: true when the (per-LOD) integer dims are not exact powers of two
// matching {log_width, log_height}. POT keeps the bit-exact shift/mask addressing
// (matches the FF TEX unit); NPOT falls back to the multiply-addressing path
// (x = floor(u_norm * width), row stride = width) which the FF unit cannot
// represent (FF vx_tex4 handles POT dims only; NPOT is SW-only).
static inline bool TexIsNPOT(uint32_t width, uint32_t height,
                             uint32_t log_width, uint32_t log_height) {
  return (width != (1u << log_width)) || (height != (1u << log_height));
}

template <uint32_t F, typename T = int32_t>
static inline void TexAddressLinear(TFixed<F,T> fu,
                                    TFixed<F,T> fv,
                                    uint32_t    width,
                                    uint32_t    height,
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
  if (TexIsNPOT(width, height, log_width, log_height)) {
    // Multiply-addressing path (NPOT): half-texel = 0.5/width in normalized
    // fixed-point; texel index = floor(u_norm * width) with an 8-bit sub-texel
    // fraction; row stride = width.
    auto delta_x = TFixed<F,T>::make((int32_t)(TFixed<F,T>::HALF / (int32_t)width));
    auto delta_y = TFixed<F,T>::make((int32_t)(TFixed<F,T>::HALF / (int32_t)height));

    uint32_t u0 = TextureWrap(fu - delta_x, wrapu);
    uint32_t u1 = TextureWrap(fu + delta_x, wrapu);
    uint32_t v0 = TextureWrap(fv - delta_y, wrapv);
    uint32_t v1 = TextureWrap(fv + delta_y, wrapv);

    uint32_t x0s = (uint32_t)(((uint64_t)u0 * width * 256) >> TFixed<F,T>::FRAC);
    uint32_t y0s = (uint32_t)(((uint64_t)v0 * height * 256) >> TFixed<F,T>::FRAC);

    uint32_t x0 = x0s >> 8;
    uint32_t y0 = y0s >> 8;
    uint32_t x1 = (uint32_t)(((uint64_t)u1 * width) >> TFixed<F,T>::FRAC);
    uint32_t y1 = (uint32_t)(((uint64_t)v1 * height) >> TFixed<F,T>::FRAC);
    // clamp to the last texel (the wrap already keeps coords in [0,1), but the
    // multiply can land on `width` at the CLAMP sentinel).
    if (x0 >= width)  x0 = width - 1;
    if (x1 >= width)  x1 = width - 1;
    if (y0 >= height) y0 = height - 1;
    if (y1 >= height) y1 = height - 1;

    *addr00 = x0 + y0 * width;
    *addr01 = x1 + y0 * width;
    *addr10 = x0 + y1 * width;
    *addr11 = x1 + y1 * width;

    *alpha = x0s & 0xff;
    *beta  = y0s & 0xff;
    return;
  }

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
                                   uint32_t    width,
                                   uint32_t    height,
                                   uint32_t    log_width,
                                   uint32_t    log_height,
                                   int         wrapu,
                                   int         wrapv,
                                   uint32_t*   addr) {
  uint32_t u = TextureWrap(fu, wrapu);
  uint32_t v = TextureWrap(fv, wrapv);

  if (TexIsNPOT(width, height, log_width, log_height)) {
    uint32_t x = (uint32_t)(((uint64_t)u * width)  >> TFixed<F,T>::FRAC);
    uint32_t y = (uint32_t)(((uint64_t)v * height) >> TFixed<F,T>::FRAC);
    if (x >= width)  x = width - 1;
    if (y >= height) y = height - 1;
    *addr = x + y * width;
    return;
  }

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
// state (no DCR object), so the host FF sampler and the device SW path share
// it. `base_addr` is the mip's base (mip_base + mip_off for this lod); `logdim`
// is {log_h<<16 | log_w} of mip 0; `u`/`v` are TEX_FXD_FRAC fixed-point.
// `width0`/`height0` carry the mip-0 integer dims for the NPOT multiply path;
// pass 0 to derive POT dims from `logdim` (bit-exact FF shift addressing).
static inline TexelRequest tex_compute_request(uint64_t base_addr,
                                               uint32_t logdim,
                                               uint32_t format,
                                               uint32_t filter,
                                               uint32_t wrap,
                                               int32_t  u,
                                               int32_t  v,
                                               uint32_t lod,
                                               uint32_t width0 = 0,
                                               uint32_t height0 = 0) {
  uint32_t log_width  = (uint32_t)tex_imax((int32_t)(logdim & 0xffff) - (int32_t)lod, 0);
  uint32_t log_height = (uint32_t)tex_imax((int32_t)(logdim >> 16) - (int32_t)lod, 0);

  // Per-LOD integer dims. When width0/height0 are given (NPOT-capable path) the
  // LOD dims are max(dim >> lod, 1); otherwise the POT dims from logdim, which
  // keeps TexIsNPOT() false and the addressing bit-exact with the FF unit.
  uint32_t width  = width0  ? (uint32_t)tex_imax((int32_t)(width0  >> lod), 1) : (1u << log_width);
  uint32_t height = height0 ? (uint32_t)tex_imax((int32_t)(height0 >> lod), 1) : (1u << log_height);

  uint32_t wrapu = wrap & 0xffff;
  uint32_t wrapv = wrap >> 16;

  uint32_t stride = FormatStride(format);

  auto xu = TFixed<TEX_FXD_FRAC>::make(u);
  auto xv = TFixed<TEX_FXD_FRAC>::make(v);

  TexelRequest req{};
  req.stride = stride;
  req.format = format;
  req.filter = filter;

  if (filter == VX_TEX_FILTER_BILINEAR) {
    uint32_t offset00, offset01, offset10, offset11;
    uint32_t alpha, beta;
    TexAddressLinear(xu, xv, width, height, log_width, log_height, wrapu, wrapv,
      &offset00, &offset01, &offset10, &offset11, &alpha, &beta);
    req.addr[0] = base_addr + offset00 * stride;
    req.addr[1] = base_addr + offset01 * stride;
    req.addr[2] = base_addr + offset10 * stride;
    req.addr[3] = base_addr + offset11 * stride;
    req.alpha   = alpha;
    req.beta    = beta;
  } else { // VX_TEX_FILTER_POINT
    uint32_t offset;
    TexAddressPoint(xu, xv, width, height, log_width, log_height, wrapu, wrapv, &offset);
    req.addr[0] = base_addr + offset * stride;
  }
  return req;
}

// CLAMP_TO_BORDER: compute the per-tap border mask for the extended SW
// sampler. A tap is the border colour when its pre-wrap coordinate leaves [0,1)
// on an axis whose wrap is VX_TEX_WRAP_BORDER. Tap order matches TexAddressLinear
// (00,01,10,11 = (u-,v-),(u+,v-),(u-,v+),(u+,v+)); POINT uses tap 0 only. The
// half-texel deltas match tex_compute_request's addressing so the border edge
// lands on the same taps the real fetch would.
static inline uint32_t TexBorderMask(int32_t u, int32_t v, uint32_t width, uint32_t height,
                                     uint32_t log_width, uint32_t log_height,
                                     uint32_t wrapu, uint32_t wrapv, uint32_t filter) {
  const int32_t ONE = 1 << TEX_FXD_FRAC;
  bool bu = (wrapu == VX_TEX_WRAP_BORDER), bv = (wrapv == VX_TEX_WRAP_BORDER);
  if (!bu && !bv) return 0;
  int32_t dx = 0, dy = 0;
  if (filter == VX_TEX_FILTER_BILINEAR) {
    bool npot = TexIsNPOT(width, height, log_width, log_height);
    dx = npot ? (int32_t)((1 << (TEX_FXD_FRAC - 1)) / (int32_t)width)  : (int32_t)((1 << (TEX_FXD_FRAC - 1)) >> log_width);
    dy = npot ? (int32_t)((1 << (TEX_FXD_FRAC - 1)) / (int32_t)height) : (int32_t)((1 << (TEX_FXD_FRAC - 1)) >> log_height);
  }
  auto out = [&](int32_t c) { return (c < 0) || (c >= ONE); };
  const int32_t us[2] = { u - dx, u + dx };
  const int32_t vs[2] = { v - dy, v + dy };
  uint32_t mask = 0;
  // tap layout: 0=(us0,vs0) 1=(us1,vs0) 2=(us0,vs1) 3=(us1,vs1)
  const int uu[4] = {0, 1, 0, 1}, vv[4] = {0, 0, 1, 1};
  uint32_t taps = (filter == VX_TEX_FILTER_BILINEAR) ? 4u : 1u;
  for (uint32_t i = 0; i < taps; ++i) {
    bool border = (bu && out(us[uu[i]])) || (bv && out(vs[vv[i]]));
    if (border) mask |= (1u << i);
  }
  return mask;
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
