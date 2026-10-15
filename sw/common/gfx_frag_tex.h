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
#include "gfx_tex_bc.h"
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
  uint8_t  sub[4];    // a block-compressed tap's texel within the block at addr
  uint32_t stride;    // bytes per texel (1, 2, or 4), or per block
  uint32_t format;    // VX_TEX_FORMAT_*
  uint32_t filter;    // VX_TEX_FILTER_POINT or _BILINEAR
  uint32_t alpha;     // u-fraction (BILINEAR only)
  uint32_t beta;      // v-fraction (BILINEAR only)
  // CLAMP_TO_BORDER: which taps left the texture and take `border` instead of
  // what their address points at. Zero unless an axis wraps to a border.
  uint32_t border_mask;
  uint32_t border;    // ARGB8888
};

// The two taps one axis of a sample reads, for a level `dim` texels wide (any
// width, not only a power of two). The coordinate is folded by its wrap in
// normalized space -- repeat and mirror into [0,1), clamp, border and
// mirror-clamp into [-1,2), which keeps every out-of-range tap out of range --
// then scaled to texels, with bilinear moving it half a texel down. The taps are
// floor(x) and floor(x)+1, each wrapped back into [0,dim) as an integer; `frac`
// is the weight of tap 1 in 1/256ths. The TEX unit computes exactly this.
struct TexAxisTaps {
  uint32_t i0, i1;
  uint32_t frac;
  bool     b0, b1;   // the tap left the texture on a border axis
};

static inline __attribute__((always_inline)) TexAxisTaps tex_axis_taps(
    int32_t c, uint32_t dim, uint32_t wrap, bool linear) {
  const int32_t one = 1 << TEX_FXD_FRAC;
  int32_t f;
  switch (wrap) {
  case VX_TEX_WRAP_REPEAT:
    f = c & (one - 1);
    break;
  case VX_TEX_WRAP_MIRROR:
    f = (c ^ ((int32_t)((uint32_t)c << (31 - TEX_FXD_FRAC)) >> 31)) & (one - 1);
    break;
  default:
    f = (c < -one) ? -one : ((c > 2 * one - 1) ? 2 * one - 1 : c);
    break;
  }
  int64_t x = ((int64_t)f * dim) >> (TEX_FXD_FRAC - VX_TEX_SUBPIXEL_BITS);
  if (linear) {
    x -= 1 << (VX_TEX_SUBPIXEL_BITS - 1);
  }
  const int32_t i0 = (int32_t)(x >> VX_TEX_SUBPIXEL_BITS);
  auto wrap_texel = [&](int32_t i, bool* border) -> uint32_t {
    *border = false;
    if (i >= 0 && i < (int32_t)dim) {
      return (uint32_t)i;
    }
    if (wrap == VX_TEX_WRAP_REPEAT) {
      return (uint32_t)((i < 0) ? i + (int32_t)dim : i - (int32_t)dim);
    }
    if (wrap == VX_TEX_WRAP_MIRROR_CLAMP && i < 0) {
      i = -1 - i;
      return (i < (int32_t)dim) ? (uint32_t)i : dim - 1u;
    }
    *border = (wrap == VX_TEX_WRAP_BORDER);
    return (i < 0) ? 0u : dim - 1u;
  };
  TexAxisTaps t;
  t.i0   = wrap_texel(i0, &t.b0);
  t.i1   = wrap_texel(i0 + 1, &t.b1);
  t.frac = linear ? (uint32_t)(x & ((1 << VX_TEX_SUBPIXEL_BITS) - 1)) : 0u;
  return t;
}

// Formats past VX_TEX_FORMAT_R8UI are described field by field: each present
// channel is a bit field of the texel, with a numeric kind. They decode through
// the float path (TexDecodeFloat4): normalized fields to float, integer fields to
// their value carried as the lane's bits -- integer formats sample point-only, so
// the lanes pass through untouched.
// TEX_KIND_SRGB: 8-bit sRGB-encoded colour channels, linear alpha.
// TEX_KIND_FLOAT: a 32-bit float field, carried as is.
enum : uint32_t { TEX_KIND_UINT = 0, TEX_KIND_SINT = 1, TEX_KIND_UNORM = 2, TEX_KIND_SNORM = 3,
                  TEX_KIND_SRGB = 4, TEX_KIND_FLOAT = 5 };

struct TexFieldFmt {
  uint32_t bpp;
  uint32_t nch;
  uint32_t kind;
  uint8_t  off[4];
  uint8_t  width[4];
};

// The block-compressed formats: a texel is one of the 16 of a 4x4 block.
static inline bool TexIsBlockFormat(uint32_t format) {
  return format >= (uint32_t)VX_TEX_FORMAT_BC1 && format <= (uint32_t)VX_TEX_FORMAT_BC7_SRGB;
}

static inline uint32_t TexBlockBytes(uint32_t format) {
  switch (format) {
  case VX_TEX_FORMAT_BC1:
  case VX_TEX_FORMAT_BC1A:
  case VX_TEX_FORMAT_BC4:
  case VX_TEX_FORMAT_BC4SN:
  case VX_TEX_FORMAT_BC1_SRGB:
  case VX_TEX_FORMAT_BC1A_SRGB:
    return 8;
  default:
    return 16;
  }
}

static inline bool TexIsFieldFormat(uint32_t format) {
  return format >= (uint32_t)VX_TEX_FORMAT_R8UI && !TexIsBlockFormat(format);
}

static inline TexFieldFmt TexFieldGeom(uint32_t format) {
  switch (format) {
  case VX_TEX_FORMAT_R8UI: return { 1, 1, TEX_KIND_UINT, {0}, {8} };
  case VX_TEX_FORMAT_R8I: return { 1, 1, TEX_KIND_SINT, {0}, {8} };
  case VX_TEX_FORMAT_RG8UI: return { 2, 2, TEX_KIND_UINT, {0, 8}, {8, 8} };
  case VX_TEX_FORMAT_RG8I: return { 2, 2, TEX_KIND_SINT, {0, 8}, {8, 8} };
  case VX_TEX_FORMAT_RGBA8UI: return { 4, 4, TEX_KIND_UINT, {0, 8, 16, 24}, {8, 8, 8, 8} };
  case VX_TEX_FORMAT_RGBA8I: return { 4, 4, TEX_KIND_SINT, {0, 8, 16, 24}, {8, 8, 8, 8} };
  case VX_TEX_FORMAT_R16UI: return { 2, 1, TEX_KIND_UINT, {0}, {16} };
  case VX_TEX_FORMAT_R16I: return { 2, 1, TEX_KIND_SINT, {0}, {16} };
  case VX_TEX_FORMAT_RG16UI: return { 4, 2, TEX_KIND_UINT, {0, 16}, {16, 16} };
  case VX_TEX_FORMAT_RG16I: return { 4, 2, TEX_KIND_SINT, {0, 16}, {16, 16} };
  case VX_TEX_FORMAT_RGBA16UI: return { 8, 4, TEX_KIND_UINT, {0, 16, 32, 48}, {16, 16, 16, 16} };
  case VX_TEX_FORMAT_RGBA16I: return { 8, 4, TEX_KIND_SINT, {0, 16, 32, 48}, {16, 16, 16, 16} };
  case VX_TEX_FORMAT_R32UI: return { 4, 1, TEX_KIND_UINT, {0}, {32} };
  case VX_TEX_FORMAT_R32I: return { 4, 1, TEX_KIND_SINT, {0}, {32} };
  case VX_TEX_FORMAT_RG32UI: return { 8, 2, TEX_KIND_UINT, {0, 32}, {32, 32} };
  case VX_TEX_FORMAT_RG32I: return { 8, 2, TEX_KIND_SINT, {0, 32}, {32, 32} };
  case VX_TEX_FORMAT_RGBA32UI: return { 16, 4, TEX_KIND_UINT, {0, 32, 64, 96}, {32, 32, 32, 32} };
  case VX_TEX_FORMAT_RGBA32I: return { 16, 4, TEX_KIND_SINT, {0, 32, 64, 96}, {32, 32, 32, 32} };
  case VX_TEX_FORMAT_RGB10A2UI: return { 4, 4, TEX_KIND_UINT, {0, 10, 20, 30}, {10, 10, 10, 2} };
  case VX_TEX_FORMAT_R8SN: return { 1, 1, TEX_KIND_SNORM, {0}, {8} };
  case VX_TEX_FORMAT_RG8SN: return { 2, 2, TEX_KIND_SNORM, {0, 8}, {8, 8} };
  case VX_TEX_FORMAT_RGBA8SN: return { 4, 4, TEX_KIND_SNORM, {0, 8, 16, 24}, {8, 8, 8, 8} };
  case VX_TEX_FORMAT_RGB10A2: return { 4, 4, TEX_KIND_UNORM, {0, 10, 20, 30}, {10, 10, 10, 2} };
  case VX_TEX_FORMAT_R16: return { 2, 1, TEX_KIND_UNORM, {0}, {16} };
  case VX_TEX_FORMAT_RG16: return { 4, 2, TEX_KIND_UNORM, {0, 16}, {16, 16} };
  case VX_TEX_FORMAT_RGBA16: return { 8, 4, TEX_KIND_UNORM, {0, 16, 32, 48}, {16, 16, 16, 16} };
  case VX_TEX_FORMAT_R16SN: return { 2, 1, TEX_KIND_SNORM, {0}, {16} };
  case VX_TEX_FORMAT_RG16SN: return { 4, 2, TEX_KIND_SNORM, {0, 16}, {16, 16} };
  case VX_TEX_FORMAT_RGBA16SN: return { 8, 4, TEX_KIND_SNORM, {0, 16, 32, 48}, {16, 16, 16, 16} };
  case VX_TEX_FORMAT_R8: return { 1, 1, TEX_KIND_UNORM, {0}, {8} };
  case VX_TEX_FORMAT_RG8: return { 2, 2, TEX_KIND_UNORM, {0, 8}, {8, 8} };
  case VX_TEX_FORMAT_RGBA8: return { 4, 4, TEX_KIND_UNORM, {0, 8, 16, 24}, {8, 8, 8, 8} };
  case VX_TEX_FORMAT_BGRA8: return { 4, 4, TEX_KIND_UNORM, {16, 8, 0, 24}, {8, 8, 8, 8} };
  case VX_TEX_FORMAT_SRGBA8: return { 4, 4, TEX_KIND_SRGB, {0, 8, 16, 24}, {8, 8, 8, 8} };
  case VX_TEX_FORMAT_SBGRA8: return { 4, 4, TEX_KIND_SRGB, {16, 8, 0, 24}, {8, 8, 8, 8} };
  case VX_TEX_FORMAT_D24: return { 4, 1, TEX_KIND_UNORM, {0}, {24} };
  case VX_TEX_FORMAT_X24S8: return { 4, 1, TEX_KIND_UINT, {24}, {8} };
  case VX_TEX_FORMAT_S8X24: return { 4, 1, TEX_KIND_UINT, {0}, {8} };
  case VX_TEX_FORMAT_D32S8: return { 8, 1, TEX_KIND_FLOAT, {0}, {32} };
  case VX_TEX_FORMAT_X32S8: return { 8, 1, TEX_KIND_UINT, {32}, {8} };
  case VX_TEX_FORMAT_B5G6R5_PACK16: return { 2, 3, TEX_KIND_UNORM, {0, 5, 11}, {5, 6, 5} };
  case VX_TEX_FORMAT_R5G5B5A1_PACK16: return { 2, 4, TEX_KIND_UNORM, {11, 6, 1, 0}, {5, 5, 5, 1} };
  case VX_TEX_FORMAT_B5G5R5A1_PACK16: return { 2, 4, TEX_KIND_UNORM, {1, 6, 11, 0}, {5, 5, 5, 1} };
  case VX_TEX_FORMAT_A1B5G5R5_PACK16: return { 2, 4, TEX_KIND_UNORM, {0, 5, 10, 15}, {5, 5, 5, 1} };
  case VX_TEX_FORMAT_A4R4G4B4_PACK16: return { 2, 4, TEX_KIND_UNORM, {8, 4, 0, 12}, {4, 4, 4, 4} };
  case VX_TEX_FORMAT_A4B4G4R4_PACK16: return { 2, 4, TEX_KIND_UNORM, {0, 4, 8, 12}, {4, 4, 4, 4} };
  case VX_TEX_FORMAT_A8_UNORM: return { 1, 4, TEX_KIND_UNORM, {0, 0, 0, 0}, {0, 0, 0, 8} };
  case VX_TEX_FORMAT_BGRA8UI: return { 4, 4, TEX_KIND_UINT, {16, 8, 0, 24}, {8, 8, 8, 8} };
  case VX_TEX_FORMAT_BGRA8I: return { 4, 4, TEX_KIND_SINT, {16, 8, 0, 24}, {8, 8, 8, 8} };
  case VX_TEX_FORMAT_BGRA8SN: return { 4, 4, TEX_KIND_SNORM, {16, 8, 0, 24}, {8, 8, 8, 8} };
  case VX_TEX_FORMAT_A2R10G10B10: return { 4, 4, TEX_KIND_UNORM, {20, 10, 0, 30}, {10, 10, 10, 2} };
  case VX_TEX_FORMAT_A2R10G10B10SN: return { 4, 4, TEX_KIND_SNORM, {20, 10, 0, 30}, {10, 10, 10, 2} };
  case VX_TEX_FORMAT_A2R10G10B10UI: return { 4, 4, TEX_KIND_UINT, {20, 10, 0, 30}, {10, 10, 10, 2} };
  case VX_TEX_FORMAT_A2B10G10R10SN: return { 4, 4, TEX_KIND_SNORM, {0, 10, 20, 30}, {10, 10, 10, 2} };
  case VX_TEX_FORMAT_RGB32UI: return { 12, 3, TEX_KIND_UINT, {0, 32, 64}, {32, 32, 32} };
  case VX_TEX_FORMAT_RGB32I: return { 12, 3, TEX_KIND_SINT, {0, 32, 64}, {32, 32, 32} };
  case VX_TEX_FORMAT_RGB32F: return { 12, 3, TEX_KIND_FLOAT, {0, 32, 64}, {32, 32, 32} };
  case VX_TEX_FORMAT_R5G6B5_PACK16: return { 2, 3, TEX_KIND_UNORM, {11, 5, 0}, {5, 6, 5} };
  case VX_TEX_FORMAT_A1R5G5B5_PACK16: return { 2, 4, TEX_KIND_UNORM, {10, 5, 0, 15}, {5, 5, 5, 1} };
  default: return { 4, 4, TEX_KIND_UNORM, {0, 8, 16, 24}, {8, 8, 8, 8} };
  }
}

// One field of a texel held as up to four little-endian words. No field of these
// formats straddles a word.
static inline uint32_t TexFieldGet(const uint32_t w[4], uint32_t off, uint32_t width) {
  uint32_t v = w[off >> 5] >> (off & 31u);
  return (width >= 32u) ? v : (v & ((1u << width) - 1u));
}

static inline float SrgbToLinearF(uint32_t c);

// Decode one field-format texel to four lanes: float for normalized formats,
// the integer's bits for integer formats. Absent channels read (0,0,1).
static inline void TexDecodeField4(uint32_t format, const void* p, float out[4]) {
  const TexFieldFmt g = TexFieldGeom(format);
  uint32_t w[4] = {0, 0, 0, 0};
  __builtin_memcpy(w, p, g.bpp);
  const bool is_int = (g.kind == TEX_KIND_UINT || g.kind == TEX_KIND_SINT);
  for (uint32_t c = 0; c < 4; ++c) {
    uint32_t lane;
    if (c >= g.nch) {
      if (c == 3u) {
        float one = 1.0f;
        __builtin_memcpy(&lane, &one, 4);
        lane = is_int ? 1u : lane;
      } else {
        lane = 0u;
      }
    } else {
      const uint32_t width = g.width[c];
      if (width == 0) {
        __builtin_memset(&out[c], 0, 4);   // a channel the format lacks
        continue;
      }
      const uint32_t v = TexFieldGet(w, g.off[c], width);
      const uint32_t sh = 32u - width;
      float f;
      switch (g.kind) {
      case TEX_KIND_SINT:
        lane = (width >= 32u) ? v : (uint32_t)(((int32_t)(v << sh)) >> sh);
        break;
      case TEX_KIND_UNORM:
        f = (float)v * (1.0f / (float)((1u << width) - 1u));
        __builtin_memcpy(&lane, &f, 4);
        break;
      case TEX_KIND_SRGB:
        f = (c < 3u) ? SrgbToLinearF(v) : (float)v * (1.0f / 255.0f);
        __builtin_memcpy(&lane, &f, 4);
        break;
      case TEX_KIND_SNORM:
        f = (float)(((int32_t)(v << sh)) >> sh) * (1.0f / (float)((1u << (width - 1u)) - 1u));
        f = (f < -1.0f) ? -1.0f : f;
        __builtin_memcpy(&lane, &f, 4);
        break;
      default:
        lane = v;
        break;
      }
    }
    __builtin_memcpy(&out[c], &lane, 4);
  }
}

// Bytes per texel, or per block of a block-compressed format.
static inline uint32_t FormatStride(uint32_t format) {
  if (TexIsBlockFormat(format)) {
    return TexBlockBytes(format);
  }
  if (TexIsFieldFormat(format)) {
    return TexFieldGeom(format).bpp;
  }
  switch (format) {
  default:
  case VX_TEX_FORMAT_A8R8G8B8:
  case VX_TEX_FORMAT_SRGB8:      // R8G8B8(X8) sRGB, 4 bytes
  case VX_TEX_FORMAT_SRGB8A8:    // R8G8B8A8 sRGB
  case VX_TEX_FORMAT_R32F:       // single float
  case VX_TEX_FORMAT_D32F:       // float depth
  case VX_TEX_FORMAT_R11G11B10F: // packed unsigned floats
  case VX_TEX_FORMAT_RGB9E5:     // shared-exponent floats
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

// The TEX unit decodes only formats 0..VX_TEX_FORMAT_FF_MAX; everything else is
// the on-device software sampler.
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

// Unsigned small float (5-bit exponent, bias 15, `mbits`-bit mantissa, no sign)
// → float: the 11- and 10-bit channels of R11G11B10F.
static inline float UFloatToFloat(uint32_t v, uint32_t mbits) {
  uint32_t exp = (v >> mbits) & 0x1f;
  uint32_t man = v & ((1u << mbits) - 1);
  uint32_t bits;
  if (exp == 0) {
    // subnormal: man * 2^(-14 - mbits), built as man * (2^-14 / 2^mbits)
    uint32_t sbits = (uint32_t)(127 - 14 - (int32_t)mbits) << 23;
    float scale;
    __builtin_memcpy(&scale, &sbits, 4);
    return (float)man * scale;
  } else if (exp == 0x1f) {
    bits = 0x7f800000u | (man << (23 - mbits));   // inf / nan
  } else {
    bits = ((exp - 15 + 127) << 23) | (man << (23 - mbits));
  }
  float f;
  __builtin_memcpy(&f, &bits, 4);
  return f;
}

// RGB9E5: three 9-bit mantissas sharing a 5-bit exponent (bias 15) with no
// implicit leading one: channel = man * 2^(exp - 24).
static inline void Rgb9e5ToFloat3(uint32_t t, float out[3]) {
  uint32_t sbits = (((t >> 27) & 0x1f) + 127 - 24) << 23;
  float scale;
  __builtin_memcpy(&scale, &sbits, 4);
  out[0] = (float)(t & 0x1ff) * scale;
  out[1] = (float)((t >> 9) & 0x1ff) * scale;
  out[2] = (float)((t >> 18) & 0x1ff) * scale;
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
  case VX_TEX_FORMAT_R11G11B10F: {
    uint32_t t; __builtin_memcpy(&t, b, 4);
    r   = FloatToUnorm8(UFloatToFloat(t & 0x7ff, 6));
    g   = FloatToUnorm8(UFloatToFloat((t >> 11) & 0x7ff, 6));
    bch = FloatToUnorm8(UFloatToFloat(t >> 22, 5));
    a   = 0xff;
    break;
  }
  case VX_TEX_FORMAT_RGB9E5: {
    uint32_t t; __builtin_memcpy(&t, b, 4);
    float f[3];
    Rgb9e5ToFloat3(t, f);
    r = FloatToUnorm8(f[0]); g = FloatToUnorm8(f[1]); bch = FloatToUnorm8(f[2]); a = 0xff;
    break;
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

// A shadow sampler's depth texel as the compare takes it: a unorm code of `bits`
// bits, or (bits == 0) a float's bits. `word` is the texel's first four bytes.
// Any other format compares its low 24 bits as a D24 texel, as the TEX unit's
// compare stage does; Vulkan defines a compare on depth formats only.
struct TexDepthTexel {
  uint32_t code;
  uint32_t bits;
};
static inline TexDepthTexel TexDepthFromWord(uint32_t format, uint32_t word) {
  switch (format) {
  case VX_TEX_FORMAT_D16:   return { word & 0xffffu, 16 };
  case VX_TEX_FORMAT_D24:   return { word & 0xffffffu, 24 };
  case VX_TEX_FORMAT_D32F:
  case VX_TEX_FORMAT_D32S8: return { word, 0 };
  default:                  return { word & 0xffffffu, 24 };
  }
}

// `ref <func> depth` (VX_OM_DEPTH_FUNC_*), exact over the reals. A unorm texel d
// of n bits stands for d / (2^n - 1); the reference, clamped to [0, 1], compares
// as ref * (2^n - 1) against d, formed exactly as its integer part and whether a
// fraction remains, so no division or rounding enters. A float texel compares as
// floats, -0 equal to +0 and a NaN equal to nothing.
static inline bool TexDepthCompare(uint32_t func, uint32_t ref_bits, TexDepthTexel t) {
  bool lt, eq, gt;
  if (t.bits == 0) {
    float r, d;
    __builtin_memcpy(&r, &ref_bits, 4);
    __builtin_memcpy(&d, &t.code, 4);
    lt = r < d;
    eq = r == d;
    gt = r > d;
  } else {
    const uint64_t k = (1ull << t.bits) - 1u;
    const uint32_t mag = ref_bits & 0x7fffffffu;
    uint64_t whole = 0;
    bool frac = false;
    if ((ref_bits >> 31) || mag == 0 || mag > 0x7f800000u) {
      // negative, zero or NaN: clamps to 0
    } else if (mag >= 0x3f800000u) {
      whole = k;   // 1 or more: clamps to 1
    } else {
      const uint32_t e = mag >> 23;
      const uint64_t m = (mag & 0x7fffffu) | (e ? 0x800000u : 0u);
      const uint32_t shift = 150u - (e ? e : 1u);   // ref = m / 2^shift, shift >= 24
      const uint64_t prod = m * k;                   // < 2^48
      whole = shift < 64 ? prod >> shift : 0;
      frac = shift < 64 ? (prod & ((1ull << shift) - 1u)) != 0 : prod != 0;
    }
    lt = whole < t.code;
    eq = whole == t.code && !frac;
    gt = whole > t.code || (whole == t.code && frac);
  }
  switch (func) {
  case VX_OM_DEPTH_FUNC_NEVER:    return false;
  case VX_OM_DEPTH_FUNC_LESS:     return lt;
  case VX_OM_DEPTH_FUNC_EQUAL:    return eq;
  case VX_OM_DEPTH_FUNC_LEQUAL:   return lt || eq;
  case VX_OM_DEPTH_FUNC_GREATER:  return gt;
  case VX_OM_DEPTH_FUNC_NOTEQUAL: return !eq;
  case VX_OM_DEPTH_FUNC_GEQUAL:   return gt || eq;
  default:                        return true;
  }
}

// The float sampled formats (VX_TEX_FORMAT_R16F..RGBA32F plus the packed
// R11G11B10F and RGB9E5). Their texels are not confined to [0,1], so decoding or
// filtering them in the 8-bit ARGB working space clamps them; a sampler must take
// the float path for these.
static inline bool TexIsFloatFormat(uint32_t format) {
  return (format >= (uint32_t)VX_TEX_FORMAT_R16F
       && format <= (uint32_t)VX_TEX_FORMAT_RGBA32F)
      || format == (uint32_t)VX_TEX_FORMAT_R11G11B10F
      || format == (uint32_t)VX_TEX_FORMAT_RGB9E5;
}

static inline bool TexIsSrgbFormat(uint32_t format) {
  return format == (uint32_t)VX_TEX_FORMAT_SRGB8
      || format == (uint32_t)VX_TEX_FORMAT_SRGB8A8;
}

// Formats a sampler must decode and filter in float: the float and field formats,
// sRGB -- Vulkan converts sRGB to linear ahead of filtering, and an 8-bit linear
// result cannot carry it (the darkest 7 sRGB codes all land on 0) -- and depth,
// whose 16 and 32 bits an 8-bit working space would truncate.
static inline bool TexFiltersInFloat(uint32_t format) {
  return TexIsFloatFormat(format) || TexIsSrgbFormat(format) || TexIsFieldFormat(format)
      || TexIsBlockFormat(format)
      || format == VX_TEX_FORMAT_D16 || format == VX_TEX_FORMAT_D32F;
}

// sRGB EOTF (IEC 61966-2-1), 8-bit code -> linear float, correctly rounded to f32.
// Table keeps the header freestanding -- no powf on the baremetal device.
static inline float SrgbToLinearF(uint32_t c) {
  static const float kSrgbLutF[256] = {
      0.0f, 0.000303526991f, 0.000607053982f, 0.000910580973f, 0.00121410796f, 0.00151763496f,
      0.00182116195f, 0.00212468882f, 0.00242821593f, 0.0027317428f, 0.00303526991f, 0.00334653584f,
      0.00367650739f, 0.00402471703f, 0.00439144205f, 0.00477695325f, 0.00518151652f, 0.00560539169f,
      0.00604883302f, 0.00651209056f, 0.00699541019f, 0.00749903219f, 0.00802319311f, 0.00856812578f,
      0.00913405884f, 0.00972121768f, 0.010329823f, 0.0109600937f, 0.0116122449f, 0.012286488f,
      0.0129830325f, 0.0137020834f, 0.0144438436f, 0.0152085144f, 0.0159962941f, 0.0168073755f,
      0.0176419541f, 0.01850022f, 0.0193823613f, 0.0202885624f, 0.0212190095f, 0.0221738853f,
      0.0231533665f, 0.0241576321f, 0.0251868591f, 0.0262412224f, 0.0273208916f, 0.02842604f,
      0.0295568351f, 0.0307134446f, 0.0318960324f, 0.0331047662f, 0.0343398079f, 0.0356013142f,
      0.0368894488f, 0.0382043719f, 0.0395462364f, 0.0409151986f, 0.0423114114f, 0.043735031f,
      0.045186203f, 0.0466650873f, 0.0481718257f, 0.0497065671f, 0.0512694567f, 0.0528606474f,
      0.054480277f, 0.0561284907f, 0.0578054301f, 0.0595112368f, 0.0612460524f, 0.0630100146f,
      0.064803265f, 0.0666259378f, 0.0684781671f, 0.0703600943f, 0.0722718537f, 0.0742135718f,
      0.0761853829f, 0.078187421f, 0.0802198201f, 0.0822827071f, 0.0843762085f, 0.0865004584f,
      0.0886555836f, 0.0908417106f, 0.0930589661f, 0.0953074694f, 0.097587347f, 0.0998987257f,
      0.102241732f, 0.104616486f, 0.107023105f, 0.10946171f, 0.111932427f, 0.114435375f,
      0.116970666f, 0.119538426f, 0.122138776f, 0.124771819f, 0.127437681f, 0.130136475f,
      0.13286832f, 0.135633335f, 0.138431609f, 0.141263291f, 0.144128472f, 0.147027269f,
      0.149959788f, 0.152926147f, 0.155926466f, 0.158960834f, 0.162029371f, 0.165132195f,
      0.168269396f, 0.171441108f, 0.174647406f, 0.177888423f, 0.18116425f, 0.18447499f,
      0.187820777f, 0.191201687f, 0.194617838f, 0.198069319f, 0.20155625f, 0.205078736f,
      0.208636865f, 0.212230757f, 0.215860501f, 0.219526201f, 0.223227963f, 0.226965874f,
      0.230740055f, 0.23455058f, 0.238397568f, 0.242281124f, 0.246201321f, 0.25015828f,
      0.254152089f, 0.258182853f, 0.262250662f, 0.266355604f, 0.270497799f, 0.274677306f,
      0.278894275f, 0.283148736f, 0.287440836f, 0.291770637f, 0.296138257f, 0.300543785f,
      0.304987311f, 0.309468925f, 0.313988715f, 0.318546772f, 0.323143214f, 0.327778101f,
      0.332451522f, 0.337163627f, 0.341914415f, 0.346704066f, 0.351532608f, 0.356400132f,
      0.361306787f, 0.366252601f, 0.371237695f, 0.376262128f, 0.38132602f, 0.386429429f,
      0.391572475f, 0.396755219f, 0.401977777f, 0.407240212f, 0.412542611f, 0.417885065f,
      0.423267663f, 0.428690493f, 0.434153646f, 0.439657182f, 0.445201188f, 0.450785786f,
      0.456411034f, 0.462076992f, 0.467783809f, 0.473531485f, 0.479320168f, 0.48514995f,
      0.491020858f, 0.496932983f, 0.502886474f, 0.50888133f, 0.514917672f, 0.520995557f,
      0.527115107f, 0.533276379f, 0.539479494f, 0.545724452f, 0.55201143f, 0.558340371f,
      0.564711511f, 0.571124852f, 0.577580452f, 0.584078431f, 0.590618849f, 0.597201765f,
      0.603827357f, 0.610495567f, 0.617206573f, 0.623960376f, 0.630757153f, 0.637596846f,
      0.644479692f, 0.651405632f, 0.658374846f, 0.665387273f, 0.672443151f, 0.679542482f,
      0.686685324f, 0.693871737f, 0.701101899f, 0.708375752f, 0.715693474f, 0.723055124f,
      0.730460763f, 0.73791039f, 0.745404184f, 0.752942204f, 0.760524511f, 0.768151164f,
      0.775822222f, 0.783537805f, 0.791297913f, 0.799102724f, 0.806952238f, 0.814846575f,
      0.822785735f, 0.830769897f, 0.838799f, 0.846873224f, 0.854992628f, 0.863157213f,
      0.871367097f, 0.8796224f, 0.887923121f, 0.896269381f, 0.904661179f, 0.913098633f,
      0.921581864f, 0.930110872f, 0.938685715f, 0.947306514f, 0.955973327f, 0.964686275f,
      0.973445296f, 0.982250571f, 0.991102099f, 1.0f};
  return kSrgbLutF[c & 0xff];
}

// Decode one texel of ANY format at `p` (stride bytes) to four float channels in
// RGBA order. A float format keeps its real magnitude — its value range is not
// [0,1], so routing it through the 8-bit ARGB working space would clamp the texel
// and destroy the range — sRGB keeps the precision of its linear value, and a
// field format decodes through TexDecodeField4. Every other format decodes
// through TexDecodeArgb8 and scales by the same multiply-by-reciprocal the shader
// unpack uses, so the result is bit-identical to the 8-bit path. Absent channels
// read (0,0,1) for g/b/a, as in TexDecodeExtended.
static inline void TexDecodeFloat4(uint32_t format, const void* p, uint32_t stride,
                                   float out[4]) {
  const uint8_t* b = (const uint8_t*)p;
  if (TexIsFieldFormat(format)) {
    TexDecodeField4(format, p, out);
    return;
  }
  out[0] = 0.0f; out[1] = 0.0f; out[2] = 0.0f; out[3] = 1.0f;
  switch (format) {
  case VX_TEX_FORMAT_R16F: {
    uint16_t h; __builtin_memcpy(&h, b, 2);
    out[0] = HalfToFloat(h);
    return;
  }
  case VX_TEX_FORMAT_RG16F: {
    uint16_t h[2]; __builtin_memcpy(h, b, 4);
    out[0] = HalfToFloat(h[0]); out[1] = HalfToFloat(h[1]);
    return;
  }
  case VX_TEX_FORMAT_RGBA16F: {
    uint16_t h[4]; __builtin_memcpy(h, b, 8);
    for (uint32_t i = 0; i < 4; ++i) {
      out[i] = HalfToFloat(h[i]);
    }
    return;
  }
  case VX_TEX_FORMAT_R32F:
    __builtin_memcpy(&out[0], b, 4);
    return;
  case VX_TEX_FORMAT_RG32F:
    __builtin_memcpy(out, b, 8);
    return;
  case VX_TEX_FORMAT_RGBA32F:
    __builtin_memcpy(out, b, 16);
    return;
  case VX_TEX_FORMAT_R11G11B10F: {
    uint32_t t; __builtin_memcpy(&t, b, 4);
    out[0] = UFloatToFloat(t & 0x7ff, 6);
    out[1] = UFloatToFloat((t >> 11) & 0x7ff, 6);
    out[2] = UFloatToFloat(t >> 22, 5);
    return;
  }
  case VX_TEX_FORMAT_RGB9E5: {
    uint32_t t; __builtin_memcpy(&t, b, 4);
    Rgb9e5ToFloat3(t, out);
    return;
  }
  // A depth texel samples as (D, 0, 0, 1).
  case VX_TEX_FORMAT_D16: {
    uint16_t d; __builtin_memcpy(&d, b, 2);
    out[0] = (float)d * (1.0f / 65535.0f);
    return;
  }
  case VX_TEX_FORMAT_D32F:
    __builtin_memcpy(&out[0], b, 4);
    return;
  case VX_TEX_FORMAT_SRGB8:
  case VX_TEX_FORMAT_SRGB8A8: {
    // Same texel layout TexDecodeExtended reads: R@16, G@8, B@0, A@24 (linear).
    uint32_t t; __builtin_memcpy(&t, b, 4);
    out[0] = SrgbToLinearF((t >> 16) & 0xff);
    out[1] = SrgbToLinearF((t >> 8) & 0xff);
    out[2] = SrgbToLinearF(t & 0xff);
    if (format == VX_TEX_FORMAT_SRGB8A8) {
      out[3] = (float)(t >> 24) * (1.0f / 255.0f);
    }
    return;
  }
  default: break;
  }
  // Every other format: the 8-bit ARGB decode, scaled to [0,1].
  const uint32_t argb = TexDecodeArgb8(format, p, stride);
  const float inv255 = 1.0f / 255.0f;
  out[0] = (float)((argb >> 16) & 0xff) * inv255;
  out[1] = (float)((argb >> 8) & 0xff) * inv255;
  out[2] = (float)(argb & 0xff) * inv255;
  out[3] = (float)(argb >> 24) * inv255;
}

// Linear -> the nearest 8-bit sRGB code, by search of the decode table (the
// device has no powf): the code whose linear value is closest.
static inline uint32_t LinearToSrgb8(float v) {
  if (!(v > 0.0f)) {
    return 0;
  }
  if (v >= 1.0f) {
    return 255;
  }
  uint32_t lo = 0, hi = 255;
  while (hi - lo > 1) {
    const uint32_t mid = (lo + hi) >> 1;
    if (SrgbToLinearF(mid) <= v) {
      lo = mid;
    } else {
      hi = mid;
    }
  }
  return (v - SrgbToLinearF(lo) <= SrgbToLinearF(hi) - v) ? lo : hi;
}

// Encode four lanes (floats for a normalized or float format, the integers for
// an integer one) into the field-format texel at `p`, leaving its other bits
// alone. An integer out of its field's range saturates, as a typed store does.
static inline void TexEncodeField4(uint32_t format, void* p, const uint32_t lanes[4]) {
  const TexFieldFmt g = TexFieldGeom(format);
  uint32_t w[4] = {0, 0, 0, 0};
  __builtin_memcpy(w, p, g.bpp);
  for (uint32_t c = 0; c < g.nch; ++c) {
    const uint32_t width = g.width[c];
    if (width == 0) {
      continue;
    }
    float f;
    __builtin_memcpy(&f, &lanes[c], 4);
    uint32_t v;
    switch (g.kind) {
    case TEX_KIND_UNORM: {
      const float cf = f > 0.0f ? (f < 1.0f ? f : 1.0f) : 0.0f;
      v = (uint32_t)(cf * (float)((1u << width) - 1u) + 0.5f);
      break;
    }
    case TEX_KIND_SNORM: {
      const float cf = f > -1.0f ? (f < 1.0f ? f : 1.0f) : -1.0f;
      const float s = cf * (float)((1u << (width - 1u)) - 1u);
      v = (uint32_t)(int32_t)(s + (s < 0.0f ? -0.5f : 0.5f));
      break;
    }
    case TEX_KIND_SRGB: {
      const float cf = f > 0.0f ? (f < 1.0f ? f : 1.0f) : 0.0f;
      v = (c < 3u) ? LinearToSrgb8(cf) : (uint32_t)(cf * 255.0f + 0.5f);
      break;
    }
    case TEX_KIND_UINT:
      v = (width < 32u && lanes[c] > (1u << width) - 1u) ? (1u << width) - 1u : lanes[c];
      break;
    case TEX_KIND_SINT: {
      int32_t iv = (int32_t)lanes[c];
      if (width < 32u) {
        const int32_t hi = (int32_t)((1u << (width - 1u)) - 1u), lo = -hi - 1;
        iv = iv > hi ? hi : (iv < lo ? lo : iv);
      }
      v = (uint32_t)iv;
      break;
    }
    default:
      v = lanes[c];
      break;
    }
    const uint32_t m = (width >= 32u) ? 0xffffffffu : ((1u << width) - 1u);
    const uint32_t sh = g.off[c] & 31u;
    w[g.off[c] >> 5] = (w[g.off[c] >> 5] & ~(m << sh)) | ((v & m) << sh);
  }
  __builtin_memcpy(p, w, g.bpp);
}

// The linear twin of an sRGB block format (the format itself otherwise).
static inline uint32_t TexBlockLinear(uint32_t format) {
  switch (format) {
  case VX_TEX_FORMAT_BC1_SRGB:  return VX_TEX_FORMAT_BC1;
  case VX_TEX_FORMAT_BC1A_SRGB: return VX_TEX_FORMAT_BC1A;
  case VX_TEX_FORMAT_BC2_SRGB:  return VX_TEX_FORMAT_BC2;
  case VX_TEX_FORMAT_BC3_SRGB:  return VX_TEX_FORMAT_BC3;
  case VX_TEX_FORMAT_BC7_SRGB:  return VX_TEX_FORMAT_BC7;
  default:                      return format;
  }
}

// Texel `t` (0..15, row-major) of the block at `p` of a block-compressed format,
// as four float channels in RGBA order.
static inline void TexDecodeBlockF4(uint32_t format, const void* p, uint32_t t, float out[4]) {
  out[0] = 0.0f; out[1] = 0.0f; out[2] = 0.0f; out[3] = 1.0f;
  switch (format) {
  case VX_TEX_FORMAT_BC4:
  case VX_TEX_FORMAT_BC4SN:
    out[0] = BcRedBlock((const uint8_t*)p, t, format == VX_TEX_FORMAT_BC4SN);
    return;
  case VX_TEX_FORMAT_BC5:
  case VX_TEX_FORMAT_BC5SN:
    out[0] = BcRedBlock((const uint8_t*)p, t, format == VX_TEX_FORMAT_BC5SN);
    out[1] = BcRedBlock((const uint8_t*)p + 8, t, format == VX_TEX_FORMAT_BC5SN);
    return;
  case VX_TEX_FORMAT_BC6UF:
  case VX_TEX_FORMAT_BC6SF: {
    uint16_t h[3];
    BcDecodeBc6(p, t, format == VX_TEX_FORMAT_BC6SF, h);
    for (uint32_t c = 0; c < 3; ++c) {
      out[c] = HalfToFloat(h[c]);
    }
    return;
  }
  default:
    break;
  }
  const uint32_t linear = TexBlockLinear(format);
  uint32_t rgba;
  switch (linear) {
  case VX_TEX_FORMAT_BC1:  rgba = BcDecodeBc1(p, t, false); break;
  case VX_TEX_FORMAT_BC1A: rgba = BcDecodeBc1(p, t, true); break;
  case VX_TEX_FORMAT_BC2:  rgba = BcDecodeBc2(p, t); break;
  case VX_TEX_FORMAT_BC3:  rgba = BcDecodeBc3(p, t); break;
  default:                 rgba = BcDecodeBc7(p, t); break;
  }
  const bool srgb = linear != format;
  const float inv255 = 1.0f / 255.0f;
  for (uint32_t c = 0; c < 3; ++c) {
    const uint32_t v = (rgba >> (8 * c)) & 0xff;
    out[c] = srgb ? SrgbToLinearF(v) : (float)v * inv255;
  }
  out[3] = (float)(rgba >> 24) * inv255;
}

// A texel's byte offset within its level `width` texels wide, and its index in
// its block (0 for a format that is not block-compressed).
static inline uint64_t TexTexelOffset(uint32_t format, uint32_t width, uint32_t x, uint32_t y,
                                      uint32_t* sub) {
  if (TexIsBlockFormat(format)) {
    *sub = ((y & 3u) << 2) | (x & 3u);
    return ((uint64_t)(y >> 2) * ((width + 3u) >> 2) + (x >> 2)) * TexBlockBytes(format);
  }
  *sub = 0;
  return ((uint64_t)y * width + x) * FormatStride(format);
}

// Bytes of one w x h image of the format (a 2D level, or a 3D level's slice).
static inline uint64_t TexImageBytes(uint32_t format, uint32_t w, uint32_t h) {
  if (TexIsBlockFormat(format)) {
    return (uint64_t)((w + 3u) >> 2) * ((h + 3u) >> 2) * TexBlockBytes(format);
  }
  return (uint64_t)w * h * FormatStride(format);
}

// TexDecodeFloat4 for a texel located by TexTexelOffset: `sub` picks it from its
// block when the format is block-compressed.
static inline void TexDecodeFloat4At(uint32_t format, const void* p, uint32_t sub,
                                     uint32_t stride, float out[4]) {
  if (TexIsBlockFormat(format)) {
    TexDecodeBlockF4(format, p, sub, out);
    return;
  }
  TexDecodeFloat4(format, p, stride, out);
}

// TexDecodeArgb8 likewise, a block texel rounded to eight bits a channel.
static inline uint32_t TexDecodeArgb8At(uint32_t format, const void* p, uint32_t sub,
                                        uint32_t stride) {
  if (!TexIsBlockFormat(format)) {
    return TexDecodeArgb8(format, p, stride);
  }
  float f[4];
  TexDecodeBlockF4(format, p, sub, f);
  uint32_t b[4];
  for (uint32_t c = 0; c < 4; ++c) {
    const float v = f[c] < 0.0f ? 0.0f : (f[c] > 1.0f ? 1.0f : f[c]);
    b[c] = (uint32_t)(v * 255.0f + 0.5f);
  }
  return (b[3] << 24) | (b[0] << 16) | (b[1] << 8) | b[2];
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
// `width0`/`height0` are the mip-0 dims; 0 takes the power of two in `logdim`.
static inline TexelRequest tex_compute_request(uint64_t base_addr,
                                               uint32_t logdim,
                                               uint32_t format,
                                               uint32_t filter,
                                               uint32_t wrap,
                                               int32_t  u,
                                               int32_t  v,
                                               uint32_t lod,
                                               uint32_t width0 = 0,
                                               uint32_t height0 = 0,
                                               uint32_t border = 0) {
  const uint32_t w0 = width0  ? width0  : (1u << (logdim & 0xffff));
  const uint32_t h0 = height0 ? height0 : (1u << (logdim >> 16));
  const uint32_t width  = (uint32_t)tex_imax((int32_t)(w0 >> lod), 1);
  const uint32_t height = (uint32_t)tex_imax((int32_t)(h0 >> lod), 1);
  const bool linear = (filter == VX_TEX_FILTER_BILINEAR);
  const TexAxisTaps tu = tex_axis_taps(u, width, wrap & 0xffff, linear);
  const TexAxisTaps tv = tex_axis_taps(v, height, wrap >> 16, linear);

  TexelRequest req{};
  req.stride = FormatStride(format);
  req.format = format;
  req.filter = filter;
  req.border = border;
  req.alpha  = tu.frac;
  req.beta   = tv.frac;
  // tap order: 0=(u0,v0) 1=(u1,v0) 2=(u0,v1) 3=(u1,v1)
  const uint32_t xs[2] = { tu.i0, tu.i1 }, ys[2] = { tv.i0, tv.i1 };
  const bool bx[2] = { tu.b0, tu.b1 }, by[2] = { tv.b0, tv.b1 };
  const uint32_t taps = linear ? 4u : 1u;
  for (uint32_t i = 0; i < taps; ++i) {
    const uint32_t x = i & 1u, y = i >> 1;
    uint32_t sub;
    req.addr[i] = base_addr + TexTexelOffset(format, width, xs[x], ys[y], &sub);
    req.sub[i] = (uint8_t)sub;
    if (bx[x] || by[y]) {
      req.border_mask |= 1u << i;
    }
  }
  return req;
}

// Free filter: format-decode + bilinear/point combine of fetched texels.
static inline uint32_t tex_apply_filter(const TexelRequest& req, const uint32_t texels[4]) {
  if (req.border_mask != 0) {
    // A border tap has no texel to decode, so the decode has to happen first and
    // the filter then runs over already-decoded ARGB. That is a different shape
    // from the path below, which unpacks inside the filter — hence the split
    // rather than one path with a mux: samples that never asked for a border
    // keep the exact code they had.
    if (req.filter == VX_TEX_FILTER_BILINEAR) {
      uint32_t argb[4];
      for (uint32_t i = 0; i < 4; ++i) {
        argb[i] = ((req.border_mask >> i) & 0x1) ? req.border
                                                 : TexFilterPoint(req.format, texels[i]);
      }
      return TexFilterLinear(VX_TEX_FORMAT_A8R8G8B8, argb[0], argb[1], argb[2], argb[3],
                             req.alpha, req.beta);
    }
    if (req.border_mask & 0x1) {
      return req.border;
    }
  }
  if (req.filter == VX_TEX_FILTER_BILINEAR)
    return TexFilterLinear(req.format, texels[0], texels[1], texels[2], texels[3],
                           req.alpha, req.beta);
  return TexFilterPoint(req.format, texels[0]);
}

// A shadow sample at one level: each tap the request reads compared against
// `ref_bits` by `func` (TexDepthCompare, per tap in tex_compare_tap), and the
// 0/1 results blended by the request's weights. A border tap's depth is
// `req.border`, an f32 (VX_DCR_TEX_BORDER for a depth-compare stage), which a
// unorm format still compares against the reference clamped to [0, 1], a NaN
// taken as 0, as its texels do. Returns the result in Q16 (65536 is 1), which
// the 8-bit weights make exact, so its float is too.
static inline uint32_t tex_compare_tap(const TexelRequest& req, uint32_t i, uint32_t word,
                                       uint32_t func, uint32_t ref_bits) {
  if ((req.border_mask >> i) & 0x1) {
    uint32_t r = ref_bits;
    if (TexDepthFromWord(req.format, 0).bits != 0) {
      if ((r & 0x80000000u) || (r & 0x7fffffffu) > 0x7f800000u) {
        r = 0;
      } else if (r >= 0x3f800000u) {
        r = 0x3f800000u;
      }
    }
    return TexDepthCompare(func, r, TexDepthTexel{ req.border, 0 }) ? 1u : 0u;
  }
  return TexDepthCompare(func, ref_bits, TexDepthFromWord(req.format, word)) ? 1u : 0u;
}
static inline uint32_t tex_apply_compare(const TexelRequest& req, const uint32_t texels[4],
                                         uint32_t func, uint32_t ref_bits) {
  auto pass = [&](uint32_t i) { return tex_compare_tap(req, i, texels[i], func, ref_bits); };
  if (req.filter != VX_TEX_FILTER_BILINEAR) {
    return pass(0) << 16;
  }
  const uint32_t a = req.alpha, b = req.beta;
  return pass(0) * (256u - a) * (256u - b) + pass(1) * a * (256u - b)
       + pass(2) * (256u - a) * b + pass(3) * a * b;
}

// Per-channel min or max of two ARGB8888 words.
static inline uint32_t TexReduce8888(uint32_t a, uint32_t b, uint32_t mode) {
  uint32_t out = 0;
  for (uint32_t s = 0; s < 32; s += 8) {
    const uint32_t x = (a >> s) & 0xff, y = (b >> s) & 0xff;
    const uint32_t r = (mode == VX_TEX_REDUCTION_MIN) ? (x < y ? x : y) : (x > y ? x : y);
    out |= r << s;
  }
  return out;
}

// Min/max reduction of one level: the per-channel minimum (VX_TEX_REDUCTION_MIN)
// or maximum of its decoded taps whose blend weight is not zero -- tap 0 always,
// tap 1 when the u weight is set, tap 2 when the v weight is, tap 3 when both
// are. A border tap takes the border colour. The levels then combine through
// TexLodReduce.
static inline uint32_t tex_apply_reduce(const TexelRequest& req, const uint32_t texels[4],
                                        uint32_t mode) {
  auto tap = [&](uint32_t i) {
    return ((req.border_mask >> i) & 0x1) ? req.border : TexFilterPoint(req.format, texels[i]);
  };
  uint32_t out = tap(0);
  if (req.filter != VX_TEX_FILTER_BILINEAR) {
    return out;
  }
  for (uint32_t i = 1; i < 4; ++i) {
    if (((i & 1u) && req.alpha == 0) || ((i & 2u) && req.beta == 0)) {
      continue;
    }
    out = TexReduce8888(out, tap(i), mode);
  }
  return out;
}

// The level combine under a reduction: the upper level counts only when its
// weight is not zero.
static inline uint32_t TexLodReduce(uint32_t c0, uint32_t c1, uint32_t frac, uint32_t mode) {
  return (frac & 0xff) ? TexReduce8888(c0, c1, mode) : c0;
}

// textureGather: channel `comp` (0..3 = R,G,B,A) of the bilinear footprint's four
// taps, each decoded to ARGB8888 (a border tap takes the border colour), as bytes
// in GL gather order: taps (u-,v+), (u+,v+), (u+,v-), (u-,v-).
static inline uint32_t tex_apply_gather(const TexelRequest& req, const uint32_t texels[4],
                                        uint32_t comp) {
  static const uint32_t kShift[4] = { 16, 8, 0, 24 };
  static const uint32_t kOrder[4] = { 2, 3, 1, 0 };
  uint32_t out = 0;
  for (uint32_t i = 0; i < 4; ++i) {
    const uint32_t t = kOrder[i];
    const uint32_t argb = ((req.border_mask >> t) & 0x1) ? req.border
                                                         : TexFilterPoint(req.format, texels[t]);
    out |= ((argb >> kShift[comp & 3u]) & 0xffu) << (8 * i);
  }
  return out;
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

// Float twins of TexFilterLinear / TexLodLerp, for a texture whose texels leave
// [0,1] and so cannot be filtered in the 8-bit working space. They take the same
// x/256 blend weights the fixed-point path uses (tex_axis_taps' tap-1 weights,
// TexLodLerp's frac), so both paths pick the same tap weights and only the
// channel arithmetic differs.
static inline __attribute__((always_inline)) void TexFilterLinearF4(
    const float t00[4], const float t01[4], const float t10[4], const float t11[4],
    uint32_t alpha, uint32_t beta, float out[4]) {
  const float fa = (float)(alpha & 0xff) * (1.0f / 256.0f);
  const float fb = (float)(beta & 0xff) * (1.0f / 256.0f);
  for (uint32_t c = 0; c < 4; ++c) {
    const float c01 = t00[c] + (t01[c] - t00[c]) * fa;
    const float c23 = t10[c] + (t11[c] - t10[c]) * fa;
    out[c] = c01 + (c23 - c01) * fb;
  }
}

static inline __attribute__((always_inline)) void TexLodLerpF4(
    const float c0[4], const float c1[4], uint32_t frac, float out[4]) {
  const float f = (float)(frac & 0xff) * (1.0f / 256.0f);
  for (uint32_t c = 0; c < 4; ++c)
    out[c] = c0[c] + (c1[c] - c0[c]) * f;
}

} // namespace gfx_tex
