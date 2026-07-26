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

// gfx_v2 on-device SIMT software fallback.
//
// The always-correct path that runs on the SIMT cores when a fixed-function
// unit cannot represent a required feature — full residency forbids a host
// (llvmpipe) fallback, so the completeness path lives on the device. This
// header is the device-compilable `libgfx_sw`; it is ALSO included by the host
// FF models (sw/common/gfx_ff_model.cpp) so the per-fragment math has a single
// source of truth — the SW path matches the FF
// path bit-for-bit because it IS the same code.
//
// First component: the software output-merger — depth/stencil test,
// blend, logic-op, write-mask — replacing the vx_om fixed-function call site.
// Freestanding (no <algorithm>/<cmath>) so it compiles for the baremetal device.

#pragma once

#include <stdint.h>
#include <VX_types.h>
#include "gfx_frag_tex.h"   // shared tex-sampling math (FF model + SW path)
#include <cocogfx/include/color.hpp>
#include <cocogfx/include/math.hpp>

namespace gfx_sw {

using cocogfx::ColorARGB;
using cocogfx::Div255;

// Local min/max so the header stays freestanding (device baremetal has no
// <algorithm>); identical results to std::min/std::max for the integer use here.
template <typename T> static inline __attribute__((always_inline)) T sw_min(T a, T b) { return a < b ? a : b; }
template <typename T> static inline __attribute__((always_inline)) T sw_max(T a, T b) { return a > b ? a : b; }

// ── Pure per-fragment ops (single source of truth for FF model + SW path) ────

static inline __attribute__((always_inline)) bool DoCompare(uint32_t func, uint32_t a, uint32_t b) {
  switch (func) {
  case VX_OM_DEPTH_FUNC_NEVER:    return false;
  case VX_OM_DEPTH_FUNC_LESS:     return (a < b);
  case VX_OM_DEPTH_FUNC_EQUAL:    return (a == b);
  case VX_OM_DEPTH_FUNC_LEQUAL:   return (a <= b);
  case VX_OM_DEPTH_FUNC_GREATER:  return (a > b);
  case VX_OM_DEPTH_FUNC_NOTEQUAL: return (a != b);
  case VX_OM_DEPTH_FUNC_GEQUAL:   return (a >= b);
  case VX_OM_DEPTH_FUNC_ALWAYS:   default: return true;
  }
}

static inline __attribute__((always_inline)) uint32_t DoStencilOp(uint32_t op, uint32_t ref, uint32_t val) {
  switch (op) {
  case VX_OM_STENCIL_OP_ZERO:      return 0;
  case VX_OM_STENCIL_OP_REPLACE:   return ref;
  case VX_OM_STENCIL_OP_INCR:      return (val < 0xff) ? (val + 1) : val;
  case VX_OM_STENCIL_OP_DECR:      return (val > 0) ? (val - 1) : val;
  case VX_OM_STENCIL_OP_INVERT:    return ~val;
  case VX_OM_STENCIL_OP_INCR_WRAP: return (val + 1) & 0xff;
  case VX_OM_STENCIL_OP_DECR_WRAP: return (val - 1) & 0xff;
  case VX_OM_STENCIL_OP_KEEP:      default: return val;
  }
}

static inline __attribute__((always_inline)) uint32_t DoLogicOp(uint32_t op, uint32_t src, uint32_t dst) {
  switch (op) {
  case VX_OM_LOGIC_OP_CLEAR:        return 0;
  case VX_OM_LOGIC_OP_AND:          return src & dst;
  case VX_OM_LOGIC_OP_AND_REVERSE:  return src & ~dst;
  case VX_OM_LOGIC_OP_COPY:         return src;
  case VX_OM_LOGIC_OP_AND_INVERTED: return ~src & dst;
  case VX_OM_LOGIC_OP_NOOP:         return dst;
  case VX_OM_LOGIC_OP_XOR:          return src ^ dst;
  case VX_OM_LOGIC_OP_OR:           return src | dst;
  case VX_OM_LOGIC_OP_NOR:          return ~(src | dst);
  case VX_OM_LOGIC_OP_EQUIV:        return ~(src ^ dst);
  case VX_OM_LOGIC_OP_INVERT:       return ~dst;
  case VX_OM_LOGIC_OP_OR_REVERSE:   return src | ~dst;
  case VX_OM_LOGIC_OP_COPY_INVERTED:return ~src;
  case VX_OM_LOGIC_OP_OR_INVERTED:  return ~src | dst;
  case VX_OM_LOGIC_OP_NAND:         return ~(src & dst);
  case VX_OM_LOGIC_OP_SET:          default: return 0xffffffff;
  }
}

static inline __attribute__((always_inline)) ColorARGB DoBlendFunc(uint32_t func, ColorARGB src, ColorARGB dst, ColorARGB cst) {
  switch (func) {
  case VX_OM_BLEND_FUNC_ZERO:               return ColorARGB(0, 0, 0, 0);
  case VX_OM_BLEND_FUNC_ONE:                return ColorARGB(0xff, 0xff, 0xff, 0xff);
  case VX_OM_BLEND_FUNC_SRC_RGB:            return src;
  case VX_OM_BLEND_FUNC_ONE_MINUS_SRC_RGB:  return ColorARGB(0xff - src.a, 0xff - src.r, 0xff - src.g, 0xff - src.b);
  case VX_OM_BLEND_FUNC_DST_RGB:            return dst;
  case VX_OM_BLEND_FUNC_ONE_MINUS_DST_RGB:  return ColorARGB(0xff - dst.a, 0xff - dst.r, 0xff - dst.g, 0xff - dst.b);
  case VX_OM_BLEND_FUNC_SRC_A:              return ColorARGB(src.a, src.a, src.a, src.a);
  case VX_OM_BLEND_FUNC_ONE_MINUS_SRC_A:    return ColorARGB(0xff - src.a, 0xff - src.a, 0xff - src.a, 0xff - src.a);
  case VX_OM_BLEND_FUNC_DST_A:              return ColorARGB(dst.a, dst.a, dst.a, dst.a);
  case VX_OM_BLEND_FUNC_ONE_MINUS_DST_A:    return ColorARGB(0xff - dst.a, 0xff - dst.a, 0xff - dst.a, 0xff - dst.a);
  case VX_OM_BLEND_FUNC_CONST_RGB:          return cst;
  case VX_OM_BLEND_FUNC_ONE_MINUS_CONST_RGB:return ColorARGB(0xff - cst.a, 0xff - cst.r, 0xff - cst.g, 0xff - cst.b);
  case VX_OM_BLEND_FUNC_CONST_A:            return ColorARGB(cst.a, cst.a, cst.a, cst.a);
  case VX_OM_BLEND_FUNC_ONE_MINUS_CONST_A:  return ColorARGB(0xff - cst.a, 0xff - cst.a, 0xff - cst.a, 0xff - cst.a);
  case VX_OM_BLEND_FUNC_ALPHA_SAT: {
    int factor = sw_min<int>(src.a, 0xff - dst.a);
    return ColorARGB(0xff, factor, factor, factor);
  }
  default:                                  return ColorARGB(0, 0, 0, 0);
  }
}

static inline __attribute__((always_inline)) ColorARGB DoBlendMode(uint32_t mode, uint32_t logic_op,
                                    ColorARGB src, ColorARGB dst, ColorARGB s, ColorARGB d) {
  switch (mode) {
  case VX_OM_BLEND_MODE_ADD:
    return ColorARGB(
      Div255(sw_min<int>(src.a * s.a + dst.a * d.a + 0x80, 0xFF00)),
      Div255(sw_min<int>(src.r * s.r + dst.r * d.r + 0x80, 0xFF00)),
      Div255(sw_min<int>(src.g * s.g + dst.g * d.g + 0x80, 0xFF00)),
      Div255(sw_min<int>(src.b * s.b + dst.b * d.b + 0x80, 0xFF00)));
  case VX_OM_BLEND_MODE_SUB:
    return ColorARGB(
      Div255(sw_max<int>(src.a * s.a - dst.a * d.a + 0x80, 0x0)),
      Div255(sw_max<int>(src.r * s.r - dst.r * d.r + 0x80, 0x0)),
      Div255(sw_max<int>(src.g * s.g - dst.g * d.g + 0x80, 0x0)),
      Div255(sw_max<int>(src.b * s.b - dst.b * d.b + 0x80, 0x0)));
  case VX_OM_BLEND_MODE_REV_SUB:
    return ColorARGB(
      Div255(sw_max<int>(dst.a * d.a - src.a * s.a + 0x80, 0x0)),
      Div255(sw_max<int>(dst.r * d.r - src.r * s.r + 0x80, 0x0)),
      Div255(sw_max<int>(dst.g * d.g - src.g * s.g + 0x80, 0x0)),
      Div255(sw_max<int>(dst.b * d.b - src.b * s.b + 0x80, 0x0)));
  case VX_OM_BLEND_MODE_MIN:
    return ColorARGB(sw_min(src.a, dst.a), sw_min(src.r, dst.r), sw_min(src.g, dst.g), sw_min(src.b, dst.b));
  case VX_OM_BLEND_MODE_MAX:
    return ColorARGB(sw_max(src.a, dst.a), sw_max(src.r, dst.r), sw_max(src.g, dst.g), sw_max(src.b, dst.b));
  case VX_OM_BLEND_MODE_LOGICOP:
    return ColorARGB(DoLogicOp(logic_op, src.value, dst.value));
  default:
    return src;
  }
}

// ── Resolved output-merger state (host fills from the same values it programs
//    into the OM DCRs; resolve_om_state() derives the enable flags / masks the
//    same way the fixed-function OM does) ─────────────────────────────────────
struct om_state_t {
  // depth/stencil (index 0 = front face, 1 = back face)
  uint32_t depth_func;
  uint32_t stencil_func[2], stencil_zpass[2], stencil_zfail[2], stencil_fail[2];
  uint32_t stencil_ref[2], stencil_mask[2], stencil_writemask[2];
  uint32_t depth_writemask;            // 0/1
  // blend
  uint32_t blend_mode_rgb, blend_mode_a;
  uint32_t blend_src_rgb, blend_src_a, blend_dst_rgb, blend_dst_a;
  uint32_t blend_const, logic_op;
  // framebuffer (device byte addresses)
  uint64_t zbuf_base, cbuf_base;
  uint32_t zbuf_pitch, cbuf_pitch;
  uint32_t cbuf_writemask4;            // 4-bit per-channel enable (as programmed)
  uint32_t color_format;               // VX_OM_COLOR_FORMAT_* (0 = A8R8G8B8)
  uint32_t depth_format;               // VX_OM_DEPTH_FORMAT_* (0 = D24S8)
  // resolved by resolve_om_state()
  uint32_t depth_enabled, stencil_enabled[2], blend_enabled;
  uint32_t cbuf_writemask;             // expanded 32-bit byte mask
  uint32_t color_read, color_write;
};

// Derive the enable flags + expanded color mask exactly as the FF unit does.
// Call host-side after filling the raw
// fields, before handing om_state_t to the device.
static inline __attribute__((always_inline)) void resolve_om_state(om_state_t& s) {
  s.depth_enabled = !((s.depth_func == VX_OM_DEPTH_FUNC_ALWAYS) && !(s.depth_writemask & 0x1));
  for (int f = 0; f < 2; ++f)
    s.stencil_enabled[f] = !((s.stencil_func[f]  == VX_OM_DEPTH_FUNC_ALWAYS)
                          && (s.stencil_zpass[f] == VX_OM_STENCIL_OP_KEEP)
                          && (s.stencil_zfail[f] == VX_OM_STENCIL_OP_KEEP));
  s.blend_enabled = !((s.blend_mode_rgb == VX_OM_BLEND_MODE_ADD)
                   && (s.blend_mode_a   == VX_OM_BLEND_MODE_ADD)
                   && (s.blend_src_rgb  == VX_OM_BLEND_FUNC_ONE)
                   && (s.blend_src_a    == VX_OM_BLEND_FUNC_ONE)
                   && (s.blend_dst_rgb  == VX_OM_BLEND_FUNC_ZERO)
                   && (s.blend_dst_a    == VX_OM_BLEND_FUNC_ZERO));
  uint32_t m4 = s.cbuf_writemask4 & 0xf;
  s.cbuf_writemask = (((m4 >> 0) & 1) * 0x000000ff) | (((m4 >> 1) & 1) * 0x0000ff00)
                   | (((m4 >> 2) & 1) * 0x00ff0000) | (((m4 >> 3) & 1) * 0xff000000);
  s.color_read  = (m4 != 0xf);
  s.color_write = (m4 != 0x0);
}

// ── attachment-format encode/decode (SW OM only) ─────────────────────────────
// The FF vx_om4 unit writes only A8R8G8B8 colour + packed D24S8 depth; every
// other colour/depth format is handled here in the software output-merger. The
// OM works internally in ARGB8888; colour is decoded from / encoded to the bound
// format on read/write (sRGB is blended in linear space per the Vulkan spec).

static inline __attribute__((always_inline)) uint32_t Linear8ToSrgb(uint32_t c) {
  static const uint8_t kLinToSrgb[256] = {
      0, 13, 22, 28, 34, 38, 42, 46, 50, 53, 56, 59, 61, 64, 66, 69,
     71, 73, 75, 77, 79, 81, 83, 85, 86, 88, 90, 92, 93, 95, 96, 98,
     99,101,102,104,105,106,108,109,110,112,113,114,115,117,118,119,
    120,121,122,124,125,126,127,128,129,130,131,132,133,134,135,136,
    137,138,139,140,141,142,143,144,145,146,147,148,148,149,150,151,
    152,153,154,155,155,156,157,158,159,159,160,161,162,163,163,164,
    165,166,167,167,168,169,170,170,171,172,173,173,174,175,175,176,
    177,178,178,179,180,180,181,182,182,183,184,185,185,186,187,187,
    188,189,189,190,190,191,192,192,193,194,194,195,196,196,197,197,
    198,199,199,200,200,201,202,202,203,203,204,205,205,206,206,207,
    208,208,209,209,210,210,211,212,212,213,213,214,214,215,215,216,
    216,217,218,218,219,219,220,220,221,221,222,222,223,223,224,224,
    225,226,226,227,227,228,228,229,229,230,230,231,231,232,232,233,
    233,234,234,235,235,236,236,237,237,238,238,238,239,239,240,240,
    241,241,242,242,243,243,244,244,245,245,246,246,246,247,247,248,
    248,249,249,250,250,251,251,251,252,252,253,253,254,254,255,255};
  return kLinToSrgb[c & 0xff];
}

static inline __attribute__((always_inline)) uint32_t om_color_bpp(uint32_t fmt) {
  switch (fmt) {
  case VX_OM_COLOR_FORMAT_R8:  return 1;
  case VX_OM_COLOR_FORMAT_RG8: return 2;
  default:                     return 4;   // A8R8G8B8 / SRGB8A8
  }
}
static inline __attribute__((always_inline)) uint32_t om_depth_bpp(uint32_t fmt) {
  switch (fmt) {
  case VX_OM_DEPTH_FORMAT_S8:  return 1;
  case VX_OM_DEPTH_FORMAT_D16: return 2;
  default:                     return 4;   // D24S8 / D32F
  }
}

// Read/write `bpp` bytes at a byte address (LSU-style, no wider access than the
// attachment's texel so R8/RG8 packed rows are addressed exactly).
static inline __attribute__((always_inline)) uint32_t om_load(uintptr_t a, uint32_t bpp) {
  if (bpp == 4) return *reinterpret_cast<volatile uint32_t*>(a);
  if (bpp == 2) return *reinterpret_cast<volatile uint16_t*>(a);
  return *reinterpret_cast<volatile uint8_t*>(a);
}
static inline __attribute__((always_inline)) void om_store(uintptr_t a, uint32_t bpp, uint32_t v) {
  if (bpp == 4)      *reinterpret_cast<volatile uint32_t*>(a) = v;
  else if (bpp == 2) *reinterpret_cast<volatile uint16_t*>(a) = (uint16_t)v;
  else               *reinterpret_cast<volatile uint8_t*>(a)  = (uint8_t)v;
}

// Decode a stored colour texel → ARGB8888 working colour (sRGB → linear so the
// blend runs in linear space).
static inline __attribute__((always_inline)) uint32_t om_decode_color(uint32_t fmt, uint32_t raw) {
  switch (fmt) {
  case VX_OM_COLOR_FORMAT_SRGB8A8: {
    uint32_t r = gfx_tex::SrgbToLinear8((raw >> 16) & 0xff);
    uint32_t g = gfx_tex::SrgbToLinear8((raw >> 8) & 0xff);
    uint32_t b = gfx_tex::SrgbToLinear8(raw & 0xff);
    return (raw & 0xff000000u) | (r << 16) | (g << 8) | b;   // alpha linear
  }
  case VX_OM_COLOR_FORMAT_R8:  return 0xff000000u | ((raw & 0xff) << 16);
  case VX_OM_COLOR_FORMAT_RG8: return 0xff000000u | ((raw & 0xff) << 16) | (((raw >> 8) & 0xff) << 8);
  default:                     return raw;    // A8R8G8B8
  }
}

// Encode an ARGB8888 working colour → stored texel (linear → sRGB on write).
static inline __attribute__((always_inline)) uint32_t om_encode_color(uint32_t fmt, uint32_t argb) {
  switch (fmt) {
  case VX_OM_COLOR_FORMAT_SRGB8A8: {
    uint32_t r = Linear8ToSrgb((argb >> 16) & 0xff);
    uint32_t g = Linear8ToSrgb((argb >> 8) & 0xff);
    uint32_t b = Linear8ToSrgb(argb & 0xff);
    return (argb & 0xff000000u) | (r << 16) | (g << 8) | b;
  }
  case VX_OM_COLOR_FORMAT_R8:  return (argb >> 16) & 0xff;
  case VX_OM_COLOR_FORMAT_RG8: return ((argb >> 16) & 0xff) | (((argb >> 8) & 0xff) << 8);
  default:                     return argb;
  }
}

// Per-format depth geometry: comparison mask, stencil presence, and the bit
// offset of the stencil field within the packed depth/stencil word.
struct om_depth_desc { uint32_t dmask; uint32_t sshift; bool has_depth; bool has_stencil; };
static inline __attribute__((always_inline)) om_depth_desc om_depth_geom(uint32_t fmt) {
  switch (fmt) {
  case VX_OM_DEPTH_FORMAT_D16:  return { 0xffffu,     0,  true,  false };
  case VX_OM_DEPTH_FORMAT_D32F: return { 0xffffffffu, 0,  true,  false };  // float bits, monotonic for [0,1]
  case VX_OM_DEPTH_FORMAT_S8:   return { 0u,          0,  false, true  };
  default:                      return { OM_DEPTH_MASK, VX_OM_DEPTH_BITS, true, true }; // D24S8
  }
}

// ── Per-fragment software ops (device + host) ────────────────────────────────

// Depth/stencil test. `ds_val` is the packed
// {stencil:depth} word at the pixel in the bound depth format; writes the merged
// word to *ds_result and returns whether the fragment passes the combined
// stencil+depth test. D32F compares the raw float bits (monotonic over [0,1]).
static inline __attribute__((always_inline)) bool ds_test(const om_state_t& s, uint32_t face, uint32_t depth,
                           uint32_t ds_val, uint32_t* ds_result) {
  const int f = face ? 1 : 0;
  om_depth_desc dd = om_depth_geom(s.depth_format);
  uint32_t depth_val   = ds_val & dd.dmask;
  uint32_t stencil_val = dd.has_stencil ? ((ds_val >> dd.sshift) & 0xff) : 0;
  uint32_t depth_ref   = depth & dd.dmask;

  uint32_t sref = s.stencil_ref[f], smask = s.stencil_mask[f];
  uint32_t sref_m = sref & smask, sval_m = stencil_val & smask;

  uint32_t stencil_op;
  bool passed = dd.has_stencil ? DoCompare(s.stencil_func[f], sref_m, sval_m) : true;
  if (passed) {
    passed = dd.has_depth ? DoCompare(s.depth_func, depth_ref, depth_val) : true;
    stencil_op = passed ? s.stencil_zpass[f] : s.stencil_zfail[f];
  } else {
    stencil_op = s.stencil_fail[f];
  }
  uint32_t stencil_result = dd.has_stencil ? DoStencilOp(stencil_op, sref, stencil_val) : 0;
  *ds_result = (stencil_result << dd.sshift) | depth_ref;
  return passed;
}

// Blend.
static inline __attribute__((always_inline)) uint32_t blend(const om_state_t& s, uint32_t src_color, uint32_t dst_color) {
  ColorARGB src(src_color), dst(dst_color), cst(s.blend_const);
  ColorARGB s_rgb = DoBlendFunc(s.blend_src_rgb, src, dst, cst);
  ColorARGB s_a   = DoBlendFunc(s.blend_src_a,   src, dst, cst);
  ColorARGB d_rgb = DoBlendFunc(s.blend_dst_rgb, src, dst, cst);
  ColorARGB d_a   = DoBlendFunc(s.blend_dst_a,   src, dst, cst);
  ColorARGB rgb = DoBlendMode(s.blend_mode_rgb, s.logic_op, src, dst, s_rgb, d_rgb);
  ColorARGB a   = DoBlendMode(s.blend_mode_a,   s.logic_op, src, dst, s_a,   d_a);
  ColorARGB result(a.a, rgb.r, rgb.g, rgb.b);
  return result.value;
}

// One output-merge read-modify-write at explicit depth/color byte addresses
// (host + device): the LSU-based equivalent of one OM lane. Reads dst depth/
// stencil + color, runs the depth/stencil test + blend, applies the write-masks,
// and writes back — the single body shared by the single-sample om_fragment and
// the per-sample MSAA path (om_fragment_msaa), so both are bit-identical to the
// FF OM unit. Returns whether the depth/stencil test passed.
//
// Non-atomic RMW: correct when each (pixel, sample) is touched once. Per-sample
// ordering for overlapping fragments is the determinism open item, handled by
// the SW path's tile serialization.
static inline __attribute__((always_inline)) bool om_sample_rmw(
    const om_state_t& s, uintptr_t z_addr, uintptr_t c_addr,
    uint32_t face, uint32_t src_color, uint32_t src_depth) {
  const int f = face ? 1 : 0;
  om_depth_desc dd = om_depth_geom(s.depth_format);
  uint32_t dbpp = om_depth_bpp(s.depth_format);
  uint32_t cbpp = om_color_bpp(s.color_format);
  bool ds_active = s.depth_enabled || s.stencil_enabled[f];
  bool need_c_read = s.color_write && (s.color_read || s.blend_enabled);

  uint32_t dst_ds    = ds_active   ? om_load(z_addr, dbpp) : 0;
  uint32_t dst_raw   = need_c_read ? om_load(c_addr, cbpp) : 0;
  uint32_t dst_color = om_decode_color(s.color_format, dst_raw);   // decode → blend in linear

  uint32_t merged = 0;
  bool ds_pass = !ds_active || ds_test(s, face, src_depth, dst_ds, &merged);
  uint32_t blended = (s.blend_enabled && ds_pass) ? blend(s, src_color, dst_color) : src_color;

  uint32_t stencil_wm = s.stencil_writemask[f];
  uint32_t ds_wm = ((s.depth_enabled && ds_pass && s.depth_writemask && dd.has_depth) ? dd.dmask : 0u)
                 | ((s.stencil_enabled[f] && dd.has_stencil) ? (stencil_wm << dd.sshift) : 0u);
  if (ds_wm)
    om_store(z_addr, dbpp, (dst_ds & ~ds_wm) | (merged & ds_wm));

  if (s.color_write && ds_pass) {
    uint32_t out_argb = (dst_color & ~s.cbuf_writemask) | (blended & s.cbuf_writemask);
    om_store(c_addr, cbpp, om_encode_color(s.color_format, out_argb));
  }
  return ds_pass;
}

// ── MSAA storage + resolve ───────────────────────────────────────────────────
// Per-sample surfaces are sample-interleaved within a pixel: a pixel's S samples
// are contiguous, so a row is W*S texels and the per-sample column offset is
// (x*S + sample). cbuf_pitch / zbuf_pitch in om_state_t already carry the
// MSAA row stride (= W*S*4). Keeping samples contiguous makes resolve a unit-
// stride gather.
static inline __attribute__((always_inline)) uintptr_t msaa_color_addr(
    const om_state_t& s, uint32_t samples, uint32_t x, uint32_t y, uint32_t sample) {
  return (uintptr_t)(s.cbuf_base + (uint64_t)y * s.cbuf_pitch + (uint64_t)(x * samples + sample) * om_color_bpp(s.color_format));
}
static inline __attribute__((always_inline)) uintptr_t msaa_depth_addr(
    const om_state_t& s, uint32_t samples, uint32_t x, uint32_t y, uint32_t sample) {
  return (uintptr_t)(s.zbuf_base + (uint64_t)y * s.zbuf_pitch + (uint64_t)(x * samples + sample) * om_depth_bpp(s.depth_format));
}

// Box resolve of one pixel's S color samples → a single ARGB8888 value
// (per-channel average, round-to-nearest). Depth is not averaged (use sample 0
// or the multisample z-buffer directly), matching standard MSAA color resolve.
static inline __attribute__((always_inline)) uint32_t msaa_resolve_color(
    const om_state_t& s, uint32_t samples, uint32_t x, uint32_t y) {
  // Read each sample at its native texel width and decode to linear ARGB before
  // averaging, then re-encode — so R8/RG8 packing and sRGB gamma resolve
  // correctly. A8R8G8B8 decode/encode are identity (byte-identical fast path).
  uint32_t bpp = om_color_bpp(s.color_format);
  uint32_t acc[4] = {0, 0, 0, 0};
  for (uint32_t k = 0; k < samples; ++k) {
    uint32_t raw = om_load(msaa_color_addr(s, samples, x, y, k), bpp);
    uint32_t c   = om_decode_color(s.color_format, raw);
    acc[0] +=  c        & 0xff;
    acc[1] += (c >>  8) & 0xff;
    acc[2] += (c >> 16) & 0xff;
    acc[3] += (c >> 24) & 0xff;
  }
  uint32_t half = samples >> 1;
  uint32_t out = 0;
  for (uint32_t ch = 0; ch < 4; ++ch) {
    out |= (((acc[ch] + half) / samples) & 0xff) << (ch * 8);
  }
  return om_encode_color(s.color_format, out);
}

// ── SW texture sampler: on-device fallback for vx_tex4 ───────────────────────
// Reads texels straight from resident texture memory (no FF tcache) using the
// shared gfx_frag_tex.h math, so the result matches the FF unit (and the cocogfx
// oracle) bit-for-bit — it IS the same compute_request/apply_filter code. Works
// on host too (base_addr a plain pointer), so the FF model + a host parity test
// can exercise it as the SW oracle.
static inline __attribute__((always_inline)) uint32_t tex_load_texel(uint64_t addr, uint32_t stride) {
  if (stride == 4) return *(const uint32_t*)(uintptr_t)addr;
  if (stride == 2) return *(const uint16_t*)(uintptr_t)addr;
  return *(const uint8_t*)(uintptr_t)addr;
}

// Extended per-LOD sample (SW-path only): any format above the FF set and/or
// a CLAMP_TO_BORDER wrap. Reuses tex_compute_request for the tap offsets (base 0
// → offset*stride), decodes each tap to ARGB8888 via TexDecodeExtended, applies
// the border colour to out-of-range taps, and filters in the 8-bit ARGB working
// space (point → tap 0, bilinear → TexFilterLinear as A8R8G8B8).
static inline __attribute__((always_inline)) uint32_t tex_sample_ext_lod(
    uint64_t base_addr, uint32_t logdim, uint32_t format, uint32_t filter,
    uint32_t wrap, int32_t u, int32_t v, uint32_t lod,
    uint32_t width, uint32_t height, uint32_t border) {
  gfx_tex::TexelRequest req =
    gfx_tex::tex_compute_request(0, logdim, format, filter, wrap, u, v, lod, width, height);
  uint32_t stride = req.stride;
  // per-LOD dims for the border range check (match tex_compute_request).
  uint32_t log_width  = (uint32_t)gfx_tex::tex_imax((int32_t)(logdim & 0xffff) - (int32_t)lod, 0);
  uint32_t log_height = (uint32_t)gfx_tex::tex_imax((int32_t)(logdim >> 16) - (int32_t)lod, 0);
  uint32_t w = width  ? (uint32_t)gfx_tex::tex_imax((int32_t)(width  >> lod), 1) : (1u << log_width);
  uint32_t h = height ? (uint32_t)gfx_tex::tex_imax((int32_t)(height >> lod), 1) : (1u << log_height);
  uint32_t bmask = gfx_tex::TexBorderMask(u, v, w, h, log_width, log_height,
                                          wrap & 0xffff, wrap >> 16, req.filter);
  if (req.filter == VX_TEX_FILTER_BILINEAR) {
    uint32_t argb[4];
    for (uint32_t i = 0; i < 4; ++i)
      argb[i] = ((bmask >> i) & 1u) ? border
              : gfx_tex::TexDecodeArgb8(format, (const void*)(uintptr_t)(base_addr + req.addr[i]), stride);
    // filter as A8R8G8B8 (taps already linear 8-bit ARGB).
    return gfx_tex::TexFilterLinear(VX_TEX_FORMAT_A8R8G8B8, argb[0], argb[1], argb[2], argb[3],
                                    req.alpha, req.beta);
  }
  if (bmask & 1u) return border;
  return gfx_tex::TexDecodeArgb8(format, (const void*)(uintptr_t)(base_addr + req.addr[0]), stride);
}

// One LOD's sample (POINT or BILINEAR). `base_addr` is the mip's base for `lod`
// (mip_base + mip_off); `logdim` is mip 0's {log_h<<16|log_w}; u/v are
// TEX_FXD_FRAC fixed-point. Caller does trilinear by blending two LODs. The
// FF-format + non-border case keeps the exact bit-identical shared path; the
// extended-format or border case routes to tex_sample_ext_lod.
static inline __attribute__((always_inline)) uint32_t tex_sample_sw_lod(
    uint64_t base_addr, uint32_t logdim, uint32_t format, uint32_t filter,
    uint32_t wrap, int32_t u, int32_t v, uint32_t lod,
    uint32_t width = 0, uint32_t height = 0, uint32_t border = 0) {
  bool has_border = ((wrap & 0xffff) == VX_TEX_WRAP_BORDER) || ((wrap >> 16) == VX_TEX_WRAP_BORDER);
  if (!gfx_tex::TexIsFFFormat(format) || has_border)
    return tex_sample_ext_lod(base_addr, logdim, format, filter, wrap, u, v, lod,
                              width, height, border);
  gfx_tex::TexelRequest req =
    gfx_tex::tex_compute_request(base_addr, logdim, format, filter, wrap, u, v, lod,
                                 width, height);
  uint32_t texels[4] = {0, 0, 0, 0};
  uint32_t taps = (req.filter == VX_TEX_FILTER_BILINEAR) ? 4u : 1u;
  for (uint32_t i = 0; i < taps; ++i)
    texels[i] = tex_load_texel(req.addr[i], req.stride);
  return gfx_tex::tex_apply_filter(req, texels);
}

// Resident texture state for one stage (the SW mirror of the texture DCR state):
// mip base,
// per-LOD mip offsets, dims, format, full filter (incl the mip-linear bit), and
// wrap. The device fragment kernel fills this from the bound texture's resident
// descriptor; the host FF model fills it from the texture DCRs — both then drive
// tex_sample_sw, so the SW path matches vx_tex4 bit-for-bit.
struct TexState {
  uint64_t base;                          // mip 0 base (TEX_ADDR << 6)
  uint32_t mip_off[VX_TEX_LOD_MAX + 1];   // per-LOD byte offset from base
  uint32_t logdim;                        // {log_h << 16 | log_w} of mip 0
  uint32_t format;                        // VX_TEX_FORMAT_*
  uint32_t filter;                        // mag/min (bit 0) | mip-linear (bit 1)
  uint32_t wrap;                          // {wrap_v << 16 | wrap_u}
  uint32_t width;                         // mip-0 integer width  (0 => POT via logdim)
  uint32_t height;                        // mip-0 integer height (0 => POT via logdim)
  uint32_t border;                        // ARGB8888 border colour (WRAP_BORDER)
  uint32_t layer_stride;                  // bytes per array layer / cube face (0 => single 2D)
  uint32_t compare_func;                  // shadow compare op (VX_OM_DEPTH_FUNC_*); 0 => none
  uint32_t swizzle;                       // view component map: r|g<<3|b<<6|a<<9 (0..3=RGBA, 4=0, 5=1)
};

// Full vx_tex4 SW fallback: a complete (u, v, lod) sample including the
// trilinear mip blend. Mirrors the FF texture sampler exactly (same per-LOD tap
// math via tex_sample_sw_lod, same two-mip TexLodLerp), so it is bit-identical
// to the FF unit. `lod` is fixed-point when the mip filter is linear.
static inline __attribute__((always_inline)) uint32_t tex_sample_sw_layer(
    const TexState& s, int32_t u, int32_t v, uint32_t lod, uint32_t layer,
    uint32_t filter) {
  // `filter` (tap in bit0, mip-linear in bit1) overrides s.filter: a fragment
  // shader resolves min-vs-mag per fragment and passes the result, since the one
  // descriptor holds both taps. mag/min selects the per-LOD tap pattern.
  uint32_t tap_filter = filter & TEX_FILTER_MAGMIN_MASK;
  uint64_t lbase = s.base + (uint64_t)layer * s.layer_stride;   // array/cube slice
  if (filter & VX_TEX_FILTER_MIP_LINEAR) {
    uint32_t li   = lod >> VX_TEX_LOD_FRAC_BITS;
    uint32_t lj   = (li + 1 < (uint32_t)VX_TEX_LOD_MAX) ? li + 1 : (uint32_t)VX_TEX_LOD_MAX;
    uint32_t frac = lod & ((1u << VX_TEX_LOD_FRAC_BITS) - 1);
    uint32_t c0 = tex_sample_sw_lod(lbase + s.mip_off[li], s.logdim, s.format,
                                    tap_filter, s.wrap, u, v, li, s.width, s.height, s.border);
    uint32_t c1 = tex_sample_sw_lod(lbase + s.mip_off[lj], s.logdim, s.format,
                                    tap_filter, s.wrap, u, v, lj, s.width, s.height, s.border);
    return gfx_tex::TexLodLerp(c0, c1, frac);
  }
  return tex_sample_sw_lod(lbase + s.mip_off[lod], s.logdim, s.format,
                           tap_filter, s.wrap, u, v, lod, s.width, s.height, s.border);
}

// Sample using the descriptor's own filter (the fixed single-filter path).
static inline __attribute__((always_inline)) uint32_t tex_sample_sw(
    const TexState& s, int32_t u, int32_t v, uint32_t lod) {
  return tex_sample_sw_layer(s, u, v, lod, 0, s.filter);
}

// Sample with a caller-resolved filter (tap in bit0, mip-linear in bit1): a
// fragment shader picks the min or mag tap per fragment from the sign of its LOD,
// since the one descriptor holds both taps.
static inline __attribute__((always_inline)) uint32_t tex_sample_sw(
    const TexState& s, int32_t u, int32_t v, uint32_t lod, uint32_t filter) {
  return tex_sample_sw_layer(s, u, v, lod, 0, filter);
}

// 2D-array view: integer layer index selects the slice at layer*layer_stride.
static inline __attribute__((always_inline)) uint32_t tex_sample_sw_array(
    const TexState& s, int32_t u, int32_t v, uint32_t layer, uint32_t lod) {
  return tex_sample_sw_layer(s, u, v, lod, layer, s.filter);
}

// Cube view: pick the face from the major axis of the (sc,tc,rc) direction and
// project to the face's [0,1] uv, then sample that face's slice (face index is the
// layer). Face order matches Vulkan/GL cube layers: +X,-X,+Y,-Y,+Z,-Z = 0..5.
// Coordinates are floats (the FS supplies the interpolated direction vector).
static inline __attribute__((always_inline)) uint32_t tex_sample_sw_cube(
    const TexState& s, float sc, float tc, float rc, uint32_t lod) {
  float asx = sc < 0 ? -sc : sc, asy = tc < 0 ? -tc : tc, asz = rc < 0 ? -rc : rc;
  uint32_t face; float ma, uc, vc;
  if (asx >= asy && asx >= asz) {
    ma = asx; face = (sc >= 0) ? 0u : 1u; uc = (sc >= 0) ? -rc : rc; vc = -tc;
  } else if (asy >= asz) {
    ma = asy; face = (tc >= 0) ? 2u : 3u; uc = sc; vc = (tc >= 0) ? rc : -rc;
  } else {
    ma = asz; face = (rc >= 0) ? 4u : 5u; uc = (rc >= 0) ? sc : -sc; vc = -tc;
  }
  float inv = (ma != 0.0f) ? (0.5f / ma) : 0.0f;
  float fu = uc * inv + 0.5f, fv = vc * inv + 0.5f;
  const int32_t ONE = 1 << TEX_FXD_FRAC;
  int32_t u = (int32_t)(fu * (float)ONE), v = (int32_t)(fv * (float)ONE);
  return tex_sample_sw_layer(s, u, v, lod, face, s.filter);
}

// texelFetch: the exact texel at integer (x,y) of integer `lod` -- no wrap, no
// filter, no mip blend. Coords are clamped into range (Vulkan leaves an
// out-of-range fetch undefined; clamp is a safe, deterministic choice). The level
// is laid out row-major and contiguous (matching the host mip-chain upload), so
// the texel index is x + y*w.
static inline __attribute__((always_inline)) uint32_t tex_fetch_sw(
    const TexState& s, int32_t x, int32_t y, uint32_t lod) {
  uint32_t log_width  = (uint32_t)gfx_tex::tex_imax((int32_t)(s.logdim & 0xffff) - (int32_t)lod, 0);
  uint32_t log_height = (uint32_t)gfx_tex::tex_imax((int32_t)(s.logdim >> 16) - (int32_t)lod, 0);
  uint32_t w = s.width  ? (uint32_t)gfx_tex::tex_imax((int32_t)(s.width  >> lod), 1) : (1u << log_width);
  uint32_t h = s.height ? (uint32_t)gfx_tex::tex_imax((int32_t)(s.height >> lod), 1) : (1u << log_height);
  uint32_t xi = (x < 0) ? 0u : ((uint32_t)x >= w ? w - 1u : (uint32_t)x);
  uint32_t yi = (y < 0) ? 0u : ((uint32_t)y >= h ? h - 1u : (uint32_t)y);
  uint32_t stride = gfx_tex::FormatStride(s.format);
  uint64_t addr = s.base + s.mip_off[lod] + (uint64_t)(xi + yi * w) * stride;
  return gfx_tex::TexDecodeArgb8(s.format, (const void*)(uintptr_t)addr, stride);
}

// textureGather: channel `comp` of the 2x2 texel footprint at (u,v) of the base
// level, unfiltered, packed in GL gather order {(i0,j1),(i1,j1),(i1,j0),(i0,j0)} as
// bytes x | y<<8 | z<<16 | w<<24. The footprint is the bilinear tap footprint, so
// reuse tex_compute_request (BILINEAR taps + wrap); gather order maps to taps
// {2,3,1,0} (tap order is (u-,v-),(u+,v-),(u-,v+),(u+,v+)). Non-border wraps only.
static inline __attribute__((always_inline)) uint32_t tex_gather_sw(
    const TexState& s, int32_t u, int32_t v, uint32_t comp) {
  // Apply the view's component swizzle: the requested output component maps to a
  // source channel (0..3 = R,G,B,A) or a constant (4 = 0, 5 = 1). Identity
  // (r|g<<3|b<<6|a<<9 = X,Y,Z,W) reproduces the un-swizzled channel select.
  uint32_t map = (s.swizzle >> ((comp & 3) * 3)) & 0x7u;
  if (map == 4u) {
    return 0x00000000u;   // PIPE_SWIZZLE_0 -> all four taps 0
  }
  if (map == 5u) {
    return 0xffffffffu;   // PIPE_SWIZZLE_1 -> all four taps 255
  }
  gfx_tex::TexelRequest req = gfx_tex::tex_compute_request(
      s.base, s.logdim, s.format, VX_TEX_FILTER_BILINEAR, s.wrap, u, v, 0,
      s.width, s.height);
  const uint32_t argb_shift[4] = { 16, 8, 0, 24 };   // R,G,B,A within ARGB8888
  const uint32_t order[4]      = { 2, 3, 1, 0 };      // GL gather order over the taps
  uint32_t sh = argb_shift[map & 3];
  uint32_t packed = 0;
  for (uint32_t i = 0; i < 4; ++i) {
    uint32_t argb = gfx_tex::TexDecodeArgb8(
        s.format, (const void*)(uintptr_t)req.addr[order[i]], req.stride);
    packed |= ((argb >> sh) & 0xffu) << (i * 8);
  }
  return packed;
}

// Read one raw depth texel as a float in [0,1] from resident memory at `p`.
// D32F is stored as a raw float; D16 as a 16-bit unorm (/65535). Any other
// format falls back to the luminance of its ARGB decode (so a colour texture
// bound to a shadow sampler still yields a deterministic value).
static inline __attribute__((always_inline)) float decode_depth_f(
    uint32_t format, const void* p, uint32_t stride) {
  if (format == VX_TEX_FORMAT_D32F) {
    float f; __builtin_memcpy(&f, p, 4); return f;
  }
  if (format == VX_TEX_FORMAT_D16) {
    uint16_t d; __builtin_memcpy(&d, p, 2); return (float)d * (1.0f / 65535.0f);
  }
  uint32_t argb = gfx_tex::TexDecodeArgb8(format, p, stride);
  return (float)((argb >> 16) & 0xff) * (1.0f / 255.0f);   // R as luminance
}

// Float compare for shadow: `ref <compare_func> depth`. Reuses the OM depth
// compare enum (VkCompareOp maps to it host-side), evaluated on the raw depth
// floats so a D32F sample keeps full precision (unlike the 8-bit ARGB decode).
static inline __attribute__((always_inline)) float shadow_compare_f(
    uint32_t compare_func, float ref, float depth) {
  bool pass;
  switch (compare_func) {
  case VX_OM_DEPTH_FUNC_NEVER:    pass = false;           break;
  case VX_OM_DEPTH_FUNC_LESS:     pass = (ref <  depth);  break;
  case VX_OM_DEPTH_FUNC_EQUAL:    pass = (ref == depth);  break;
  case VX_OM_DEPTH_FUNC_LEQUAL:   pass = (ref <= depth);  break;
  case VX_OM_DEPTH_FUNC_GREATER:  pass = (ref >  depth);  break;
  case VX_OM_DEPTH_FUNC_NOTEQUAL: pass = (ref != depth);  break;
  case VX_OM_DEPTH_FUNC_GEQUAL:   pass = (ref >= depth);  break;
  default:                        pass = true;            break;   // ALWAYS
  }
  return pass ? 1.0f : 0.0f;
}

// sampler2DShadow: sample the depth texture at (u,v), compare each tap against
// `ref` with s.compare_func, and return the result as a float in [0,1]. A point
// (mag=POINT) sampler compares one texel (0/1); a bilinear (mag=BILINEAR)
// sampler does PCF — the four 2x2 taps' 0/1 results bilinearly blended by the
// footprint's alpha/beta fractions. Base level only (non-mipmapped). Returns the
// float bit-pattern so the C ABI stays uint32_t like the other samplers.
static inline __attribute__((always_inline)) uint32_t tex_shadow_sw(
    const TexState& s, int32_t u, int32_t v, uint32_t ref_bits, uint32_t filter) {
  float ref; __builtin_memcpy(&ref, &ref_bits, 4);
  bool pcf = (filter & TEX_FILTER_MAGMIN_MASK) == VX_TEX_FILTER_BILINEAR;
  gfx_tex::TexelRequest req = gfx_tex::tex_compute_request(
      s.base, s.logdim, s.format, pcf ? VX_TEX_FILTER_BILINEAR : VX_TEX_FILTER_POINT,
      s.wrap, u, v, 0, s.width, s.height);
  float result;
  if (pcf) {
    // tap order (u-,v-),(u+,v-),(u-,v+),(u+,v+); alpha = u-frac, beta = v-frac
    // (both 8-bit); blend in u then v to match the colour bilinear path.
    float p0 = shadow_compare_f(s.compare_func, ref,
                 decode_depth_f(s.format, (const void*)(uintptr_t)req.addr[0], req.stride));
    float p1 = shadow_compare_f(s.compare_func, ref,
                 decode_depth_f(s.format, (const void*)(uintptr_t)req.addr[1], req.stride));
    float p2 = shadow_compare_f(s.compare_func, ref,
                 decode_depth_f(s.format, (const void*)(uintptr_t)req.addr[2], req.stride));
    float p3 = shadow_compare_f(s.compare_func, ref,
                 decode_depth_f(s.format, (const void*)(uintptr_t)req.addr[3], req.stride));
    float a = (float)req.alpha * (1.0f / 256.0f);
    float b = (float)req.beta  * (1.0f / 256.0f);
    float row0 = p0 * (1.0f - a) + p1 * a;
    float row1 = p2 * (1.0f - a) + p3 * a;
    result = row0 * (1.0f - b) + row1 * b;
  } else {
    result = shadow_compare_f(s.compare_func, ref,
               decode_depth_f(s.format, (const void*)(uintptr_t)req.addr[0], req.stride));
  }
  uint32_t out; __builtin_memcpy(&out, &result, 4);
  return out;
}

// libgfx_sw build contract: om_fragment's full depth+blend+ROP merge (below)
// inflates the fragment kernel past the Vortex divergence pass's default 100-BB
// guard. If the guard trips, the pass silently skips StructurizeCFG + split/join
// and miscompiles the kernel (unselectable uniform markers, unmasked divergent
// control flow). The fix is a *build* flag (-mllvm -vortex-divergence-max-bbs),
// not a source change, so it can't be enforced in the header alone — encapsulate
// it in sw/gfx/libgfx_sw.mk, which raises the guard AND defines
// GFX_SW_DIVERGENCE_OK. Fail loudly here if a DEVICE kernel pulls in om_fragment
// without it, rather than miscompiling silently. (Host builds — the FF model and
// the gfx_msaa parity test — compile the merge normally; there is no divergence
// pass on the host, so no flag is required there.)
#if defined(__VORTEX__) && !defined(GFX_SW_DIVERGENCE_OK)
#error "gfx_sw.h om_fragment needs the divergence-bbs build flag: include sw/gfx/libgfx_sw.mk and add $(LIBGFX_SW_VX_CFLAGS) to the kernel VX_CFLAGS"
#endif
// Software output-merger for one fragment: the LSU-based
// equivalent of vx_om(). Reads dst depth/stencil + color from resident memory,
// runs the test + blend, applies the write-masks, and writes back — mirroring
// the FF OM compute+write sequence exactly. Caller supplies the same
// (x, y, face, src_color, src_depth) it would pass to vx_om.
//
// NOTE: like a single OM lane this is a non-atomic read-modify-write; it is
// correct when each pixel is touched once (no concurrent same-pixel fragments).
// Per-pixel ordering/atomicity for overlapping fragments is the determinism
// open item, handled with the SW path's tile
// serialization / MSAA work.
//
// TOOLCHAIN NOTE: inline this whole merge into the fragment kernel
// (always_inline) and build the kernel with -mllvm -vortex-divergence-max-bbs=N
// (N large enough to cover the kernel). The full depth+blend+ROP merge inflates
// the kernel past the Vortex divergence pass's default 100-BB guard; if the
// guard trips, the pass skips StructurizeCFG + split/join, leaving the kernel's
// uniform OM-state reads as unselectable llvm.riscv.vx.uniform markers and its
// divergent control flow unmasked. With the guard raised the kernel is processed
// normally and the merge is bit-exact vs the FF OM unit (validated, all configs).
static inline __attribute__((always_inline)) void om_fragment(const om_state_t& s, uint32_t x, uint32_t y,
                               uint32_t face, uint32_t src_color, uint32_t src_depth) {
  uintptr_t z_addr = (uintptr_t)(s.zbuf_base + (uint64_t)y * s.zbuf_pitch + (uint64_t)x * om_depth_bpp(s.depth_format));
  uintptr_t c_addr = (uintptr_t)(s.cbuf_base + (uint64_t)y * s.cbuf_pitch + (uint64_t)x * om_color_bpp(s.color_format));
  om_sample_rmw(s, z_addr, c_addr, face, src_color, src_depth);
}

// Per-sample MSAA output-merge for one fragment: run the OM merge at each
// covered sample's storage slot. `samples` is the sample count (1/2/4),
// `sample_mask` is the per-pixel coverage from rast_sample_mask (bit k = sample
// k covered). Each covered sample gets its own depth-test + blend + ROP against
// its own per-sample depth/color slot — equivalent to S independent OM lanes for
// this pixel. `src_color`/`src_depth` are the shaded fragment values (shared
// across the pixel's samples; per-sample attribute interpolation can refine
// src_depth per sample later — centroid shading keeps a single color).
static inline __attribute__((always_inline)) void om_fragment_msaa(
    const om_state_t& s, uint32_t samples, uint32_t x, uint32_t y, uint32_t face,
    uint32_t sample_mask, uint32_t src_color, uint32_t src_depth) {
  for (uint32_t k = 0; k < samples; ++k) {
    if (!((sample_mask >> k) & 1u)) continue;
    om_sample_rmw(s, msaa_depth_addr(s, samples, x, y, k),
                  msaa_color_addr(s, samples, x, y, k), face, src_color, src_depth);
  }
}

// ── Multiple render targets ──────────────────────────────────────────────────
// The depth/stencil test is shared across all colour attachments (single depth
// buffer), so MRT keeps om_state_t as the depth/stencil + shared framebuffer
// geometry and adds a per-attachment colour descriptor. Vulkan requires
// maxColorAttachments >= 4.
#ifndef VX_OM_MAX_RT
#define VX_OM_MAX_RT 4
#endif

// One colour attachment's blend/write configuration (mirror of the per-RT slice
// of the OM blend DCRs). Depth/stencil is not here — it is shared in om_state_t.
struct om_color_t {
  uint64_t cbuf_base;                  // attachment byte base
  uint32_t cbuf_pitch;                 // attachment row stride (bytes)
  uint32_t blend_mode_rgb, blend_mode_a;
  uint32_t blend_src_rgb, blend_src_a, blend_dst_rgb, blend_dst_a;
  uint32_t blend_const, logic_op;
  uint32_t cbuf_writemask4;            // 4-bit per-channel enable (as programmed)
  uint32_t color_format;               // VX_OM_COLOR_FORMAT_* (0 = A8R8G8B8)
  // resolved by resolve_om_color()
  uint32_t blend_enabled;
  uint32_t cbuf_writemask;             // expanded 32-bit byte mask
  uint32_t color_read, color_write;
};

// Derive a colour attachment's enable flags + expanded mask exactly as
// resolve_om_state() does for the single-RT fields.
static inline __attribute__((always_inline)) void resolve_om_color(om_color_t& c) {
  c.blend_enabled = !((c.blend_mode_rgb == VX_OM_BLEND_MODE_ADD)
                   && (c.blend_mode_a   == VX_OM_BLEND_MODE_ADD)
                   && (c.blend_src_rgb  == VX_OM_BLEND_FUNC_ONE)
                   && (c.blend_src_a    == VX_OM_BLEND_FUNC_ONE)
                   && (c.blend_dst_rgb  == VX_OM_BLEND_FUNC_ZERO)
                   && (c.blend_dst_a    == VX_OM_BLEND_FUNC_ZERO));
  uint32_t m4 = c.cbuf_writemask4 & 0xf;
  c.cbuf_writemask = (((m4 >> 0) & 1) * 0x000000ff) | (((m4 >> 1) & 1) * 0x0000ff00)
                   | (((m4 >> 2) & 1) * 0x00ff0000) | (((m4 >> 3) & 1) * 0xff000000);
  c.color_read  = (m4 != 0xf);
  c.color_write = (m4 != 0x0);
}

// Blend for an explicit per-attachment colour descriptor (the MRT twin of
// blend(); both go through the same free DoBlendFunc/DoBlendMode ops so an
// attachment's result matches the single-RT path bit-for-bit).
static inline __attribute__((always_inline)) uint32_t blend_rt(const om_color_t& c, uint32_t src_color, uint32_t dst_color) {
  ColorARGB src(src_color), dst(dst_color), cst(c.blend_const);
  ColorARGB s_rgb = DoBlendFunc(c.blend_src_rgb, src, dst, cst);
  ColorARGB s_a   = DoBlendFunc(c.blend_src_a,   src, dst, cst);
  ColorARGB d_rgb = DoBlendFunc(c.blend_dst_rgb, src, dst, cst);
  ColorARGB d_a   = DoBlendFunc(c.blend_dst_a,   src, dst, cst);
  ColorARGB rgb = DoBlendMode(c.blend_mode_rgb, c.logic_op, src, dst, s_rgb, d_rgb);
  ColorARGB a   = DoBlendMode(c.blend_mode_a,   c.logic_op, src, dst, s_a,   d_a);
  return ColorARGB(a.a, rgb.r, rgb.g, rgb.b).value;
}

// Software output-merger for one fragment writing N colour attachments: the
// depth/stencil test + write-back runs ONCE against the shared z-buffer, then
// each attachment k does its own blend + colour write of src_colors[k] at its
// own base/pitch (matching a real MRT OM: one depth op, N colour ops). Bit-
// identical to N single-RT om_fragment calls that share one depth result.
static inline __attribute__((always_inline)) void om_fragment_mrt(
    const om_state_t& s, const om_color_t* colors, uint32_t num_color,
    uint32_t x, uint32_t y, uint32_t face,
    const uint32_t* src_colors, uint32_t src_depth) {
  const int f = face ? 1 : 0;
  // clamp the attachment count to the hardware bound so a stale/short
  // colors[] or an over-large num_color can never read past colors[]/src_colors[].
  if (num_color > VX_OM_MAX_RT) num_color = VX_OM_MAX_RT;
  bool ds_active = s.depth_enabled || s.stencil_enabled[f];
  om_depth_desc dd = om_depth_geom(s.depth_format);
  uint32_t dbpp = om_depth_bpp(s.depth_format);
  uintptr_t z_addr = (uintptr_t)(s.zbuf_base + (uint64_t)y * s.zbuf_pitch + (uint64_t)x * dbpp);

  uint32_t dst_ds = ds_active ? om_load(z_addr, dbpp) : 0;
  uint32_t merged = 0;
  bool ds_pass = !ds_active || ds_test(s, face, src_depth, dst_ds, &merged);

  uint32_t stencil_wm = s.stencil_writemask[f];
  uint32_t ds_wm = ((s.depth_enabled && ds_pass && s.depth_writemask && dd.has_depth) ? dd.dmask : 0u)
                 | ((s.stencil_enabled[f] && dd.has_stencil) ? (stencil_wm << dd.sshift) : 0u);
  if (ds_wm)
    om_store(z_addr, dbpp, (dst_ds & ~ds_wm) | (merged & ds_wm));

  if (!ds_pass)
    return;

  for (uint32_t k = 0; k < num_color; ++k) {
    const om_color_t& c = colors[k];
    if (!c.color_write) continue;
    uint32_t cbpp = om_color_bpp(c.color_format);
    uintptr_t c_addr = (uintptr_t)(c.cbuf_base + (uint64_t)y * c.cbuf_pitch + (uint64_t)x * cbpp);
    bool need_c_read = c.color_read || c.blend_enabled;
    uint32_t dst_raw = need_c_read ? om_load(c_addr, cbpp) : 0;
    uint32_t dst_color = om_decode_color(c.color_format, dst_raw);
    uint32_t blended = c.blend_enabled ? blend_rt(c, src_colors[k], dst_color) : src_colors[k];
    uint32_t out_argb = (dst_color & ~c.cbuf_writemask) | (blended & c.cbuf_writemask);
    om_store(c_addr, cbpp, om_encode_color(c.color_format, out_argb));
  }
}

} // namespace gfx_sw
