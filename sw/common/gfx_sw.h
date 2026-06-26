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

// gfx_v2 on-device SIMT software fallback (charter §6.5 / gfx_v2_software_fallback.md).
//
// The always-correct path that runs on the SIMT cores when a fixed-function
// unit cannot represent a required feature — full residency forbids a host
// (llvmpipe) fallback, so the completeness path lives on the device. This
// header is the device-compilable `libgfx_sw`; it is ALSO included by the host
// FF models (sw/common/gfx_render.cpp) so the per-fragment math has a single
// source of truth (gfx_v2_software_fallback.md §7) — the SW path matches the FF
// path bit-for-bit because it IS the same code.
//
// First component: the software output-merger (§4.3) — depth/stencil test,
// blend, logic-op, write-mask — replacing the vx_om fixed-function call site.
// Freestanding (no <algorithm>/<cmath>) so it compiles for the baremetal device.

#ifndef _GFX_SW_H_
#define _GFX_SW_H_

#include <stdint.h>
#include <VX_types.h>
#include "tex_sample.h"   // shared tex-sampling math (FF model + SW path, §4.2)
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
  case VX_OM_BLEND_FUNC_ONE_MINUS_CONST_A:  return ColorARGB(0xff - cst.a, 0xff - cst.r, 0xff - cst.g, 0xff - cst.b);
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
//    same way DepthTencil/Blender/om_core configure() do) ─────────────────────
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
  // resolved by resolve_om_state()
  uint32_t depth_enabled, stencil_enabled[2], blend_enabled;
  uint32_t cbuf_writemask;             // expanded 32-bit byte mask
  uint32_t color_read, color_write;
};

// Derive the enable flags + expanded color mask exactly as the FF unit does
// (DepthTencil/Blender/om_core configure). Call host-side after filling the raw
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

// ── Per-fragment software ops (device + host) ────────────────────────────────

// Depth/stencil test (port of DepthTencil::test). `ds_val` is the packed
// {stencil:depth} word at the pixel; writes the merged word to *ds_result and
// returns whether the fragment passes the combined stencil+depth test.
static inline __attribute__((always_inline)) bool ds_test(const om_state_t& s, uint32_t face, uint32_t depth,
                           uint32_t ds_val, uint32_t* ds_result) {
  const int f = face ? 1 : 0;
  uint32_t depth_val   = ds_val & VX_OM_DEPTH_MASK;
  uint32_t stencil_val = ds_val >> VX_OM_DEPTH_BITS;
  uint32_t depth_ref   = depth & VX_OM_DEPTH_MASK;

  uint32_t sref = s.stencil_ref[f], smask = s.stencil_mask[f];
  uint32_t sref_m = sref & smask, sval_m = stencil_val & smask;

  uint32_t stencil_op;
  bool passed = DoCompare(s.stencil_func[f], sref_m, sval_m);
  if (passed) {
    passed = DoCompare(s.depth_func, depth_ref, depth_val);
    stencil_op = passed ? s.stencil_zpass[f] : s.stencil_zfail[f];
  } else {
    stencil_op = s.stencil_fail[f];
  }
  uint32_t stencil_result = DoStencilOp(stencil_op, sref, stencil_val);
  *ds_result = (stencil_result << VX_OM_DEPTH_BITS) | depth_ref;
  return passed;
}

// Blend (port of Blender::blend).
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
// ordering for overlapping fragments is the determinism open item
// (gfx_v2_software_fallback.md §11), handled by the SW path's tile serialization.
static inline __attribute__((always_inline)) bool om_sample_rmw(
    const om_state_t& s, uintptr_t z_addr, uintptr_t c_addr,
    uint32_t face, uint32_t src_color, uint32_t src_depth) {
  const int f = face ? 1 : 0;
  bool ds_active = s.depth_enabled || s.stencil_enabled[f];
  bool need_c_read = s.color_write && (s.color_read || s.blend_enabled);

  uint32_t dst_ds    = ds_active   ? *reinterpret_cast<volatile uint32_t*>(z_addr) : 0;
  uint32_t dst_color = need_c_read ? *reinterpret_cast<volatile uint32_t*>(c_addr) : 0;

  uint32_t merged = 0;
  bool ds_pass = !ds_active || ds_test(s, face, src_depth, dst_ds, &merged);
  uint32_t blended = (s.blend_enabled && ds_pass) ? blend(s, src_color, dst_color) : src_color;

  uint32_t stencil_wm = s.stencil_writemask[f];
  uint32_t ds_wm = ((s.depth_enabled && ds_pass && s.depth_writemask) ? VX_OM_DEPTH_MASK : 0u)
                 | (s.stencil_enabled[f] ? (stencil_wm << VX_OM_DEPTH_BITS) : 0u);
  if (ds_wm)
    *reinterpret_cast<volatile uint32_t*>(z_addr) = (dst_ds & ~ds_wm) | (merged & ds_wm);

  if (s.color_write && ds_pass)
    *reinterpret_cast<volatile uint32_t*>(c_addr) = (dst_color & ~s.cbuf_writemask) | (blended & s.cbuf_writemask);
  return ds_pass;
}

// ── MSAA storage + resolve (§6) ──────────────────────────────────────────────
// Per-sample surfaces are sample-interleaved within a pixel: a pixel's S samples
// are contiguous, so a row is W*S texels and the per-sample column offset is
// (x*S + sample). cbuf_pitch / zbuf_pitch in om_state_t already carry the
// MSAA row stride (= W*S*4). Keeping samples contiguous makes resolve a unit-
// stride gather.
static inline __attribute__((always_inline)) uintptr_t msaa_color_addr(
    const om_state_t& s, uint32_t samples, uint32_t x, uint32_t y, uint32_t sample) {
  return (uintptr_t)(s.cbuf_base + (uint64_t)y * s.cbuf_pitch + (uint64_t)(x * samples + sample) * 4);
}
static inline __attribute__((always_inline)) uintptr_t msaa_depth_addr(
    const om_state_t& s, uint32_t samples, uint32_t x, uint32_t y, uint32_t sample) {
  return (uintptr_t)(s.zbuf_base + (uint64_t)y * s.zbuf_pitch + (uint64_t)(x * samples + sample) * 4);
}

// Box resolve of one pixel's S color samples → a single ARGB8888 value
// (per-channel average, round-to-nearest). Depth is not averaged (use sample 0
// or the multisample z-buffer directly), matching standard MSAA color resolve.
static inline __attribute__((always_inline)) uint32_t msaa_resolve_color(
    const om_state_t& s, uint32_t samples, uint32_t x, uint32_t y) {
  uint32_t acc[4] = {0, 0, 0, 0};
  for (uint32_t k = 0; k < samples; ++k) {
    uint32_t c = *reinterpret_cast<volatile uint32_t*>(msaa_color_addr(s, samples, x, y, k));
    acc[0] +=  c        & 0xff;
    acc[1] += (c >>  8) & 0xff;
    acc[2] += (c >> 16) & 0xff;
    acc[3] += (c >> 24) & 0xff;
  }
  uint32_t half = samples >> 1;
  uint32_t out = 0;
  for (uint32_t ch = 0; ch < 4; ++ch)
    out |= (((acc[ch] + half) / samples) & 0xff) << (ch * 8);
  return out;
}

// ── SW texture sampler (§4.2): on-device fallback for vx_tex4 ─────────────────
// Reads texels straight from resident texture memory (no FF tcache) using the
// shared tex_sample.h math, so the result matches the FF unit (and the cocogfx
// oracle) bit-for-bit — it IS the same compute_request/apply_filter code. Works
// on host too (base_addr a plain pointer), so the FF model + a host parity test
// can exercise it as the SW oracle.
static inline __attribute__((always_inline)) uint32_t tex_load_texel(uint64_t addr, uint32_t stride) {
  if (stride == 4) return *(const uint32_t*)(uintptr_t)addr;
  if (stride == 2) return *(const uint16_t*)(uintptr_t)addr;
  return *(const uint8_t*)(uintptr_t)addr;
}

// One LOD's sample (POINT or BILINEAR). `base_addr` is the mip's base for `lod`
// (mip_base + mip_off); `logdim` is mip 0's {log_h<<16|log_w}; u/v are
// VX_TEX_FXD_FRAC fixed-point. Caller does trilinear by blending two LODs.
static inline __attribute__((always_inline)) uint32_t tex_sample_sw_lod(
    uint64_t base_addr, uint32_t logdim, uint32_t format, uint32_t filter,
    uint32_t wrap, int32_t u, int32_t v, uint32_t lod) {
  gfx_tex::TexelRequest req =
    gfx_tex::tex_compute_request(base_addr, logdim, format, filter, wrap, u, v, lod);
  uint32_t texels[4] = {0, 0, 0, 0};
  uint32_t taps = (req.filter == VX_TEX_FILTER_BILINEAR) ? 4u : 1u;
  for (uint32_t i = 0; i < taps; ++i)
    texels[i] = tex_load_texel(req.addr[i], req.stride);
  return gfx_tex::tex_apply_filter(req, texels);
}

// Resident texture state for one stage (the SW mirror of TexDCRS): mip base,
// per-LOD mip offsets, dims, format, full filter (incl the mip-linear bit), and
// wrap. The device fragment kernel fills this from the bound texture's resident
// descriptor; the host FF model fills it from TexDCRS — both then drive
// tex_sample_sw, so the SW path matches vx_tex4 bit-for-bit.
struct TexState {
  uint64_t base;                          // mip 0 base (TEX_ADDR << 6)
  uint32_t mip_off[VX_TEX_LOD_MAX + 1];   // per-LOD byte offset from base
  uint32_t logdim;                        // {log_h << 16 | log_w} of mip 0
  uint32_t format;                        // VX_TEX_FORMAT_*
  uint32_t filter;                        // mag/min (bit 0) | mip-linear (bit 1)
  uint32_t wrap;                          // {wrap_v << 16 | wrap_u}
};

// Full vx_tex4 SW fallback: a complete (u, v, lod) sample including the §6.8
// trilinear mip blend. Mirrors TextureSampler::read() exactly (same per-LOD tap
// math via tex_sample_sw_lod, same two-mip TexLodLerp), so it is bit-identical
// to the FF unit. `lod` is fixed-point when the mip filter is linear.
static inline __attribute__((always_inline)) uint32_t tex_sample_sw(
    const TexState& s, int32_t u, int32_t v, uint32_t lod) {
  // mag/min selects the per-LOD tap pattern; the mip-linear bit is consumed here.
  uint32_t tap_filter = s.filter & VX_TEX_FILTER_BITS;
  if (s.filter & VX_TEX_FILTER_MIP_LINEAR) {
    uint32_t li   = lod >> VX_TEX_LOD_FRAC_BITS;
    uint32_t lj   = (li + 1 < (uint32_t)VX_TEX_LOD_MAX) ? li + 1 : (uint32_t)VX_TEX_LOD_MAX;
    uint32_t frac = lod & ((1u << VX_TEX_LOD_FRAC_BITS) - 1);
    uint32_t c0 = tex_sample_sw_lod(s.base + s.mip_off[li], s.logdim, s.format,
                                    tap_filter, s.wrap, u, v, li);
    uint32_t c1 = tex_sample_sw_lod(s.base + s.mip_off[lj], s.logdim, s.format,
                                    tap_filter, s.wrap, u, v, lj);
    return gfx_tex::TexLodLerp(c0, c1, frac);
  }
  return tex_sample_sw_lod(s.base + s.mip_off[lod], s.logdim, s.format,
                           tap_filter, s.wrap, u, v, lod);
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
// the om_core COMPUTE+WRITE sequence exactly. Caller supplies the same
// (x, y, face, src_color, src_depth) it would pass to vx_om.
//
// NOTE: like a single OM lane this is a non-atomic read-modify-write; it is
// correct when each pixel is touched once (no concurrent same-pixel fragments).
// Per-pixel ordering/atomicity for overlapping fragments is the determinism
// open item (gfx_v2_software_fallback.md §11), handled with the SW path's tile
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
  uintptr_t z_addr = (uintptr_t)(s.zbuf_base + (uint64_t)y * s.zbuf_pitch + (uint64_t)x * 4);
  uintptr_t c_addr = (uintptr_t)(s.cbuf_base + (uint64_t)y * s.cbuf_pitch + (uint64_t)x * 4);
  om_sample_rmw(s, z_addr, c_addr, face, src_color, src_depth);
}

// Per-sample MSAA output-merge for one fragment (§6): run the OM merge at each
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

} // namespace gfx_sw

#endif // _GFX_SW_H_
