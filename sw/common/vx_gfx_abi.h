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

// On-wire graphics ABI: the host↔device serialized layout shared by
// every layer of the graphics stack. Edits here change the binding
// contract — they must land in lock-step on:
//   - sw/runtime (host-side serializers — sw/runtime/graphics.cpp Binning)
//   - sw/kernel  (device-side consumers — gfx_draw3d/gfx_raster kernels)
//   - sim/simx   (host hardware mirror  — sw/common/gfx_ff_model.cpp)
//   - hw/rtl     (RTL packed types)
//
// Single source of truth: both sw/kernel/include/vx_graphics.h (public
// SDK kernel header) and sw/common/gfx_ff_model.h (simx-internal mirror)
// include this file. Lives in sw/common/ because the isolation rule
// forbids simx from reaching into sw/kernel/include/; installed
// alongside the public kernel headers because vx_graphics.h depends on it.

#pragma once

#include <stdint.h>
#include <type_traits>

namespace vortex {
namespace graphics {

///////////////////////////////////////////////////////////////////////////////
// On-wire fixed-point type. F is the number of fractional bits.
// Trivially-copyable: the ABI is just the int32_t bits. Includes full
// arithmetic so host/device code can compute directly in fixed-point
// (matches cocogfx::TFixed<F> semantics, but self-contained — no external
// header dependency).
///////////////////////////////////////////////////////////////////////////////

template <int F>
struct fixed_t {
  static constexpr int     FRAC = F;
  static constexpr int32_t ONE  = int32_t(1) << F;
  static constexpr int32_t HALF = int32_t(1) << (F - 1);
  static constexpr int32_t MASK = ONE - 1;

  int32_t bits;

  constexpr fixed_t() : bits(0) {}
  constexpr explicit fixed_t(int v) : bits(int32_t(v) << F) {}
  constexpr explicit fixed_t(float f)
    : bits(static_cast<int32_t>(f * static_cast<float>(ONE))) {}

  // Q-format conversion from a different fractional-bit-count fixed_t.
  // Right-shift loses precision (intentional, matches cocogfx::TFixed's
  // converting-ctor behaviour); left-shift extends precision losslessly.
  template <int F2>
  constexpr explicit fixed_t(fixed_t<F2> other)
    : bits(F >= F2 ? (other.bits << (F - F2))
                   : (other.bits >> (F2 - F))) {}

  static constexpr fixed_t make(int32_t raw) {
    fixed_t x; x.bits = raw; return x;
  }
  constexpr int32_t data() const { return bits; }

  // Explicit float conversion (Q?.F → float). Used by kernel-side
  // reciprocal math where the bcoord CSR readback feeds a 1/(F0+F1+F2)
  // reciprocal.
  constexpr explicit operator float() const {
    return static_cast<float>(bits) / static_cast<float>(ONE);
  }

  // Explicit integral conversion (Q?.F → integer, integer-part only).
  // Drops the F fractional bits. Used by kernel-side OM writes that
  // pack interpolated color/depth into uint8_t/uint32_t output words.
  template <typename T,
            typename = std::enable_if_t<std::is_integral_v<T>>>
  constexpr explicit operator T() const {
    return static_cast<T>(bits >> F);
  }

  // Arithmetic
  constexpr fixed_t operator+(fixed_t r) const { return make(bits + r.bits); }
  constexpr fixed_t operator-(fixed_t r) const { return make(bits - r.bits); }
  constexpr fixed_t operator-()           const { return make(-bits); }
  constexpr fixed_t operator*(int r)      const { return make(bits * r); }
  constexpr fixed_t operator*(fixed_t r)  const {
    return make(static_cast<int32_t>(
        (static_cast<int64_t>(bits) * r.bits) >> F));
  }
  constexpr fixed_t operator/(fixed_t r) const {
    return make(static_cast<int32_t>(
        (static_cast<int64_t>(bits) << F) / r.bits));
  }

  // Shifts
  constexpr fixed_t operator<<(int n) const { return make(bits << n); }
  constexpr fixed_t operator>>(int n) const { return make(bits >> n); }

  // Compound
  fixed_t& operator+=(fixed_t r) { bits += r.bits; return *this; }
  fixed_t& operator-=(fixed_t r) { bits -= r.bits; return *this; }

  // Comparison
  constexpr bool operator==(fixed_t r) const { return bits == r.bits; }
  constexpr bool operator!=(fixed_t r) const { return bits != r.bits; }
  constexpr bool operator< (fixed_t r) const { return bits <  r.bits; }
  constexpr bool operator<=(fixed_t r) const { return bits <= r.bits; }
  constexpr bool operator> (fixed_t r) const { return bits >  r.bits; }
  constexpr bool operator>=(fixed_t r) const { return bits >= r.bits; }
};

template <int F>
constexpr fixed_t<F> operator*(int l, fixed_t<F> r) { return r * l; }

// RASTER uses Q15.16 for edge equations and Q7.24 for attribute deltas.
using fixed16_t = fixed_t<16>;
using fixed24_t = fixed_t<24>;

using FloatE = fixed16_t;
using FloatA = fixed24_t;

struct vec2e_t { FloatE x, y; };
struct vec3e_t { FloatE x, y, z; };

///////////////////////////////////////////////////////////////////////////////
// On-wire RASTER buffer layout (host writes, hardware reads)
///////////////////////////////////////////////////////////////////////////////

struct rast_bbox_t {
  uint32_t left, right, top, bottom;
};

struct rast_tile_header_t {
  uint16_t tile_x, tile_y, pids_offset, pids_count;
};

// gfx_v2 coarse-bin header. On-device binning groups prims into 128 px
// bins (VX_CFG_RASTER_BIN_LOGSIZE) and the RASTER front end descends
// bin -> block -> quad. One header per bin, in bin_id order (bin_x/bin_y
// decoded so the front end needs no divide). pids_offset is an absolute index
// into the sorted_pids array that follows the dense header block
// (pid_addr = tbuf + num_bins*sizeof(rast_bin_header_t) + pids_offset*4); the
// 32-bit fields lift the 16-bit tile-header limits the coarse bins would hit.
struct rast_bin_header_t {
  uint16_t bin_x, bin_y;     // bin coords (x BIN_SIZE = pixel origin fed to te)
  uint32_t pids_offset;      // start index into the sorted pid array
  uint32_t pids_count;       // prims overlapping this bin
};

struct rast_attrib_t {
  FloatA x, y, z;
};

// Attribute planes carried per primitive (Q7.24 barycentric deltas {a0-a2,
// a1-a2, a2}). Depth `z` is a screen-space affine plane (correct as-is). The
// colour/texcoord planes r,g,b,a,u,v carry the *perspective-premultiplied*
// attribute a·(1/w); `rhw` carries the (max-normalized) 1/w plane. The FS
// interpolates all planes affinely in screen space, then divides the colour/uv
// planes by the interpolated 1/w to recover the perspective-correct attribute.
// The 1/w values are normalized by their per-triangle max in setup so the
// stored fixed-point stays in range (the scale cancels in the FS divide); when
// w is constant this reduces exactly to plain affine interpolation. Setup also
// folds an extra common power-of-2 downscale into 1/w when a premultiplied
// texcoord would exceed FloatA's Q7.24 range (large tiling/wrap UV), which
// likewise cancels in the FS divide — so tiled UV well beyond 1.0 stays exact.
struct rast_attribs_t {
  rast_attrib_t z, r, g, b, a, u, v, rhw;
};

struct rast_prim_t {
  vec3e_t        edges[3];
  rast_attribs_t attribs;
};

///////////////////////////////////////////////////////////////////////////////
// Fragment-wave payload (RASTER dispatch v2).
//
// One per active lane of a launched fragment wave. At fragment-wave launch the
// raster work distributor stages NUM_THREADS of these into the warp's gfx
// register window (lane t at slot VX_GFX_FRAG_SLOT_BASE..); the FS reads its own
// lane's record via vx_frag_load()/GETWS — no LMEM traffic, no polling. The
// record is just {pos_mask, pid}; the FS recomputes per-corner edge values from
// the primitive edges + the quad origin (pos_mask decodes to pos_y<<18 |
// pos_x<<4 | cov_mask).
///////////////////////////////////////////////////////////////////////////////
struct frag_payload_t {
  uint32_t pos_mask;            // cov_mask[3:0] | (pos_x<<4) | (pos_y<<18)
  uint32_t pid;                 // primitive id
};

///////////////////////////////////////////////////////////////////////////////
// 8-bit-per-channel pixel helpers (kernel FS color packing + host blending)
///////////////////////////////////////////////////////////////////////////////

static inline void Unpack8888(uint32_t texel, uint32_t* lo, uint32_t* hi) {
  *lo = texel & 0x00ff00ff;
  *hi = (texel >> 8) & 0x00ff00ff;
}

static inline uint32_t Pack8888(uint32_t lo, uint32_t hi) {
  return (hi << 8) | lo;
}

static inline uint32_t Lerp8888(uint32_t a, uint32_t b, uint32_t f) {
  uint32_t p = a * (0xff - f) + b * f + 0x00800080;
  uint32_t q = (p >> 8) & 0x00ff00ff;
  return ((p + q) >> 8) & 0x00ff00ff;
}

} // namespace graphics
} // namespace vortex
