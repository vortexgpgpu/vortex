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

// Canonical graphics ABI header. Two sections:
//   - on-wire types + 8888 pixel helpers (host-visible, always available)
//   - device-side intrinsics for TEX / OM / RASTER (kernel-only, gated
//     on __VORTEX__)
//
// Host code (mesa, runtime, simx) and device code both include this
// header for the shared types. graphics.h in sw/runtime/include/
// builds on top of this and adds the host-only API (Binning, vertex_t,
// DCR address helpers).

#pragma once

#include <stdint.h>
#include <VX_types.h>

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
    : bits(static_cast<int32_t>(f * static_cast<float>(ONE)
                                + (f < 0 ? -0.5f : 0.5f))) {}

  static constexpr fixed_t make(int32_t raw) {
    fixed_t x; x.bits = raw; return x;
  }
  constexpr int32_t data() const { return bits; }

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

struct rast_attrib_t {
  FloatA x, y, z;
};

struct rast_attribs_t {
  rast_attrib_t z, r, g, b, a, u, v;
};

struct rast_prim_t {
  vec3e_t        edges[3];
  rast_attribs_t attribs;
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

///////////////////////////////////////////////////////////////////////////////
// Kernel-only intrinsics for the fixed-function TEX / OM / RASTER units.
// Encodings (CUSTOM1 family):
//   funct3=1, R4-type, funct2=stage : vx_tex          (texture sample)
//   funct3=2, R4-type, funct2=0     : vx_om           (output-merger write)
//   funct3=3, R-type,  funct7=0     : vx_rast         (raster pop)
//   funct3=4, R-type,  funct7=0     : vx_rast_begin   (per-frame trigger)
// Trap as illegal-instruction unless VX_CFG_EXT_TEX_ENABLE /
// VX_CFG_EXT_OM_ENABLE / VX_CFG_EXT_RASTER_ENABLE is set.
///////////////////////////////////////////////////////////////////////////////

#ifdef __VORTEX__

#include <vx_intrinsics.h>

namespace vortex {
namespace graphics {

// Texture sample: (stage, u, v, lod) -> texel
inline unsigned vx_tex(unsigned stage, unsigned u, unsigned v, unsigned lod) {
  unsigned ret;
  __asm__ volatile (".insn r4 %1, 1, %2, %0, %3, %4, %5"
      : "=r"(ret)
      : "i"(RISCV_CUSTOM1), "i"(stage), "r"(u), "r"(v), "r"(lod));
  return ret;
}

// Output-merger write: (x, y, face, color, depth)
inline void vx_om(unsigned x, unsigned y, unsigned face, unsigned color, unsigned depth) {
  unsigned pos_face = (y << 16) | (x << 1) | face;
  __asm__ volatile (".insn r4 %0, 2, 0, x0, %1, %2, %3"
      :: "i"(RISCV_CUSTOM1), "r"(pos_face), "r"(color), "r"(depth));
}

// Raster pop: returns next quad descriptor from the rasterizer.
inline unsigned vx_rast() {
  unsigned ret;
  __asm__ volatile (".insn r %1, 3, 0, %0, x0, x0"
      : "=r"(ret) : "i"(RISCV_CUSTOM1));
  return ret;
}

// Raster begin: per-frame trigger. Idempotent in hardware (subsequent
// calls during an active fetch are deduped via the raster's
// fetch_triggered state), so multiple warps can call it concurrently
// without a barrier. Must be issued once per frame by at least one
// participating warp before any vx_rast() call.
inline void vx_rast_begin() {
  __asm__ volatile (".insn r %0, 4, 0, x0, x0, x0"
      :: "i"(RISCV_CUSTOM1) : "memory");
}

} // namespace graphics
} // namespace vortex

#endif // __VORTEX__
