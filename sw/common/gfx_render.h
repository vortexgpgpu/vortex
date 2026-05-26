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

// h/w-internal host classes that mirror the fixed-function TEX / OM / RASTER
// units in software. Consumed only by simx (cycle-approximate model). This
// header MUST be self-contained — it lives in sw/common/ which is the
// shared layer across sw/ and sim/, and must not reach into either
// sw/kernel/include/ or sw/runtime/include/ (per the sw/↔sim/+hw/
// bidirectional isolation rule; see AGENTS.md §6).
//
// The on-wire types defined inline below MIRROR the SDK-canonical copy
// in sw/kernel/include/vx_graphics.h. The two must stay in sync — any
// change to the on-wire ABI must be applied in both locations. CI
// enforces sync via ci/check_sw_sim_boundary.sh (TODO: add diff check).

#pragma once

#include <cassert>
#include <cstdint>
#include <VX_types.h>

namespace vortex {
namespace graphics {

///////////////////////////////////////////////////////////////////////////////
// On-wire fixed-point type. F is the number of fractional bits.
// Trivially-copyable: the ABI is just the int32_t bits.
// Mirror of sw/kernel/include/vx_graphics.h::fixed_t — keep in sync.
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

  constexpr fixed_t operator<<(int n) const { return make(bits << n); }
  constexpr fixed_t operator>>(int n) const { return make(bits >> n); }

  fixed_t& operator+=(fixed_t r) { bits += r.bits; return *this; }
  fixed_t& operator-=(fixed_t r) { bits -= r.bits; return *this; }

  constexpr bool operator==(fixed_t r) const { return bits == r.bits; }
  constexpr bool operator!=(fixed_t r) const { return bits != r.bits; }
  constexpr bool operator< (fixed_t r) const { return bits <  r.bits; }
  constexpr bool operator<=(fixed_t r) const { return bits <= r.bits; }
  constexpr bool operator> (fixed_t r) const { return bits >  r.bits; }
  constexpr bool operator>=(fixed_t r) const { return bits >= r.bits; }
};

template <int F>
constexpr fixed_t<F> operator*(int l, fixed_t<F> r) { return r * l; }

using fixed16_t = fixed_t<16>;
using fixed24_t = fixed_t<24>;
using FloatE    = fixed16_t;
using FloatA    = fixed24_t;

struct vec2e_t { FloatE x, y; };
struct vec3e_t { FloatE x, y, z; };

struct rast_bbox_t        { uint32_t left, right, top, bottom; };
struct rast_tile_header_t { uint16_t tile_x, tile_y, pids_offset, pids_count; };
struct rast_attrib_t      { FloatA x, y, z; };
struct rast_attribs_t     { rast_attrib_t z, r, g, b, a, u, v; };
struct rast_prim_t        { vec3e_t edges[3]; rast_attribs_t attribs; };

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

// DCR address → state-index mapping (skybox-era function-like macros).
// VX_types.toml only emits scalar `#define`s, so define these helpers
// inline rather than re-introducing a backtick macro on the SV side.
#ifndef VX_DCR_TEX_STATE
#define VX_DCR_TEX_STATE(addr)    ((addr) - VX_DCR_TEX_STATE_BEGIN)
#endif
#ifndef VX_DCR_RASTER_STATE
#define VX_DCR_RASTER_STATE(addr) ((addr) - VX_DCR_RASTER_STATE_BEGIN)
#endif
#ifndef VX_DCR_OM_STATE
#define VX_DCR_OM_STATE(addr)     ((addr) - VX_DCR_OM_STATE_BEGIN)
#endif
#ifndef VX_DCR_TEX_MIPOFF
#define VX_DCR_TEX_MIPOFF(lod)    (VX_DCR_TEX_MIPOFF_BASE + (lod))
#endif

namespace vortex {

// Pull the on-wire types into vortex:: so the host class declarations
// below stay free of `graphics::` prefixes.
using namespace graphics;

class RasterDCRS {
public:
  RasterDCRS() {
    this->clear();
  }

  void clear() {
    for (auto& state : states_) {
      state = 0;
    }
  }

  uint32_t read(uint32_t addr) const {
    uint32_t state = VX_DCR_RASTER_STATE(addr);
    assert(state < VX_DCR_RASTER_STATE_COUNT);
    return states_[state];
  }

  void write(uint32_t addr, uint32_t value) {
    uint32_t state = VX_DCR_RASTER_STATE(addr);
    assert(state < VX_DCR_RASTER_STATE_COUNT);
    states_[state] = value;
  }

private:
  uint32_t states_[VX_DCR_RASTER_STATE_COUNT];
};

///////////////////////////////////////////////////////////////////////////////

class OMDCRS {
public:
  OMDCRS() {
    this->clear();
  }

  void clear() {
    for (auto& state : states_) {
      state = 0;
    }
  }

  uint32_t read(uint32_t addr) const {
    uint32_t state = VX_DCR_OM_STATE(addr);
    assert(state < VX_DCR_OM_STATE_COUNT);
    return states_[state];
  }

  void write(uint32_t addr, uint32_t value) {
    uint32_t state = VX_DCR_OM_STATE(addr);
    assert(state < VX_DCR_OM_STATE_COUNT);
    states_[state] = value;
  }

private:
  uint32_t states_[VX_DCR_OM_STATE_COUNT];
};

///////////////////////////////////////////////////////////////////////////////

class TexDCRS {
public:
  uint32_t read(uint32_t stage, uint32_t addr) const {
    uint32_t state = VX_DCR_TEX_STATE(addr-1);
    assert(stage < VX_TEX_STAGE_COUNT);
    assert(state < VX_DCR_TEX_STATE_COUNT);
    return states_[stage][state];
  }

  uint32_t read(uint32_t addr) const {
    if (addr == VX_DCR_TEX_STAGE)
      return stage_;
    uint32_t state = VX_DCR_TEX_STATE(addr-1);
    assert(state < VX_DCR_TEX_STATE_COUNT);
    return states_[stage_][state];
  }

  void write(uint32_t addr, uint32_t value) {
    if (addr == VX_DCR_TEX_STAGE) {
      assert(value < VX_TEX_STAGE_COUNT);
      stage_ = value;
      return;
    }
    uint32_t state = VX_DCR_TEX_STATE(addr-1);
    assert(state < VX_DCR_TEX_STATE_COUNT);
    states_[stage_][state] = value;
  }

private:
  uint32_t states_[VX_TEX_STAGE_COUNT][VX_DCR_TEX_STATE_COUNT-1];
  uint32_t stage_;
};

///////////////////////////////////////////////////////////////////////////////

// Decoupled address/filter description for one (u, v, lod) sample. Mirrors
// VX_tex_addr's outputs in the RTL: per-sample byte addresses, stride,
// blend fractions, and the format/filter selectors that VX_tex_sampler needs.
struct TexelRequest {
  uint64_t addr[4];   // [0] always populated; [1..3] only for BILINEAR
  uint32_t stride;    // bytes per texel (1, 2, or 4)
  uint32_t format;    // VX_TEX_FORMAT_*
  uint32_t filter;    // VX_TEX_FILTER_POINT or _BILINEAR
  uint32_t alpha;     // u-fraction (BILINEAR only)
  uint32_t beta;      // v-fraction (BILINEAR only)
};

class TextureSampler {
public:
  typedef void (*MemoryCB)(
    uint32_t* out,
    const uint64_t* addr,
    uint32_t stride,
    uint32_t size,
    void* cb_arg
  );

  TextureSampler(const MemoryCB& mem_cb, void* cb_arg);
  ~TextureSampler();

  void configure(const TexDCRS& dcrs);

  uint32_t read(uint32_t stage, int32_t u, int32_t v, uint32_t lod) const;

  // Pure: produce the TexelRequest for a (stage, u, v, lod) without touching
  // memory. Caller fetches the texels at req.addr[0..k-1] (k=4 for BILINEAR,
  // k=1 for POINT) and feeds them to apply_filter().
  TexelRequest compute_request(uint32_t stage, int32_t u, int32_t v, uint32_t lod) const;

  // Pure: apply the format-decode + bilinear/point filter to fetched texels.
  static uint32_t apply_filter(const TexelRequest& req, const uint32_t texels[4]);

protected:
  TexDCRS  dcrs_;
  MemoryCB mem_cb_;
  void*    cb_arg_;
};

///////////////////////////////////////////////////////////////////////////////

class DepthTencil {
public:
  DepthTencil();
  ~DepthTencil();

  void configure(const OMDCRS& dcrs);

  bool test(uint32_t is_backface,
            uint32_t depth,
            uint32_t depthstencil_val,
            uint32_t* depthstencil_result) const;

  bool depth_enabled() const {
    return depth_enabled_;
  }

  bool stencil_enabled(bool is_backface) const {
    return is_backface ? stencil_back_enabled_ : stencil_front_enabled_;
  }

protected:

  uint32_t depth_func_;
  uint32_t stencil_front_func_;
  uint32_t stencil_front_zpass_;
  uint32_t stencil_front_zfail_;
  uint32_t stencil_front_fail_;
  uint32_t stencil_front_mask_;
  uint32_t stencil_front_ref_;
  uint32_t stencil_back_func_;
  uint32_t stencil_back_zpass_;
  uint32_t stencil_back_zfail_;
  uint32_t stencil_back_fail_;
  uint32_t stencil_back_mask_;
  uint32_t stencil_back_ref_;

  bool depth_enabled_;
  bool stencil_front_enabled_;
  bool stencil_back_enabled_;
};

///////////////////////////////////////////////////////////////////////////////

class Blender {
public:
  Blender();
  ~Blender();

  void configure(const OMDCRS& dcrs);

  uint32_t blend(uint32_t srcColor, uint32_t dstColor) const;

  bool enabled() const {
    return enabled_;
  }

protected:

  uint32_t blend_mode_rgb_;
  uint32_t blend_mode_a_;
  uint32_t blend_src_rgb_;
  uint32_t blend_src_a_;
  uint32_t blend_dst_rgb_;
  uint32_t blend_dst_a_;
  uint32_t blend_const_;
  uint32_t logic_op_;

  bool enabled_;
};

///////////////////////////////////////////////////////////////////////////////

class Rasterizer {
public:
  typedef void (*ShaderCB)(
    uint32_t  pos_mask,
    vec3e_t   bcoords[4],
    uint32_t  pid,
    void*     cb_arg
  );

  Rasterizer(const ShaderCB& shader_cb,
             void* cb_arg,
             uint32_t tile_logsize,
             uint32_t block_logsize);
  ~Rasterizer();

  void configure(const RasterDCRS& dcrs);

  void renderPrimitive(uint32_t x,
                       uint32_t y,
                       uint32_t pid,
                       vec3e_t edges[4]) const;

protected:

  struct delta_t {
    vec3e_t dx;
    vec3e_t dy;
    vec3e_t extents;
  };

  void renderTile(uint32_t subTileLogSize,
                  uint32_t x,
                  uint32_t y,
                  uint32_t id,
                  const vec3e_t& edges,
                  const delta_t& delta) const;

  void renderQuad(uint32_t x,
                  uint32_t y,
                  uint32_t id,
                  const vec3e_t& edges,
                  const delta_t& delta) const;

  ShaderCB shader_cb_;
  void*    cb_arg_;
  uint32_t tile_logsize_;
  uint32_t block_logsize_;
  uint32_t scissor_left_;
  uint32_t scissor_top_;
  uint32_t scissor_right_;
  uint32_t scissor_bottom_;
};

} // namespace vortex
