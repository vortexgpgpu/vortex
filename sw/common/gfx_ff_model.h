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

// Host-side software models of the fixed-function TEX / OM / RASTER units.
// Consumed by simx. Lives in sw/common/ (shared across sw/ and sim/);
// must not include from sw/kernel/include/ or sw/runtime/include/.
//
// The on-wire ABI types (fixed_t<F>, vec3e_t, rast_prim_t, ...) come from
// <vx_gfx_abi.h>, shared with sw/kernel/include/vx_graphics.h.
// Do not redefine them here.

#pragma once

#include <cassert>
#include <cstdint>
#include "vx_gfx_abi.h"
#include "gfx_frag_tex.h"   // single source of truth for the tex-sampling math
#include "gfx_dcr.h"      // DCR address → state-index helpers (shared w/ graphics.h)
#include <VX_types.h>

// DCR-state counts derived locally from the state windows (VX_types keeps the
// BEGIN/END leaves); these size the per-unit FF state arrays below.
#ifndef VX_DCR_TEX_STATE_COUNT
#define VX_DCR_TEX_STATE_COUNT    (VX_DCR_TEX_STATE_END - VX_DCR_TEX_STATE_BEGIN)
#endif
#ifndef VX_DCR_RASTER_STATE_COUNT
#define VX_DCR_RASTER_STATE_COUNT (VX_DCR_RASTER_STATE_END - VX_DCR_RASTER_STATE_BEGIN)
#endif
#ifndef VX_DCR_OM_STATE_COUNT
#define VX_DCR_OM_STATE_COUNT     (VX_DCR_OM_STATE_END - VX_DCR_OM_STATE_BEGIN)
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
    for (auto& rt : states_) {
      for (auto& state : rt) {
        state = 0;
      }
    }
    rt_ = 0;
  }

  uint32_t read(uint32_t rt, uint32_t addr) const {
    assert(rt < VX_OM_MAX_RT);
    uint32_t state = VX_DCR_OM_STATE(addr);
    assert(state < VX_DCR_OM_STATE_COUNT);
    return states_[rt][state];
  }

  uint32_t read(uint32_t addr) const {
    return this->read(0, addr);
  }

  void write(uint32_t addr, uint32_t value) {
    if (addr == VX_DCR_OM_RT_SELECT) {
      assert(value < VX_OM_MAX_RT);
      rt_ = value;
      return;
    }
    uint32_t state = VX_DCR_OM_STATE(addr);
    assert(state < VX_DCR_OM_STATE_COUNT);
    // A shared register is stored in every attachment's copy so that reading it
    // back never depends on which attachment happened to be selected last.
    if (is_per_rt(addr)) {
      states_[rt_][state] = value;
    } else {
      for (auto& rt : states_) {
        rt[state] = value;
      }
    }
  }

private:
  static bool is_per_rt(uint32_t addr) {
    switch (addr) {
    case VX_DCR_OM_CBUF_ADDR:
    case VX_DCR_OM_CBUF_PITCH:
    case VX_DCR_OM_CBUF_WRITEMASK:
    case VX_DCR_OM_BLEND_MODE:
    case VX_DCR_OM_BLEND_FUNC:
    case VX_DCR_OM_BLEND_CONST:
    case VX_DCR_OM_LOGIC_OP:
      return true;
    default:
      return false;
    }
  }

  uint32_t states_[VX_OM_MAX_RT][VX_DCR_OM_STATE_COUNT];
  uint32_t rt_;
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

// Address/filter descriptor for one (u, v, lod) sample: per-sample byte
// addresses, stride, blend fractions, and format/filter selectors for
// VX_tex_sampler.
// The address/filter descriptor + the sampling math live in gfx_frag_tex.h (the
// single source of truth shared with the device SW fallback); alias it here so
// existing call sites (TextureSampler, tex_core) are unchanged.
using TexelRequest = gfx_tex::TexelRequest;

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

  // is the mip filter trilinear (blend two LODs) for this stage?
  bool mip_linear(uint32_t stage) const {
    return (dcrs_.read(stage, VX_DCR_TEX_FILTER) & VX_TEX_FILTER_MIP_LINEAR) != 0;
  }

protected:
  TexDCRS  dcrs_;
  MemoryCB mem_cb_;
  void*    cb_arg_;
};

// Trilinear LOD blend lives in gfx_frag_tex.h (single source of truth shared with
// the device SW fallback); alias it so existing call sites are unchanged.
using gfx_tex::TexLodLerp;

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

  void configure(const OMDCRS& dcrs, uint32_t rt);

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
             uint32_t tile_logsize);
  ~Rasterizer();

  void configure(const RasterDCRS& dcrs);

  void renderPrimitive(uint32_t x,
                       uint32_t y,
                       uint32_t pid,
                       vec3e_t edges[4]) const;

protected:

  // The recursive tile→quad coverage walk lives in gfx_frag_rast.h (single source of
  // truth shared with the device SW fallback); renderPrimitive forwards to
  // it with this class's ShaderCB as the emit sink.
  ShaderCB shader_cb_;
  void*    cb_arg_;
  uint32_t tile_logsize_;
  uint32_t scissor_left_;
  uint32_t scissor_top_;
  uint32_t scissor_right_;
  uint32_t scissor_bottom_;
};

} // namespace vortex
