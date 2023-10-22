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

#pragma once

#include <cocogfx/include/fixed.hpp>
#include <cocogfx/include/math.hpp>
#include <VX_types.h>

#define FIXEDPOINT_RASTERIZER

namespace graphics {

using fixed16_t = cocogfx::TFixed<16>;
using fixed24_t = cocogfx::TFixed<24>;

#ifdef FIXEDPOINT_RASTERIZER
using FloatE = fixed16_t;
using FloatA = fixed24_t;
#else
using FloatE = float;
using FloatA = float;
#endif

using vec2e_t = cocogfx::TVector2<FloatE>;
using vec3e_t = cocogfx::TVector3<FloatE>;

typedef struct {
  FloatA x;
  FloatA y;
  FloatA z;
} rast_attrib_t;

typedef struct {
  rast_attrib_t z;
  rast_attrib_t r;
  rast_attrib_t g;
  rast_attrib_t b;
  rast_attrib_t a;
  rast_attrib_t u;
  rast_attrib_t v;
} rast_attribs_t;

typedef struct {  
  uint32_t left;
  uint32_t right;
  uint32_t top;
  uint32_t bottom;
} rast_bbox_t;

typedef struct {
  vec3e_t edges[3];
  rast_attribs_t attribs;
} rast_prim_t;

typedef struct {
  uint16_t tile_x;
  uint16_t tile_y;
  uint16_t pids_offset;
  uint16_t pids_count;
} rast_tile_header_t;

inline void Unpack8888(uint32_t texel, uint32_t* lo, uint32_t* hi) {
  *lo = texel & 0x00ff00ff;
  *hi = (texel >> 8) & 0x00ff00ff;
}

inline uint32_t Pack8888(uint32_t lo, uint32_t hi) {
  return (hi << 8) | lo;
}

inline uint32_t Lerp8888(uint32_t a, uint32_t b, uint32_t f) {
  uint32_t p = a * (0xff - f) + b * f + 0x00800080;
  uint32_t q = (p >> 8) & 0x00ff00ff;
  return ((p + q) >> 8) & 0x00ff00ff;
}

///////////////////////////////////////////////////////////////////////////////

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

class RopDCRS {
public:
  RopDCRS() {
    this->clear();
  }

  void clear() {
    for (auto& state : states_) {
      state = 0;
    }
  }

  uint32_t read(uint32_t addr) const {    
    uint32_t state = VX_DCR_ROP_STATE(addr);
    assert(state < VX_DCR_ROP_STATE_COUNT);
    return states_[state];
  }

  void write(uint32_t addr, uint32_t value) {    
    uint32_t state = VX_DCR_ROP_STATE(addr);
    assert(state < VX_DCR_ROP_STATE_COUNT);
    states_[state] = value;
  }

private:
  uint32_t states_[VX_DCR_ROP_STATE_COUNT];
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

  void configure(const RopDCRS& dcrs);

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

  void configure(const RopDCRS& dcrs);

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

} // namespace graphics
