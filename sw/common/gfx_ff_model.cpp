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

#include "gfx_ff_model.h"
#include "gfx_sw.h"   // single source of truth for the per-fragment OM ops (§7)
#include "gfx_frag_rast.h"  // single source of truth for the rasterizer coverage walk (§7)
#include "bitmanip.h"
#include <assert.h>
#include <cocogfx/include/color.hpp>
#include <cocogfx/include/fixed.hpp>
#include <cocogfx/include/math.hpp>

#ifdef LLVM_VORTEX
#include <vx_print.h>
#else
#include <stdio.h>
#define vx_printf printf
#endif

using namespace cocogfx;
using namespace vortex;
using namespace vortex::graphics;

///////////////////////////////////////////////////////////////////////////////

using namespace gfx_tex;  // tex-sampling math + helpers now live in gfx_frag_tex.h

namespace vortex {

TextureSampler::TextureSampler(const MemoryCB& mem_cb, void* cb_arg)
  : mem_cb_(mem_cb)
  , cb_arg_(cb_arg)
{}
  
TextureSampler::~TextureSampler() {}

void TextureSampler::configure(const TexDCRS& dcrs) {
  dcrs_ = dcrs;
}

TexelRequest TextureSampler::compute_request(uint32_t stage, int32_t u, int32_t v, uint32_t lod) const {
  auto mip_off  = dcrs_.read(stage, VX_DCR_TEX_MIPOFF(lod));
  auto mip_base = uint64_t(dcrs_.read(stage, VX_DCR_TEX_ADDR)) << 6;
  auto logdim   = dcrs_.read(stage, VX_DCR_TEX_LOGDIM);
  auto format   = dcrs_.read(stage, VX_DCR_TEX_FORMAT);
  // mag/min filter is the low bit; the mip-filter bit (trilinear) lives above
  // it and is handled by the caller, so mask it off for per-LOD tap selection.
  auto filter   = dcrs_.read(stage, VX_DCR_TEX_FILTER) & TEX_FILTER_MAGMIN_MASK;
  auto wrap     = dcrs_.read(stage, VX_DCR_TEX_WRAP);

  // The sampling math is shared with the on-device SW fallback (gfx_frag_tex.h).
  return gfx_tex::tex_compute_request(mip_base + mip_off, logdim, format,
                                      filter, wrap, u, v, lod);
}

uint32_t TextureSampler::apply_filter(const TexelRequest& req, const uint32_t texels[4]) {
  return gfx_tex::tex_apply_filter(req, texels);
}

uint32_t TextureSampler::read(uint32_t stage, int32_t u, int32_t v, uint32_t lod) const {
  auto sample_lod = [&](uint32_t l) -> uint32_t {
    auto req = this->compute_request(stage, u, v, l);
    uint32_t texels[4] = {0, 0, 0, 0};
    uint32_t count = (req.filter == VX_TEX_FILTER_BILINEAR) ? 4 : 1;
    mem_cb_(texels, req.addr, req.stride, count, cb_arg_);
    return apply_filter(req, texels);
  };
  // gfx_v2 §6.8 trilinear: `lod` is fixed-point — sample the two bracketing
  // mips and blend by the fractional part (lod1 clamped to VX_TEX_LOD_MAX).
  if (this->mip_linear(stage)) {
    uint32_t li   = lod >> VX_TEX_LOD_FRAC_BITS;
    uint32_t lj   = (li + 1 < (uint32_t)VX_TEX_LOD_MAX) ? li + 1 : (uint32_t)VX_TEX_LOD_MAX;
    uint32_t frac = lod & ((1u << VX_TEX_LOD_FRAC_BITS) - 1);
    return TexLodLerp(sample_lod(li), sample_lod(lj), frac);
  }
  return sample_lod(lod);
}

///////////////////////////////////////////////////////////////////////////////

namespace {

// The per-fragment output-merger ops live in <gfx_sw.h> as the single source of
// truth shared by this host FF model and the on-device SW fallback
// (gfx_v2_software_fallback.md §7). Thin forwarders keep the existing call
// sites (DepthTencil::test / Blender::blend) unchanged.
inline bool DoCompare(uint32_t func, uint32_t a, uint32_t b) {
  return gfx_sw::DoCompare(func, a, b);
}
inline uint32_t DoStencilOp(uint32_t op, uint32_t ref, uint32_t val) {
  return gfx_sw::DoStencilOp(op, ref, val);
}
inline ColorARGB DoBlendFunc(uint32_t func, ColorARGB src, ColorARGB dst, ColorARGB cst) {
  return gfx_sw::DoBlendFunc(func, src, dst, cst);
}
inline ColorARGB DoBlendMode(uint32_t mode, uint32_t logic_op,
                             ColorARGB src, ColorARGB dst, ColorARGB s, ColorARGB d) {
  return gfx_sw::DoBlendMode(mode, logic_op, src, dst, s, d);
}

}

///////////////////////////////////////////////////////////////////////////////

DepthTencil::DepthTencil() {}

DepthTencil::~DepthTencil() {}

void DepthTencil::configure(const OMDCRS& dcrs) {
  depth_func_          = dcrs.read(VX_DCR_OM_DEPTH_FUNC);
  bool depth_writemask = dcrs.read(VX_DCR_OM_DEPTH_WRITEMASK) & 0x1;

  stencil_front_func_ = dcrs.read(VX_DCR_OM_STENCIL_FUNC) & 0xffff;
  stencil_front_zpass_= dcrs.read(VX_DCR_OM_STENCIL_ZPASS) & 0xffff;
  stencil_front_zfail_= dcrs.read(VX_DCR_OM_STENCIL_ZFAIL) & 0xffff;
  stencil_front_fail_ = dcrs.read(VX_DCR_OM_STENCIL_FAIL) & 0xffff;
  stencil_front_ref_  = dcrs.read(VX_DCR_OM_STENCIL_REF) & 0xffff;
  stencil_front_mask_ = dcrs.read(VX_DCR_OM_STENCIL_MASK) & 0xffff;

  stencil_back_func_  = dcrs.read(VX_DCR_OM_STENCIL_FUNC) >> 16;
  stencil_back_zpass_ = dcrs.read(VX_DCR_OM_STENCIL_ZPASS) >> 16;
  stencil_back_zfail_ = dcrs.read(VX_DCR_OM_STENCIL_ZFAIL) >> 16;
  stencil_back_fail_  = dcrs.read(VX_DCR_OM_STENCIL_FAIL) >> 16;    
  stencil_back_ref_   = dcrs.read(VX_DCR_OM_STENCIL_REF) >> 16;
  stencil_back_mask_  = dcrs.read(VX_DCR_OM_STENCIL_MASK) >> 16;

  depth_enabled_ = !((depth_func_ == VX_OM_DEPTH_FUNC_ALWAYS) && !depth_writemask);
  
  stencil_front_enabled_ = !((stencil_front_func_  == VX_OM_DEPTH_FUNC_ALWAYS) 
                          && (stencil_front_zpass_ == VX_OM_STENCIL_OP_KEEP)
                          && (stencil_front_zfail_ == VX_OM_STENCIL_OP_KEEP));
  
  stencil_back_enabled_ = !((stencil_back_func_  == VX_OM_DEPTH_FUNC_ALWAYS) 
                          && (stencil_back_zpass_ == VX_OM_STENCIL_OP_KEEP)
                          && (stencil_back_zfail_ == VX_OM_STENCIL_OP_KEEP));
}

bool DepthTencil::test(uint32_t is_backface, 
                       uint32_t depth, 
                       uint32_t depthstencil_val, 
                       uint32_t* depthstencil_result) const {
  auto depth_val   = depthstencil_val & OM_DEPTH_MASK;
  auto stencil_val = depthstencil_val >> VX_OM_DEPTH_BITS;
  auto depth_ref   = depth & OM_DEPTH_MASK;
    
  auto stencil_func = is_backface ? stencil_back_func_ : stencil_front_func_;    
  auto stencil_ref  = is_backface ? stencil_back_ref_  : stencil_front_ref_;    
  auto stencil_mask = is_backface ? stencil_back_mask_ : stencil_front_mask_;
  
  auto stencil_ref_m = stencil_ref & stencil_mask;
  auto stencil_val_m = stencil_val & stencil_mask;

  uint32_t stencil_op;

  auto passed = DoCompare(stencil_func, stencil_ref_m, stencil_val_m);
  if (passed) {
    passed = DoCompare(depth_func_, depth_ref, depth_val);
    if (passed) {
      stencil_op = is_backface ? stencil_back_zpass_ : stencil_front_zpass_;              
    } else {
      stencil_op = is_backface ? stencil_back_zfail_ : stencil_front_zfail_;
    } 
  } else {
    stencil_op = is_backface ? stencil_back_fail_ : stencil_front_fail_;
  }
  
  auto stencil_result = DoStencilOp(stencil_op, stencil_ref, stencil_val);
  *depthstencil_result = (stencil_result << VX_OM_DEPTH_BITS) | depth_ref;
  return passed;
}

///////////////////////////////////////////////////////////////////////////////

Blender::Blender() {}
Blender::~Blender() {}

void Blender::configure(const OMDCRS& dcrs) {
  blend_mode_rgb_ = dcrs.read(VX_DCR_OM_BLEND_MODE) & 0xffff;
  blend_mode_a_   = dcrs.read(VX_DCR_OM_BLEND_MODE) >> 16;
  blend_src_rgb_  = (dcrs.read(VX_DCR_OM_BLEND_FUNC) >>  0) & 0xff;
  blend_src_a_    = (dcrs.read(VX_DCR_OM_BLEND_FUNC) >>  8) & 0xff;
  blend_dst_rgb_  = (dcrs.read(VX_DCR_OM_BLEND_FUNC) >> 16) & 0xff;
  blend_dst_a_    = (dcrs.read(VX_DCR_OM_BLEND_FUNC) >> 24) & 0xff;
  blend_const_    = dcrs.read(VX_DCR_OM_BLEND_CONST);
  logic_op_       = dcrs.read(VX_DCR_OM_LOGIC_OP);  

  enabled_        = !((blend_mode_rgb_ == VX_OM_BLEND_MODE_ADD)
                   && (blend_mode_a_   == VX_OM_BLEND_MODE_ADD) 
                   && (blend_src_rgb_  == VX_OM_BLEND_FUNC_ONE) 
                   && (blend_src_a_    == VX_OM_BLEND_FUNC_ONE) 
                   && (blend_dst_rgb_  == VX_OM_BLEND_FUNC_ZERO) 
                   && (blend_dst_a_    == VX_OM_BLEND_FUNC_ZERO));
}

uint32_t Blender::blend(uint32_t srcColor, uint32_t dstColor) const {
  ColorARGB src(srcColor);
  ColorARGB dst(dstColor);
  ColorARGB cst(blend_const_);

  auto s_rgb = DoBlendFunc(blend_src_rgb_, src, dst, cst);
  auto s_a   = DoBlendFunc(blend_src_a_, src, dst, cst);
  auto d_rgb = DoBlendFunc(blend_dst_rgb_, src, dst, cst);
  auto d_a   = DoBlendFunc(blend_dst_a_, src, dst, cst);
  auto rgb   = DoBlendMode(blend_mode_rgb_, logic_op_, src, dst, s_rgb, d_rgb);
  auto a     = DoBlendMode(blend_mode_a_, logic_op_, src, dst, s_a, d_a);
  ColorARGB result(a.a, rgb.r, rgb.g, rgb.b);

  return result.value;
}

///////////////////////////////////////////////////////////////////////////////

Rasterizer::Rasterizer(const ShaderCB& shader_cb,
                       void* cb_arg,
                       uint32_t tile_logsize)
  : shader_cb_(shader_cb)
  , cb_arg_(cb_arg)
  , tile_logsize_(tile_logsize) {
  assert(tile_logsize >= 1);
}

Rasterizer::~Rasterizer() {} 

void Rasterizer::configure(const RasterDCRS& dcrs) {
  scissor_left_  = dcrs.read(VX_DCR_RASTER_SCISSOR_X) & 0xffff;
  scissor_right_ = dcrs.read(VX_DCR_RASTER_SCISSOR_X) >> 16;
  scissor_top_   = dcrs.read(VX_DCR_RASTER_SCISSOR_Y) & 0xffff;
  scissor_bottom_= dcrs.read(VX_DCR_RASTER_SCISSOR_Y) >> 16;
}

void Rasterizer::renderPrimitive(uint32_t x,
                                 uint32_t y,
                                 uint32_t pid,
                                 vec3e_t edges[3]) const {
  // The coverage walk is the single source of truth shared with the device SW
  // fallback (gfx_frag_rast.h, §7); forward the FF model's ShaderCB as the emit sink.
  gfx_rast::RastConfig cfg{
    tile_logsize_,
    scissor_left_, scissor_top_, scissor_right_, scissor_bottom_
  };
  gfx_rast::rast_walk_primitive(cfg, x, y, pid, edges,
    [&](uint32_t pos_mask, vortex::graphics::vec3e_t* bcoords, uint32_t prim_id) {
      shader_cb_(pos_mask, bcoords, prim_id, cb_arg_);
    });
}
} // namespace vortex
