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

#include "graphics.h"
#include "bitmanip.h"
#include <assert.h>
#include <cocogfx/include/color.hpp>

#ifdef LLVM_VORTEX
#include <vx_print.h>
#else
#include <stdio.h>
#define vx_printf printf
#endif

using namespace cocogfx;
using namespace graphics;

static FloatE fxZero(0);

///////////////////////////////////////////////////////////////////////////////

namespace {

template <uint32_t F>
int32_t TextureWrap(TFixed<F> fx, uint32_t wrap) {
  int32_t ret;
  switch (wrap) {
  default: 
    assert(false);
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

inline uint32_t FormatStride(uint32_t format) {
  switch (format) {
  default: 
    assert(false);
  case VX_TEX_FORMAT_A8R8G8B8: 
    return 4;
  case VX_TEX_FORMAT_R5G6B5:
  case VX_TEX_FORMAT_A1R5G5B5:
  case VX_TEX_FORMAT_A4R4G4B4:
  case VX_TEX_FORMAT_A8L8:
    return 2;
  case VX_TEX_FORMAT_L8:
  case VX_TEX_FORMAT_A8:
    return 1;
  }
}

inline void Unpack8888(uint32_t format, uint32_t texel, uint32_t* lo, uint32_t* hi) {
  uint32_t r, g, b, a;
  switch (format) {
  default: 
    assert(false);
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

template <uint32_t F, typename T = int32_t>
void TexAddressLinear(TFixed<F,T> fu, 
                      TFixed<F,T> fv, 
                      uint32_t    log_width,
                      uint32_t    log_height,
                      uint32_t    wrapu,
                      uint32_t    wrapv,
                      uint32_t*   addr00,
                      uint32_t*   addr01,
                      uint32_t*   addr10,
                      uint32_t*   addr11,
                      uint32_t*   alpha,
                      uint32_t*   beta
) {
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

  //printf("*** fu=0x%x, fv=0x%x, u0=0x%x, u1=0x%x, v0=0x%x, v1=0x%x, x0=0x%x, x1=0x%x, y0=0x%x, y1=0x%x, addr00=0x%x, addr01=0x%x, addr10=0x%x, addr11=0x%x\n", fu.data(), fv.data(), u0, u1, v0, v1, x0, x1, y0, y1, *addr00, *addr01, *addr10, *addr11);
}

template <uint32_t F, typename T = int32_t>
void TexAddressPoint(TFixed<F,T> fu, 
                     TFixed<F,T> fv, 
                     uint32_t    log_width,
                     uint32_t    log_height,
                     int         wrapu,
                     int         wrapv,
                     uint32_t*   addr
) {
  uint32_t u = TextureWrap(fu, wrapu);
  uint32_t v = TextureWrap(fv, wrapv);
  
  uint32_t x = u >> (TFixed<F,T>::FRAC - log_width);
  uint32_t y = v >> (TFixed<F,T>::FRAC - log_height);
  
  *addr = x + (y << log_width);

  //printf("*** fu=0x%x, fv=0x%x, u=0x%x, v=0x%x, x=0x%x, y=0x%x, addr=0x%x\n", fu.data(), fv.data(), u, v, x, y, *addr);
}

inline uint32_t TexFilterLinear(
  uint32_t format,
  uint32_t texel00,  
  uint32_t texel01,
  uint32_t texel10,
  uint32_t texel11,
  uint32_t alpha,
  uint32_t beta
) {
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

  //printf("*** texel00=0x%x, texel01=0x%x, texel10=0x%x, texel11=0x%x, color=0x%x\n", texel00, texel01, texel10, texel11, color);

  return color;
}

inline uint32_t TexFilterPoint(int format, uint32_t texel) {
  uint32_t color;
  {
    uint32_t cl, ch;
    Unpack8888(format, texel, &cl, &ch);
    color = Pack8888(cl, ch);
  }

  //printf("*** texel=0x%x, color=0x%x\n", texel, color);

  return color;
}

}

TextureSampler::TextureSampler(const MemoryCB& mem_cb, void* cb_arg) 
  : mem_cb_(mem_cb)
  , cb_arg_(cb_arg)
{}
  
TextureSampler::~TextureSampler() {}

void TextureSampler::configure(const TexDCRS& dcrs) {
  dcrs_ = dcrs;
}

uint32_t TextureSampler::read(uint32_t stage, int32_t u, int32_t v, uint32_t lod) const {
  auto mip_off  = dcrs_.read(stage, VX_DCR_TEX_MIPOFF(lod));
  auto mip_base = uint64_t(dcrs_.read(stage, VX_DCR_TEX_ADDR)) << 6;
  auto logdim   = dcrs_.read(stage, VX_DCR_TEX_LOGDIM);      
  auto format   = dcrs_.read(stage, VX_DCR_TEX_FORMAT);    
  auto filter   = dcrs_.read(stage, VX_DCR_TEX_FILTER);    
  auto wrap     = dcrs_.read(stage, VX_DCR_TEX_WRAP);
  
  auto base_addr = mip_base + mip_off;

  auto log_width  = std::max<int32_t>((logdim & 0xffff) - lod, 0);
  auto log_height = std::max<int32_t>((logdim >> 16) - lod, 0);
  
  auto wrapu = wrap & 0xffff;
  auto wrapv = wrap >> 16;

  auto stride = FormatStride(format);

  auto xu = TFixed<VX_TEX_FXD_FRAC>::make(u);
  auto xv = TFixed<VX_TEX_FXD_FRAC>::make(v);

  switch (filter) {
  default:
    assert(false);
  case VX_TEX_FILTER_BILINEAR: {
    // addressing
    uint32_t offset00, offset01, offset10, offset11;
    uint32_t alpha, beta;
    TexAddressLinear(xu, xv, log_width, log_height, wrapu, wrapv,
      &offset00, &offset01, &offset10, &offset11, &alpha, &beta);
    
    // memory lookup
    uint32_t texel[4];
    uint64_t addr[4] = {
      base_addr + offset00 * stride,
      base_addr + offset01 * stride,
      base_addr + offset10 * stride,
      base_addr + offset11 * stride
    };
    mem_cb_(texel, addr, stride, 4, cb_arg_);

    // filtering
    auto color = TexFilterLinear(
      format, texel[0], texel[1], texel[2], texel[3], alpha, beta);
    return color;
  }
  case VX_TEX_FILTER_POINT: {
    // addressing
    uint32_t offset;
    TexAddressPoint(xu, xv, log_width, log_height, wrapu, wrapv, &offset);    
    
    // memory lookup
    uint32_t texel;
    uint64_t addr = base_addr + offset * stride;
    mem_cb_(&texel, &addr, stride, 1, cb_arg_);

    // filtering
    auto color = TexFilterPoint(format, texel);
    return color;
  }
  }
}

///////////////////////////////////////////////////////////////////////////////

namespace {

bool DoCompare(uint32_t func, uint32_t a, uint32_t b) {
  switch (func) {
  default:
    assert(false);
  case VX_ROP_DEPTH_FUNC_NEVER:
    return false;
  case VX_ROP_DEPTH_FUNC_LESS:
    return (a < b);
  case VX_ROP_DEPTH_FUNC_EQUAL:
    return (a == b);
  case VX_ROP_DEPTH_FUNC_LEQUAL:
    return (a <= b);
  case VX_ROP_DEPTH_FUNC_GREATER:
    return (a > b);
  case VX_ROP_DEPTH_FUNC_NOTEQUAL:
    return (a != b);
  case VX_ROP_DEPTH_FUNC_GEQUAL:
    return (a >= b);
  case VX_ROP_DEPTH_FUNC_ALWAYS:
    return true;
  }
}

uint32_t DoStencilOp(uint32_t op, uint32_t ref, uint32_t val) {
  switch (op) {
  default:
    assert(false);
  case VX_ROP_STENCIL_OP_KEEP:
    return val;
  case VX_ROP_STENCIL_OP_ZERO:
    return 0;
  case VX_ROP_STENCIL_OP_REPLACE:
    return ref;
  case VX_ROP_STENCIL_OP_INCR:
    return (val < 0xff) ? (val + 1) : val;
  case VX_ROP_STENCIL_OP_DECR:
    return (val > 0) ? (val - 1) : val;
  case VX_ROP_STENCIL_OP_INVERT:
    return ~val;
  case VX_ROP_STENCIL_OP_INCR_WRAP:
    return (val + 1) & 0xff;
  case VX_ROP_STENCIL_OP_DECR_WRAP:
    return (val - 1) & 0xff;
  }
}

uint32_t DoLogicOp(uint32_t op, uint32_t src, uint32_t dst) {
  switch (op) {
  default:
    assert(false);
  case VX_ROP_LOGIC_OP_CLEAR:
    return 0;
  case VX_ROP_LOGIC_OP_AND:
    return src & dst;
  case VX_ROP_LOGIC_OP_AND_REVERSE:
    return src & ~dst;
  case VX_ROP_LOGIC_OP_COPY:
    return src;
  case VX_ROP_LOGIC_OP_AND_INVERTED:
    return ~src & dst;
  case VX_ROP_LOGIC_OP_NOOP:
    return dst;
  case VX_ROP_LOGIC_OP_XOR:
    return src ^ dst;
  case VX_ROP_LOGIC_OP_OR:
    return src | dst;
  case VX_ROP_LOGIC_OP_NOR:
    return ~(src | dst);
  case VX_ROP_LOGIC_OP_EQUIV:
    return ~(src ^ dst);
  case VX_ROP_LOGIC_OP_INVERT:
    return ~dst;
  case VX_ROP_LOGIC_OP_OR_REVERSE:
    return src | ~dst;
  case VX_ROP_LOGIC_OP_COPY_INVERTED:
    return ~src;
  case VX_ROP_LOGIC_OP_OR_INVERTED:
    return ~src | dst;
  case VX_ROP_LOGIC_OP_NAND:
    return ~(src & dst);
  case VX_ROP_LOGIC_OP_SET:
    return 0xffffffff;
  }
}

ColorARGB DoBlendFunc(uint32_t func, 
                      ColorARGB src, 
                      ColorARGB dst,
                      ColorARGB cst) {
  switch (func) {
  default:
    assert(false);
  case VX_ROP_BLEND_FUNC_ZERO:
    return ColorARGB(0, 0, 0, 0);
  case VX_ROP_BLEND_FUNC_ONE:
    return ColorARGB(0xff, 0xff, 0xff, 0xff);
  case VX_ROP_BLEND_FUNC_SRC_RGB:
    return src;
  case VX_ROP_BLEND_FUNC_ONE_MINUS_SRC_RGB:
    return ColorARGB(
      0xff - src.a,
      0xff - src.r,
      0xff - src.g,
      0xff - src.b
    );
  case VX_ROP_BLEND_FUNC_DST_RGB:
    return dst;
  case VX_ROP_BLEND_FUNC_ONE_MINUS_DST_RGB:
    return ColorARGB(
      0xff - dst.a,
      0xff - dst.r,
      0xff - dst.g,
      0xff - dst.b
    );
  case VX_ROP_BLEND_FUNC_SRC_A:
    return ColorARGB(src.a, src.a, src.a, src.a);
  case VX_ROP_BLEND_FUNC_ONE_MINUS_SRC_A:
    return ColorARGB(
      0xff - src.a,
      0xff - src.a,
      0xff - src.a,
      0xff - src.a
    );
  case VX_ROP_BLEND_FUNC_DST_A:
    return ColorARGB(dst.a, dst.a, dst.a, dst.a);
  case VX_ROP_BLEND_FUNC_ONE_MINUS_DST_A:
    return ColorARGB(
      0xff - dst.a,
      0xff - dst.a,
      0xff - dst.a,
      0xff - dst.a
    );
  case VX_ROP_BLEND_FUNC_CONST_RGB:
    return cst;
  case VX_ROP_BLEND_FUNC_ONE_MINUS_CONST_RGB:
    return ColorARGB(
      0xff - cst.a,
      0xff - cst.r,
      0xff - cst.g,
      0xff - cst.b
    );
  case VX_ROP_BLEND_FUNC_CONST_A:
    return ColorARGB(cst.a, cst.a, cst.a, cst.a);
  case VX_ROP_BLEND_FUNC_ONE_MINUS_CONST_A:
    return ColorARGB(
      0xff - cst.a,
      0xff - cst.r,
      0xff - cst.g,
      0xff - cst.b
    );
  case VX_ROP_BLEND_FUNC_ALPHA_SAT: {
    auto factor = std::min<int>(src.a, 0xff - dst.a);
    return ColorARGB(0xff, factor, factor, factor);
  }
  }
}

ColorARGB DoBlendMode(uint32_t mode, 
                      uint32_t logic_op,
                      ColorARGB src, 
                      ColorARGB dst,
                      ColorARGB s, 
                      ColorARGB d) {
  switch (mode) {
  default:
    assert(false);
  case VX_ROP_BLEND_MODE_ADD:
    return ColorARGB(
      Div255(std::min<int>(src.a * s.a + dst.a * d.a + 0x80, 0xFF00)),
      Div255(std::min<int>(src.r * s.r + dst.r * d.r + 0x80, 0xFF00)),
      Div255(std::min<int>(src.g * s.g + dst.g * d.g + 0x80, 0xFF00)),
      Div255(std::min<int>(src.b * s.b + dst.b * d.b + 0x80, 0xFF00))
    );
  case VX_ROP_BLEND_MODE_SUB:
    return ColorARGB(
      Div255(std::max<int>(src.a * s.a - dst.a * d.a + 0x80, 0x0)),
      Div255(std::max<int>(src.r * s.r - dst.r * d.r + 0x80, 0x0)),
      Div255(std::max<int>(src.g * s.g - dst.g * d.g + 0x80, 0x0)),
      Div255(std::max<int>(src.b * s.b - dst.b * d.b + 0x80, 0x0))
    );
  case VX_ROP_BLEND_MODE_REV_SUB:
    return ColorARGB(
      Div255(std::max<int>(dst.a * d.a - src.a * s.a + 0x80, 0x0)),
      Div255(std::max<int>(dst.r * d.r - src.r * s.r + 0x80, 0x0)),
      Div255(std::max<int>(dst.g * d.g - src.g * s.g + 0x80, 0x0)),
      Div255(std::max<int>(dst.b * d.b - src.b * s.b + 0x80, 0x0))
    );
  case VX_ROP_BLEND_MODE_MIN:
    return ColorARGB(
      std::min(src.a, dst.a),
      std::min(src.r, dst.r),
      std::min(src.g, dst.g),
      std::min(src.b, dst.b)
    );
  case VX_ROP_BLEND_MODE_MAX:
    return ColorARGB(
      std::max(src.a, dst.a),
      std::max(src.r, dst.r),
      std::max(src.g, dst.g),
      std::max(src.b, dst.b)
    );
  case VX_ROP_BLEND_MODE_LOGICOP:
    return ColorARGB(DoLogicOp(logic_op, src.value, dst.value));
  }
}

}

///////////////////////////////////////////////////////////////////////////////

DepthTencil::DepthTencil() {}

DepthTencil::~DepthTencil() {}

void DepthTencil::configure(const RopDCRS& dcrs) {
  // get device configuration
  depth_func_          = dcrs.read(VX_DCR_ROP_DEPTH_FUNC);
  bool depth_writemask = dcrs.read(VX_DCR_ROP_DEPTH_WRITEMASK) & 0x1;

  stencil_front_func_ = dcrs.read(VX_DCR_ROP_STENCIL_FUNC) & 0xffff;
  stencil_front_zpass_= dcrs.read(VX_DCR_ROP_STENCIL_ZPASS) & 0xffff;
  stencil_front_zfail_= dcrs.read(VX_DCR_ROP_STENCIL_ZFAIL) & 0xffff;
  stencil_front_fail_ = dcrs.read(VX_DCR_ROP_STENCIL_FAIL) & 0xffff;
  stencil_front_ref_  = dcrs.read(VX_DCR_ROP_STENCIL_REF) & 0xffff;
  stencil_front_mask_ = dcrs.read(VX_DCR_ROP_STENCIL_MASK) & 0xffff;

  stencil_back_func_  = dcrs.read(VX_DCR_ROP_STENCIL_FUNC) >> 16;
  stencil_back_zpass_ = dcrs.read(VX_DCR_ROP_STENCIL_ZPASS) >> 16;
  stencil_back_zfail_ = dcrs.read(VX_DCR_ROP_STENCIL_ZFAIL) >> 16;
  stencil_back_fail_  = dcrs.read(VX_DCR_ROP_STENCIL_FAIL) >> 16;    
  stencil_back_ref_   = dcrs.read(VX_DCR_ROP_STENCIL_REF) >> 16;
  stencil_back_mask_  = dcrs.read(VX_DCR_ROP_STENCIL_MASK) >> 16;

  depth_enabled_ = !((depth_func_ == VX_ROP_DEPTH_FUNC_ALWAYS) && !depth_writemask);
  
  stencil_front_enabled_ = !((stencil_front_func_  == VX_ROP_DEPTH_FUNC_ALWAYS) 
                          && (stencil_front_zpass_ == VX_ROP_STENCIL_OP_KEEP)
                          && (stencil_front_zfail_ == VX_ROP_STENCIL_OP_KEEP));
  
  stencil_back_enabled_ = !((stencil_back_func_  == VX_ROP_DEPTH_FUNC_ALWAYS) 
                          && (stencil_back_zpass_ == VX_ROP_STENCIL_OP_KEEP)
                          && (stencil_back_zfail_ == VX_ROP_STENCIL_OP_KEEP));
}

bool DepthTencil::test(uint32_t is_backface, 
                       uint32_t depth, 
                       uint32_t depthstencil_val, 
                       uint32_t* depthstencil_result) const {
  auto depth_val   = depthstencil_val & VX_ROP_DEPTH_MASK;
  auto stencil_val = depthstencil_val >> VX_ROP_DEPTH_BITS;
  auto depth_ref   = depth & VX_ROP_DEPTH_MASK;
    
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
  *depthstencil_result = (stencil_result << VX_ROP_DEPTH_BITS) | depth_ref;
  return passed;
}

///////////////////////////////////////////////////////////////////////////////

Blender::Blender() {}
Blender::~Blender() {}

void Blender::configure(const RopDCRS& dcrs) {
  // get device configuration
  blend_mode_rgb_ = dcrs.read(VX_DCR_ROP_BLEND_MODE) & 0xffff;
  blend_mode_a_   = dcrs.read(VX_DCR_ROP_BLEND_MODE) >> 16;
  blend_src_rgb_  = (dcrs.read(VX_DCR_ROP_BLEND_FUNC) >>  0) & 0xff;
  blend_src_a_    = (dcrs.read(VX_DCR_ROP_BLEND_FUNC) >>  8) & 0xff;
  blend_dst_rgb_  = (dcrs.read(VX_DCR_ROP_BLEND_FUNC) >> 16) & 0xff;
  blend_dst_a_    = (dcrs.read(VX_DCR_ROP_BLEND_FUNC) >> 24) & 0xff;
  blend_const_    = dcrs.read(VX_DCR_ROP_BLEND_CONST);
  logic_op_       = dcrs.read(VX_DCR_ROP_LOGIC_OP);  

  enabled_        = !((blend_mode_rgb_ == VX_ROP_BLEND_MODE_ADD)
                   && (blend_mode_a_   == VX_ROP_BLEND_MODE_ADD) 
                   && (blend_src_rgb_  == VX_ROP_BLEND_FUNC_ONE) 
                   && (blend_src_a_    == VX_ROP_BLEND_FUNC_ONE) 
                   && (blend_dst_rgb_  == VX_ROP_BLEND_FUNC_ZERO) 
                   && (blend_dst_a_    == VX_ROP_BLEND_FUNC_ZERO));
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

inline FloatE EvalEdgeFunction(const vec3e_t& e, int x, int y) {
  return (e.x * x) + (e.y * y) + e.z;
}

FloatE CalcEdgeExtents(const vec3e_t& e) {
  return (e.y >= fxZero) ? ((e.x >= fxZero) ? (e.x + e.y) : e.y) : 
                           ((e.x >= fxZero) ? e.x : fxZero);
}

inline float ShiftLeft(float value, uint32_t dist) {
  return ldexpf(value, dist);
}

inline float ShiftRight(float value, uint32_t dist) {
  return ldexpf(value, -dist);
}

template <uint32_t F>
inline TFixed<F> ShiftLeft(const TFixed<F>& value, uint32_t dist) {
  return (value << dist);
}

template <uint32_t F>
inline TFixed<F> ShiftRight(const TFixed<F>& value, uint32_t dist) {
  return (value >> dist);
}

template <bool Select>
struct HalfScaler {};

template <>
struct HalfScaler<0> {
  static inline float run(float value) {
    return value * 0.5;
  }

  template <uint32_t F>
  static inline TFixed<F> run(const TFixed<F>& value) {
    return (value >> 1);
  }
};

template <>
struct HalfScaler<1> {
  static inline float run(float value) {
    return value * 1.5;
  }

  template <uint32_t F>
  static inline TFixed<F> run(const TFixed<F>& value) {
    return (value * 3) >> 1;
  }
};

Rasterizer::Rasterizer(const ShaderCB& shader_cb,
                       void* cb_arg,
                       uint32_t tile_logsize, 
                       uint32_t block_logsize) 
  : shader_cb_(shader_cb)
  , cb_arg_(cb_arg)
  , tile_logsize_(tile_logsize)
  , block_logsize_(block_logsize) {
  assert(block_logsize >= 1);
  assert(tile_logsize >= block_logsize);
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
  /*printf("*** raster-edges={{0x%x, 0x%x, 0x%x}, {0x%x, 0x%x, 0x%x}, {0x%x, 0x%x, 0x%x}}\n", 
    edges[0].x.data(), edges[0].y.data(), edges[0].z.data(),
    edges[1].x.data(), edges[1].y.data(), edges[1].z.data(),
    edges[2].x.data(), edges[2].y.data(), edges[2].z.data());*/

  delta_t delta{
    {edges[0].x, edges[1].x, edges[2].x},
    {edges[0].y, edges[1].y, edges[2].y},
    {CalcEdgeExtents(edges[0]), 
     CalcEdgeExtents(edges[1]), 
     CalcEdgeExtents(edges[2])}
  };

  // Evaluate edge equation start values
  vec3e_t value{
    EvalEdgeFunction(edges[0], x, y),
    EvalEdgeFunction(edges[1], x, y),
    EvalEdgeFunction(edges[2], x, y)
  };

  // Render the tile
  this->renderTile(tile_logsize_, x, y, pid, value, delta);
}

void Rasterizer::renderTile(uint32_t tileLogSize,
                            uint32_t x, 
                            uint32_t y, 
                            uint32_t pid,
                            const vec3e_t& edges, 
                            const delta_t& delta) const {
  // check if tile overlap triangle    
  if ((edges.x + ShiftLeft(delta.extents.x, tileLogSize)) < fxZero 
   || (edges.y + ShiftLeft(delta.extents.y, tileLogSize)) < fxZero
   || (edges.z + ShiftLeft(delta.extents.z, tileLogSize)) < fxZero)
    return;
  
  if (tileLogSize > 1) {
    // printf("*** raster-tile: x=%d, y=%d\n", x, y);
    --tileLogSize;
    auto subTileSize = 1 << tileLogSize;    
    {
      // draw top-left subtile
      this->renderTile(tileLogSize, x, y, pid, edges, delta);
    }
    {
      // draw top-right subtile
      int sx = x + subTileSize;
      vec3e_t sedges{
          edges.x + ShiftLeft(delta.dx.x, tileLogSize),
          edges.y + ShiftLeft(delta.dx.y, tileLogSize),
          edges.z + ShiftLeft(delta.dx.z, tileLogSize)
      };
      this->renderTile(tileLogSize, sx, y, pid, sedges, delta);
    }
    {
      // draw bottom-left subtile
      int sy = y + subTileSize;
      vec3e_t sedges{
          edges.x + ShiftLeft(delta.dy.x, tileLogSize),
          edges.y + ShiftLeft(delta.dy.y, tileLogSize),
          edges.z + ShiftLeft(delta.dy.z, tileLogSize)
      };
      this->renderTile(tileLogSize, x, sy, pid, sedges, delta);
    }
    {
      // draw bottom-right subtile
      int sx = x + subTileSize;
      int sy = y + subTileSize;
      vec3e_t sedges{
          edges.x + ShiftLeft(delta.dx.x, tileLogSize) + ShiftLeft(delta.dy.x, tileLogSize),
          edges.y + ShiftLeft(delta.dx.y, tileLogSize) + ShiftLeft(delta.dy.y, tileLogSize),
          edges.z + ShiftLeft(delta.dx.z, tileLogSize) + ShiftLeft(delta.dy.z, tileLogSize)
      };
      this->renderTile(tileLogSize, sx, sy, pid, sedges, delta);
    }
  } else {
    this->renderQuad(x, y, pid, edges, delta);
  }
}

void Rasterizer::renderQuad(uint32_t x, 
                            uint32_t y, 
                            uint32_t pid,
                            const vec3e_t& edges, 
                            const delta_t& delta) const {
  // check if quad overlap triangle    
  if ((edges.x + ShiftLeft(delta.extents.x, 1)) < fxZero 
   || (edges.y + ShiftLeft(delta.extents.y, 1)) < fxZero
   || (edges.z + ShiftLeft(delta.extents.z, 1)) < fxZero)
    return;

  vec3e_t bcoords[4];
  uint32_t mask = 0;  

  #define PREPARE_QUAD(i, j) { \
    auto ee0 = edges.x + (delta.dx.x * i) + (delta.dy.x * j); \
    auto ee1 = edges.y + (delta.dx.y * i) + (delta.dy.y * j); \
    auto ee2 = edges.z + (delta.dx.z * i) + (delta.dy.z * j); \
    bool coverage = (ee0 >= fxZero && ee1 >= fxZero && ee2 >= fxZero \
                  && (x+i) >= scissor_left_ && (x+i) < scissor_right_ \
                  && (y+j) >= scissor_top_  && (y+j) < scissor_bottom_); \
    uint32_t p = j * 2 + i;   \
    mask |= (coverage << p);  \
    bcoords[p].x = ee0;       \
    bcoords[p].y = ee1;       \
    bcoords[p].z = ee2;       \
  }

  PREPARE_QUAD(0, 0)
  PREPARE_QUAD(1, 0)
  PREPARE_QUAD(0, 1)
  PREPARE_QUAD(1, 1)
  
  if (mask) {
    auto quad_x = x / 2;
    auto quad_y = y / 2;
     /*printf("*** raster-quad: x_loc = %d, y_loc = %d, pid = %d, mask=%d, bcoords = %d %d %d %d, %d %d %d %d, %d %d %d %d\n",
       quad_x, quad_y, pid, mask,
       bcoords[0].x.data(), bcoords[1].x.data(), bcoords[2].x.data(), bcoords[3].x.data(),
       bcoords[0].y.data(), bcoords[1].y.data(), bcoords[2].y.data(), bcoords[3].y.data(),
       bcoords[0].z.data(), bcoords[1].z.data(), bcoords[2].z.data(), bcoords[3].z.data());*/
    auto pos_mask = (quad_y << (4 + VX_RASTER_DIM_BITS-1)) | (quad_x << 4) | mask;
    shader_cb_(pos_mask, bcoords, pid, cb_arg_);
  }
}  