#pragma once

#include "common.h"
#include <cocogfx/include/fixed.hpp>
#include <cocogfx/include/math.hpp>
#include <cocogfx/include/color.hpp>
#include <algorithm>
#include <vx_intrinsics.h>

inline bool DoCompare(uint32_t func, uint32_t a, uint32_t b) {
  switch (func) {
  default:
    std::abort();
  case ROP_DEPTH_FUNC_NEVER:
    return false;
  case ROP_DEPTH_FUNC_LESS:
    return (a < b);
  case ROP_DEPTH_FUNC_EQUAL:
    return (a == b);
  case ROP_DEPTH_FUNC_LEQUAL:
    return (a <= b);
  case ROP_DEPTH_FUNC_GREATER:
    return (a > b);
  case ROP_DEPTH_FUNC_NOTEQUAL:
    return (a != b);
  case ROP_DEPTH_FUNC_GEQUAL:
    return (a >= b);
  case ROP_DEPTH_FUNC_ALWAYS:
    return true;
  }
}

inline uint32_t DoStencilOp(uint32_t op, uint32_t ref, uint32_t val) {
  switch (op) {
  default:
    std::abort();
  case ROP_STENCIL_OP_KEEP:
    return val;
  case ROP_STENCIL_OP_ZERO:
    return 0;
  case ROP_STENCIL_OP_REPLACE:
    return ref;
  case ROP_STENCIL_OP_INCR:
    return (val < 0xff) ? (val + 1) : val;
  case ROP_STENCIL_OP_DECR:
    return (val > 0) ? (val - 1) : val;
  case ROP_STENCIL_OP_INVERT:
    return ~val;
  case ROP_STENCIL_OP_INCR_WRAP:
    return (val + 1) & 0xff;
  case ROP_STENCIL_OP_DECR_WRAP:
    return (val - 1) & 0xff;
  }
}

class DepthTencil {
private:
  uint32_t buf_baseaddr_;
  uint32_t buf_pitch_;
  uint32_t depth_func_;
  bool     depth_writemask_;
  uint32_t stencil_front_func_;
  uint32_t stencil_front_zpass_;
  uint32_t stencil_front_zfail_;
  uint32_t stencil_front_fail_;
  uint32_t stencil_front_mask_;
  uint32_t stencil_front_writemask_;
  uint32_t stencil_front_ref_;
  uint32_t stencil_back_func_;
  uint32_t stencil_back_zpass_;
  uint32_t stencil_back_zfail_;
  uint32_t stencil_back_fail_;
  uint32_t stencil_back_mask_;
  uint32_t stencil_back_ref_;
  uint32_t stencil_back_writemask_;
  bool depth_enabled_;
  bool stencil_front_enabled_;
  bool stencil_back_enabled_;

  bool doDepthStencilTest(uint32_t x, uint32_t y, uint32_t is_backface, uint32_t depth)  { 
    auto depth_ref     = depth & ROP_DEPTH_MASK;    
    auto stencil_func  = is_backface ? stencil_back_func_ : stencil_front_func_;    
    auto stencil_ref   = is_backface ? stencil_back_ref_ : stencil_front_ref_;    
    auto stencil_mask  = is_backface ? stencil_back_mask_ : stencil_front_mask_;
    auto stencil_zpass = is_backface ? stencil_back_zpass_ : stencil_front_zpass_;
    auto stencil_zfail = is_backface ? stencil_back_zfail_ : stencil_front_zfail_;
    auto stencil_fail  = is_backface ? stencil_back_fail_ : stencil_front_fail_;
    auto stencil_writemask = is_backface ? stencil_back_writemask_ : stencil_front_writemask_;
    auto stencil_ref_m = stencil_ref & stencil_mask;

    auto buf_ptr = reinterpret_cast<uint32_t*>(buf_baseaddr_ + y * buf_pitch_ + x * 4);

    uint32_t stored_value = *buf_ptr;

    uint32_t stencil_val = stored_value >> ROP_DEPTH_BITS;
    uint32_t depth_val   = stored_value & ROP_DEPTH_MASK;   

    uint32_t stencil_val_m = stencil_val & stencil_mask;

    uint32_t writeMask = (stencil_writemask << ROP_DEPTH_BITS) 
                       | (depth_writemask_ ? ROP_DEPTH_MASK : 0);

    uint32_t stencil_op = stencil_fail;

    bool passed = false;

    bool spassed = DoCompare(stencil_func, stencil_ref_m, stencil_val_m);    
    if (spassed) {
      bool dpassed = DoCompare(depth_func_, depth_ref, depth_val);
      if (dpassed) {
        stencil_op = stencil_zpass;
        passed = true;
      } else {
        stencil_op = stencil_zfail;
      } 
    }

    auto stencil_result = DoStencilOp(stencil_op, stencil_ref, stencil_val);

    // Write the depth stencil value
    if (writeMask) {
      auto merged_value = (stencil_result << ROP_DEPTH_BITS) | depth_ref;
      auto write_value = (stored_value & ~writeMask) | (merged_value & writeMask);
      *buf_ptr = write_value;
    }

    return passed;
  }

public:

  void initialize(const RopDCRS& dcrs) {
    // get device configuration
    buf_baseaddr_       = dcrs.read(DCR_ROP_ZBUF_ADDR);
    buf_pitch_          = dcrs.read(DCR_ROP_ZBUF_PITCH);
    
    depth_func_         = dcrs.read(DCR_ROP_DEPTH_FUNC);
    depth_writemask_    = dcrs.read(DCR_ROP_DEPTH_WRITEMASK) & 0x1;

    stencil_front_func_ = dcrs.read(DCR_ROP_STENCIL_FUNC) & 0xffff;
    stencil_front_zpass_= dcrs.read(DCR_ROP_STENCIL_ZPASS) & 0xffff;
    stencil_front_zfail_= dcrs.read(DCR_ROP_STENCIL_ZFAIL) & 0xffff;
    stencil_front_fail_ = dcrs.read(DCR_ROP_STENCIL_FAIL) & 0xffff;
    stencil_front_ref_  = dcrs.read(DCR_ROP_STENCIL_REF) & 0xffff;
    stencil_front_mask_ = dcrs.read(DCR_ROP_STENCIL_MASK) & 0xffff;    
    stencil_front_writemask_ = dcrs.read(DCR_ROP_STENCIL_WRITEMASK) & 0xffff;

    stencil_back_func_  = dcrs.read(DCR_ROP_STENCIL_FUNC) >> 16;
    stencil_back_zpass_ = dcrs.read(DCR_ROP_STENCIL_ZPASS) >> 16;
    stencil_back_zfail_ = dcrs.read(DCR_ROP_STENCIL_ZFAIL) >> 16;
    stencil_back_fail_  = dcrs.read(DCR_ROP_STENCIL_FAIL) >> 16;    
    stencil_back_ref_   = dcrs.read(DCR_ROP_STENCIL_REF) >> 16;
    stencil_back_mask_  = dcrs.read(DCR_ROP_STENCIL_MASK) >> 16;
    stencil_back_writemask_ = dcrs.read(DCR_ROP_STENCIL_WRITEMASK) >> 16;

    depth_enabled_      = !((depth_func_ == ROP_DEPTH_FUNC_ALWAYS) 
                         && !depth_writemask_);
    
    stencil_front_enabled_ = !((stencil_front_func_  == ROP_DEPTH_FUNC_ALWAYS) 
                            && (stencil_front_zpass_ == ROP_STENCIL_OP_KEEP)
                            && (stencil_front_zfail_ == ROP_STENCIL_OP_KEEP));
    
    stencil_back_enabled_ = !((stencil_back_func_  == ROP_DEPTH_FUNC_ALWAYS) 
                           && (stencil_back_zpass_ == ROP_STENCIL_OP_KEEP)
                           && (stencil_back_zfail_ == ROP_STENCIL_OP_KEEP));
  }

  bool write(uint32_t x, uint32_t y, bool is_backface, uint32_t depth) {
    bool stencil_enabled = is_backface ? stencil_back_enabled_ : stencil_front_enabled_;
    if (!stencil_front_enabled_ && !depth_enabled_)
      return true;
      
    return this->doDepthStencilTest(x, y, is_backface, depth);
  }
};

///////////////////////////////////////////////////////////////////////////////

inline uint32_t DoLogicOp(uint32_t op, uint32_t src, uint32_t dst) {
  switch (op) {
  default:
    std::abort();
  case ROP_LOGIC_OP_CLEAR:
    return 0;
  case ROP_LOGIC_OP_AND:
    return src & dst;
  case ROP_LOGIC_OP_AND_REVERSE:
    return src & ~dst;
  case ROP_LOGIC_OP_COPY:
    return src;
  case ROP_LOGIC_OP_AND_INVERTED:
    return ~src & dst;
  case ROP_LOGIC_OP_NOOP:
    return dst;
  case ROP_LOGIC_OP_XOR:
    return src ^ dst;
  case ROP_LOGIC_OP_OR:
    return src | dst;
  case ROP_LOGIC_OP_NOR:
    return ~(src | dst);
  case ROP_LOGIC_OP_EQUIV:
    return ~(src ^ dst);
  case ROP_LOGIC_OP_INVERT:
    return ~dst;
  case ROP_LOGIC_OP_OR_REVERSE:
    return src | ~dst;
  case ROP_LOGIC_OP_COPY_INVERTED:
    return ~src;
  case ROP_LOGIC_OP_OR_INVERTED:
    return ~src | dst;
  case ROP_LOGIC_OP_NAND:
    return ~(src & dst);
  case ROP_LOGIC_OP_SET:
    return 0xffffffff;
  }
}

inline cocogfx::ColorARGB DoBlendFunc(uint32_t func, 
                                      cocogfx::ColorARGB src, 
                                      cocogfx::ColorARGB dst,
                                      cocogfx::ColorARGB cst) {
  switch (func) {
  default:
    std::abort();
  case ROP_BLEND_FUNC_ZERO:
    return cocogfx::ColorARGB(0, 0, 0, 0);
  case ROP_BLEND_FUNC_ONE:
    return cocogfx::ColorARGB(0xff, 0xff, 0xff, 0xff);
  case ROP_BLEND_FUNC_SRC_RGB:
    return src;
  case ROP_BLEND_FUNC_ONE_MINUS_SRC_RGB:
    return cocogfx::ColorARGB(
      0xff - src.a,
      0xff - src.r,
      0xff - src.g,
      0xff - src.b
    );
  case ROP_BLEND_FUNC_DST_RGB:
    return dst;
  case ROP_BLEND_FUNC_ONE_MINUS_DST_RGB:
    return cocogfx::ColorARGB(
      0xff - dst.a,
      0xff - dst.r,
      0xff - dst.g,
      0xff - dst.b
    );
  case ROP_BLEND_FUNC_SRC_A:
    return cocogfx::ColorARGB(src.a, src.a, src.a, src.a);
  case ROP_BLEND_FUNC_ONE_MINUS_SRC_A:
    return cocogfx::ColorARGB(
      0xff - src.a,
      0xff - src.a,
      0xff - src.a,
      0xff - src.a
    );
  case ROP_BLEND_FUNC_DST_A:
    return cocogfx::ColorARGB(dst.a, dst.a, dst.a, dst.a);
  case ROP_BLEND_FUNC_ONE_MINUS_DST_A:
    return cocogfx::ColorARGB(
      0xff - dst.a,
      0xff - dst.a,
      0xff - dst.a,
      0xff - dst.a
    );
  case ROP_BLEND_FUNC_CONST_RGB:
    return cst;
  case ROP_BLEND_FUNC_ONE_MINUS_CONST_RGB:
    return cocogfx::ColorARGB(
      0xff - cst.a,
      0xff - cst.r,
      0xff - cst.g,
      0xff - cst.b
    );
  case ROP_BLEND_FUNC_CONST_A:
    return cocogfx::ColorARGB(cst.a, cst.a, cst.a, cst.a);
  case ROP_BLEND_FUNC_ONE_MINUS_CONST_A:
    return cocogfx::ColorARGB(
      0xff - cst.a,
      0xff - cst.r,
      0xff - cst.g,
      0xff - cst.b
    );
  case ROP_BLEND_FUNC_ALPHA_SAT: {
    auto factor = std::min<int>(src.a, 0xff - dst.a);
    return cocogfx::ColorARGB(0xff, factor, factor, factor);
  }
  }
}

inline cocogfx::ColorARGB DoBlendMode(uint32_t mode, 
                                      uint32_t logic_op,
                                      cocogfx::ColorARGB src, 
                                      cocogfx::ColorARGB dst,
                                      cocogfx::ColorARGB s, 
                                      cocogfx::ColorARGB d) {
  switch (mode) {
  default:
    std::abort();
  case ROP_BLEND_MODE_ADD:
    return cocogfx::ColorARGB(
      cocogfx::Add8(cocogfx::Mul8(src.a, s.a), cocogfx::Mul8(dst.a, d.a)),
      cocogfx::Add8(cocogfx::Mul8(src.r, s.r), cocogfx::Mul8(dst.r, d.r)),
      cocogfx::Add8(cocogfx::Mul8(src.g, s.g), cocogfx::Mul8(dst.g, d.g)),
      cocogfx::Add8(cocogfx::Mul8(src.b, s.b), cocogfx::Mul8(dst.b, d.b))
    );
  case ROP_BLEND_MODE_SUB:
    return cocogfx::ColorARGB(
      cocogfx::Sub8(cocogfx::Mul8(src.a, s.a), cocogfx::Mul8(dst.a, d.a)),
      cocogfx::Sub8(cocogfx::Mul8(src.r, s.r), cocogfx::Mul8(dst.r, d.r)),
      cocogfx::Sub8(cocogfx::Mul8(src.g, s.g), cocogfx::Mul8(dst.g, d.g)),
      cocogfx::Sub8(cocogfx::Mul8(src.b, s.b), cocogfx::Mul8(dst.b, d.b))
    );
  case ROP_BLEND_MODE_REV_SUB:
    return cocogfx::ColorARGB(
      cocogfx::Sub8(cocogfx::Mul8(dst.a, d.a), cocogfx::Mul8(src.a, s.a)),
      cocogfx::Sub8(cocogfx::Mul8(dst.r, d.r), cocogfx::Mul8(src.r, s.r)),
      cocogfx::Sub8(cocogfx::Mul8(dst.g, d.g), cocogfx::Mul8(src.g, s.g)),
      cocogfx::Sub8(cocogfx::Mul8(dst.b, d.b), cocogfx::Mul8(src.b, s.b))
    );
  case ROP_BLEND_MODE_MIN:
    return cocogfx::ColorARGB(
      std::min(src.a, dst.a),
      std::min(src.r, dst.r),
      std::min(src.g, dst.g),
      std::min(src.b, dst.b)
    );
  case ROP_BLEND_MODE_MAX:
    return cocogfx::ColorARGB(
      std::max(src.a, dst.a),
      std::max(src.r, dst.r),
      std::max(src.g, dst.g),
      std::max(src.b, dst.b)
    );
  case ROP_BLEND_MODE_LOGICOP:
    return cocogfx::ColorARGB(DoLogicOp(logic_op, src.value, dst.value));
  }
}

class Blender {
private:  
  uint32_t buf_baseaddr_;
  uint32_t buf_pitch_;
  uint32_t buf_writemask_;

  uint32_t blend_mode_rgb_;
  uint32_t blend_mode_a_;
  uint32_t blend_src_rgb_;
  uint32_t blend_src_a_;
  uint32_t blend_dst_rgb_;
  uint32_t blend_dst_a_;
  uint32_t blend_const_;
  uint32_t logic_op_;
  
  bool blend_enabled_;

  cocogfx::ColorARGB doBlend(cocogfx::ColorARGB src, cocogfx::ColorARGB dst, cocogfx::ColorARGB cst) {        
    auto s_rgb = DoBlendFunc(blend_src_rgb_, src, dst, cst);
    auto s_a   = DoBlendFunc(blend_src_a_, src, dst, cst);
    auto d_rgb = DoBlendFunc(blend_dst_rgb_, src, dst, cst);
    auto d_a   = DoBlendFunc(blend_dst_a_, src, dst, cst);
    auto rgb   = DoBlendMode(blend_mode_rgb_, logic_op_, src, dst, s_rgb, d_rgb);
    auto a     = DoBlendMode(blend_mode_a_, logic_op_, src, dst, s_a, d_a);
    return cocogfx::ColorARGB(a.a, rgb.r, rgb.g, rgb.b);
  }

public:

  void initialize(const RopDCRS& dcrs) {
    // get device configuration
    buf_baseaddr_   = dcrs.read(DCR_ROP_CBUF_ADDR);
    buf_pitch_      = dcrs.read(DCR_ROP_CBUF_PITCH);
    buf_writemask_  = dcrs.read(DCR_ROP_CBUF_WRITEMASK) & 0xf;

    blend_mode_rgb_ = dcrs.read(DCR_ROP_BLEND_MODE) & 0xffff;
    blend_mode_a_   = dcrs.read(DCR_ROP_BLEND_MODE) >> 16;
    blend_src_rgb_  = (dcrs.read(DCR_ROP_BLEND_FUNC) >>  0) & 0xff;
    blend_src_a_    = (dcrs.read(DCR_ROP_BLEND_FUNC) >>  8) & 0xff;
    blend_dst_rgb_  = (dcrs.read(DCR_ROP_BLEND_FUNC) >> 16) & 0xff;
    blend_dst_a_    = (dcrs.read(DCR_ROP_BLEND_FUNC) >> 24) & 0xff;
    blend_const_    = dcrs.read(DCR_ROP_BLEND_CONST);
    logic_op_       = dcrs.read(DCR_ROP_LOGIC_OP);  

    blend_enabled_  = !((blend_mode_rgb_ == ROP_BLEND_MODE_ADD)
                     && (blend_mode_a_   == ROP_BLEND_MODE_ADD) 
                     && (blend_src_rgb_  == ROP_BLEND_FUNC_ONE) 
                     && (blend_src_a_    == ROP_BLEND_FUNC_ONE) 
                     && (blend_dst_rgb_  == ROP_BLEND_FUNC_ZERO) 
                     && (blend_dst_a_    == ROP_BLEND_FUNC_ZERO));
  }

  void write(uint32_t x, uint32_t y, uint32_t color) {
    if (0 == buf_writemask_)
      return;

    auto buf_ptr = reinterpret_cast<uint32_t*>(buf_baseaddr_ + y * buf_pitch_ + x * 4);

    uint32_t stored_value = *buf_ptr;

    cocogfx::ColorARGB src(color);
    cocogfx::ColorARGB dst(stored_value);
    cocogfx::ColorARGB cst(blend_const_);

    cocogfx::ColorARGB result_color;
    if (blend_enabled_) {
      result_color = this->doBlend(src, dst, cst);
    } else {
      result_color = src;
    }

    uint32_t writemask = (((buf_writemask_ >> 0) & 0x1) * 0x000000ff) 
                       | (((buf_writemask_ >> 1) & 0x1) * 0x0000ff00) 
                       | (((buf_writemask_ >> 2) & 0x1) * 0x00ff0000) 
                       | (((buf_writemask_ >> 3) & 0x1) * 0xff000000);
    if (writemask) {
      auto write_value = (stored_value & ~writemask) | (result_color & writemask);
      *buf_ptr = write_value;
    }
  }
};

///////////////////////////////////////////////////////////////////////////////

using fixed16_t = cocogfx::TFixed<16>;
using vec2_fx_t = cocogfx::TVector2<fixed16_t>;
using vec3_fx_t = cocogfx::TVector3<fixed16_t>;

typedef void (*RasterShaderCB)(
  kernel_arg_t* arg,
  uint32_t  x,
  uint32_t  y,
  uint32_t  mask,
  const vec3_fx_t* bcoords,
  uint32_t  pid
);

void shader_function_sw_rast_cb(kernel_arg_t* kernel_arg, uint32_t  x, uint32_t  y, uint32_t  mask, const vec3_fx_t* bcoords, uint32_t  pid);

// Evaluate edge function
inline fixed16_t evalEdgeFunction(const vec3_fx_t& e, uint32_t x, uint32_t y) {
  return (e.x * x) + (e.y * y) + e.z;
}

// Calculate the edge extents for square corners
inline  fixed16_t calcEdgeExtents(const vec3_fx_t& e) {
  auto fxZero = fixed16_t::make(0x0);
  vec2_fx_t corners[4] = {{fxZero, fxZero},  // 00
                          {e.x,    fxZero},  // 10
                          {fxZero, e.y},     // 01
                          {e.x,    e.y}};    // 11
  auto i = (e.y >= fxZero) ? ((e.x >= fxZero) ? 3 : 2) : (e.x >= fxZero) ? 1 : 0;
  return corners[i].x + corners[i].y;
}

class Rasterizer {
private:
  struct primitive_t {
    vec3_fx_t edges[3];
    fixed16_t extents[3];
  };

  kernel_arg_t* kernel_arg_;
  uint32_t tile_logsize_;
  uint32_t block_logsize_;    
  uint32_t num_tiles_;
  uint32_t tbuf_baseaddr_;    
  uint32_t pbuf_baseaddr_;
  uint32_t pbuf_stride_;
  uint32_t dst_width_;
  uint32_t dst_height_;
  uint32_t tile_x_;
  uint32_t tile_y_;
  uint32_t block_pitch_;
  uint32_t log_num_tasks_;

  void renderQuad(uint32_t pid,
                  const primitive_t& primitive, 
                  uint32_t  x, 
                  uint32_t  y, 
                  fixed16_t e0, 
                  fixed16_t e1, 
                  fixed16_t e2) {
    auto fxZero = fixed16_t::make(0x0);
    uint32_t mask = 0;
    vec3_fx_t bcoords[4];

    for (uint32_t j = 0; j < 2; ++j) {
      auto ee0 = e0;
      auto ee1 = e1;
      auto ee2 = e2;
      for (uint32_t i = 0; i < 2; ++i) {
        // test if pixel overlaps triangle
        if (ee0 >= fxZero && ee1 >= fxZero && ee2 >= fxZero) {
          // test if the pixel overlaps rendering region
          if ((x+i) < dst_width_ && (y+j) < dst_height_) {
            uint32_t f = j * 2 + i;          
            mask |= (1 << f);                
            bcoords[f].x = ee0;
            bcoords[f].y = ee1;
            bcoords[f].z = ee2;           
          }
        }
        // update edge equation x components
        ee0 += primitive.edges[0].x;
        ee1 += primitive.edges[1].x;
        ee2 += primitive.edges[2].x;
      }
      // update edge equation y components
      e0 += primitive.edges[0].y;
      e1 += primitive.edges[1].y;
      e2 += primitive.edges[2].y;
    }
    
    if (mask != 0) {
      // invoke the shader
      auto pos_x = x >> 1;
      auto pos_y = y >> 1;
      shader_function_sw_rast_cb(kernel_arg_, pos_x, pos_y, mask, bcoords, pid);
    }
  }

  void renderBlock(uint32_t task_id,
                   uint32_t pid,
                   uint32_t subBlockLogSize, 
                   const primitive_t& primitive, 
                   uint32_t  x, 
                   uint32_t  y, 
                   fixed16_t e0, 
                   fixed16_t e1, 
                   fixed16_t e2) {
    auto fxZero = fixed16_t::make(0x0);

    uint32_t block_x_idx = x >> block_logsize_;
    uint32_t block_y_idx = y >> block_logsize_;
    uint32_t block_idx = block_x_idx + block_y_idx * block_pitch_;

    uint32_t q = block_idx & (log_num_tasks_ - 1);
    if (q != task_id)
      return;

    // check if block overlap triangle    
    if ((e0 + (primitive.extents[0] << subBlockLogSize)) < fxZero 
     || (e1 + (primitive.extents[1] << subBlockLogSize)) < fxZero
     || (e2 + (primitive.extents[2] << subBlockLogSize)) < fxZero)
      return; 
  
    if (subBlockLogSize > 1) {
      // printf("*** raster-block: x=%d, y=%d\n", x, y);

      --subBlockLogSize;
      auto subBlockSize = 1 << subBlockLogSize;
      // draw top-left subtile
      {
        auto sx  = x;
        auto sy  = y;
        auto se0 = e0;
        auto se1 = e1;
        auto se2 = e2;
        this->renderBlock(task_id, pid, subBlockLogSize, primitive, sx, sy, se0, se1, se2);
      }

      // draw top-right subtile
      {
        auto sx  = x + subBlockSize;
        auto sy  = y;
        auto se0 = e0 + (primitive.edges[0].x << subBlockLogSize);
        auto se1 = e1 + (primitive.edges[1].x << subBlockLogSize);
        auto se2 = e2 + (primitive.edges[2].x << subBlockLogSize);
        this->renderBlock(task_id, pid, subBlockLogSize, primitive, sx, sy, se0, se1, se2);
      }

      // draw bottom-left subtile
      {
        auto sx  = x;
        auto sy  = y + subBlockSize;
        auto se0 = e0 + (primitive.edges[0].y << subBlockLogSize);
        auto se1 = e1 + (primitive.edges[1].y << subBlockLogSize);
        auto se2 = e2 + (primitive.edges[2].y << subBlockLogSize);
        this->renderBlock(task_id, pid, subBlockLogSize, primitive, sx, sy, se0, se1, se2);
      }

      // draw bottom-right subtile
      {
        auto sx  = x + subBlockSize;
        auto sy  = y + subBlockSize;
        auto se0 = e0 + (primitive.edges[0].x << subBlockLogSize) + (primitive.edges[0].y << subBlockLogSize);
        auto se1 = e1 + (primitive.edges[1].x << subBlockLogSize) + (primitive.edges[1].y << subBlockLogSize);
        auto se2 = e2 + (primitive.edges[2].x << subBlockLogSize) + (primitive.edges[2].y << subBlockLogSize);
        this->renderBlock(task_id, pid, subBlockLogSize, primitive, sx, sy, se0, se1, se2);
      }
    } else {
      // draw low-level block
      this->renderQuad(pid, primitive, x, y, e0, e1, e2);
    }
  }

  void renderTile(uint32_t task_id,
                  uint32_t pid,
                  uint32_t subTileLogSize, 
                  const primitive_t& primitive, 
                  uint32_t  x, 
                  uint32_t  y, 
                  fixed16_t e0, 
                  fixed16_t e1, 
                  fixed16_t e2) {
    auto fxZero = fixed16_t::make(0x0);
    // check if tile overlap triangle    
    if ((e0 + (primitive.extents[0] << subTileLogSize)) < fxZero 
     || (e1 + (primitive.extents[1] << subTileLogSize)) < fxZero
     || (e2 + (primitive.extents[2] << subTileLogSize)) < fxZero)
      return; 
    
    if (subTileLogSize > block_logsize_) {
      // printf("*** raster-tile: x=%d, y=%d\n", x, y);

      --subTileLogSize;
      auto subTileSize = 1 << subTileLogSize;
      // draw top-left subtile
      {
        auto sx  = x;
        auto sy  = y;
        auto se0 = e0;
        auto se1 = e1;
        auto se2 = e2;
        this->renderTile(task_id, pid, subTileLogSize, primitive, sx, sy, se0, se1, se2);
      }

      // draw top-right subtile
      {
        auto sx  = x + subTileSize;
        auto sy  = y;
        auto se0 = e0 + (primitive.edges[0].x << subTileLogSize);
        auto se1 = e1 + (primitive.edges[1].x << subTileLogSize);
        auto se2 = e2 + (primitive.edges[2].x << subTileLogSize);
        this->renderTile(task_id, pid, subTileLogSize, primitive, sx, sy, se0, se1, se2);
      }

      // draw bottom-left subtile
      {
        auto sx  = x;
        auto sy  = y + subTileSize;
        auto se0 = e0 + (primitive.edges[0].y << subTileLogSize);
        auto se1 = e1 + (primitive.edges[1].y << subTileLogSize);
        auto se2 = e2 + (primitive.edges[2].y << subTileLogSize);
        this->renderTile(task_id, pid, subTileLogSize, primitive, sx, sy, se0, se1, se2);
      }

      // draw bottom-right subtile
      {
        auto sx  = x + subTileSize;
        auto sy  = y + subTileSize;
        auto se0 = e0 + (primitive.edges[0].x << subTileLogSize) + (primitive.edges[0].y << subTileLogSize);
        auto se1 = e1 + (primitive.edges[1].x << subTileLogSize) + (primitive.edges[1].y << subTileLogSize);
        auto se2 = e2 + (primitive.edges[2].x << subTileLogSize) + (primitive.edges[2].y << subTileLogSize);
        this->renderTile(task_id, pid, subTileLogSize, primitive, sx, sy, se0, se1, se2);
      }
    } else {
      if (block_logsize_ > 1) {
        // draw low-level block
        this->renderBlock(task_id, pid, subTileLogSize, primitive, x, y, e0, e1, e2);
      } else {
        // draw low-level quad
        this->renderQuad(pid, primitive, x, y, e0, e1, e2);
      }
    }
  }

public:

  void initialize(const RasterDCRS& dcrs, uint32_t log_num_tasks) {
    // get device configuration
    tile_logsize_  = RASTER_TILE_LOGSIZE;
    block_logsize_ = RASTER_BLOCK_LOGSIZE;
    num_tiles_     = dcrs.read(DCR_RASTER_TILE_COUNT);
    tbuf_baseaddr_ = dcrs.read(DCR_RASTER_TBUF_ADDR);
    pbuf_baseaddr_ = dcrs.read(DCR_RASTER_PBUF_ADDR);
    pbuf_stride_   = dcrs.read(DCR_RASTER_PBUF_STRIDE);
    dst_width_     = dcrs.read(DCR_RASTER_DST_SIZE) & 0xffff;
    dst_height_    = dcrs.read(DCR_RASTER_DST_SIZE) >> 16;    
    block_pitch_   = (dst_width_ + (1 << block_logsize_) - 1) >> block_logsize_;
    log_num_tasks_ = log_num_tasks;
  }

  void render(uint32_t task_id) {  
    auto tbud_ptr = reinterpret_cast<uint32_t*>(tbuf_baseaddr_);

    for (uint32_t cur_tile = 0; cur_tile < num_tiles_; ++cur_tile) {
      // read next tile header from tile buffer
      uint32_t tile_xy   = *tbud_ptr++;
      uint32_t num_prims = *tbud_ptr++;

      tile_x_ = (tile_xy & 0xffff) << tile_logsize_;
      tile_y_ = (tile_xy >> 16) << tile_logsize_;

      for (uint32_t cur_prim = 0; cur_prim < num_prims; ++cur_prim) {
        // read next primitive index from tile buffer
        uint32_t pid = *tbud_ptr++;

        uint32_t x = tile_x_;
        uint32_t y = tile_y_;

        // get primitive edges        
        auto pbuf_ptr = reinterpret_cast<uint32_t*>(pbuf_baseaddr_ + pid * pbuf_stride_);
        primitive_t primitive;
        for (int i = 0; i < 3; ++i) {
          primitive.edges[i].x = fixed16_t::make(*pbuf_ptr++);
          primitive.edges[i].y = fixed16_t::make(*pbuf_ptr++);
          primitive.edges[i].z = fixed16_t::make(*pbuf_ptr++);
        }

        // Add tile corner edge offsets
        primitive.extents[0] = calcEdgeExtents(primitive.edges[0]);
        primitive.extents[1] = calcEdgeExtents(primitive.edges[1]);
        primitive.extents[2] = calcEdgeExtents(primitive.edges[2]);

        // Evaluate edge equation for the starting tile
        auto e0 = evalEdgeFunction(primitive.edges[0], x, y);
        auto e1 = evalEdgeFunction(primitive.edges[1], x, y);
        auto e2 = evalEdgeFunction(primitive.edges[2], x, y);

        // Render the tile
        if (tile_logsize_ > block_logsize_) {
          this->renderTile(task_id, pid, tile_logsize_, primitive, x, y, e0, e1, e2);
        } else {
          this->renderBlock(task_id, pid, block_logsize_, primitive, x, y, e0, e1, e2);
        }
      }
    }
  }
};

///////////////////////////////////////////////////////////////////////////////

class GpuSW {
public:

  void initialize(kernel_arg_t* arg, uint32_t log_num_tasks) {
    rasterizer_.initialize(arg->raster_dcrs, log_num_tasks);
    depthtencil_.initialize(arg->rop_dcrs);
    blender_.initialize(arg->rop_dcrs);
  }

  void rasterize(unsigned task_id) {
    rasterizer_.render(task_id);
  }

  void rop(unsigned x, unsigned y, unsigned isbackface, unsigned color, unsigned depth) {
    bool pass = depthtencil_.write(x, y, isbackface, depth);
    if (pass) {
      blender_.write(x, y, color);
    }
  }

private:  
  Rasterizer  rasterizer_;
  DepthTencil depthtencil_;
  Blender     blender_;
};