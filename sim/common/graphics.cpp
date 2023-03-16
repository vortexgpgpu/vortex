#include "graphics.h"
#include <assert.h>
#include <cocogfx/include/color.hpp>

using namespace cocogfx;
using namespace graphics;

static FloatE fxZero(0);

static inline bool DoCompare(uint32_t func, uint32_t a, uint32_t b) {
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

static inline uint32_t DoStencilOp(uint32_t op, uint32_t ref, uint32_t val) {
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

static inline uint32_t DoLogicOp(uint32_t op, uint32_t src, uint32_t dst) {
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

static cocogfx::ColorARGB DoBlendFunc(uint32_t func, 
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

static cocogfx::ColorARGB DoBlendMode(uint32_t mode, 
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
      cocogfx::Div255(std::min<int>(src.a * s.a + dst.a * d.a, 0xFF00)),
      cocogfx::Div255(std::min<int>(src.r * s.r + dst.r * d.r, 0xFF00)),
      cocogfx::Div255(std::min<int>(src.g * s.g + dst.g * d.g, 0xFF00)),
      cocogfx::Div255(std::min<int>(src.b * s.b + dst.b * d.b, 0xFF00))
    );
  case ROP_BLEND_MODE_SUB:
    return cocogfx::ColorARGB(
      cocogfx::Div255(std::max<int>(src.a * s.a - dst.a * d.a, 0x0)),
      cocogfx::Div255(std::max<int>(src.r * s.r - dst.r * d.r, 0x0)),
      cocogfx::Div255(std::max<int>(src.g * s.g - dst.g * d.g, 0x0)),
      cocogfx::Div255(std::max<int>(src.b * s.b - dst.b * d.b, 0x0))
    );
  case ROP_BLEND_MODE_REV_SUB:
    return cocogfx::ColorARGB(
      cocogfx::Div255(std::max<int>(dst.a * d.a - src.a * s.a, 0x0)),
      cocogfx::Div255(std::max<int>(dst.r * d.r - src.r * s.r, 0x0)),
      cocogfx::Div255(std::max<int>(dst.g * d.g - src.g * s.g, 0x0)),
      cocogfx::Div255(std::max<int>(dst.b * d.b - src.b * s.b, 0x0))
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

///////////////////////////////////////////////////////////////////////////////

DepthTencil::DepthTencil() {}

DepthTencil::~DepthTencil() {}

void DepthTencil::configure(const RopDCRS& dcrs) {
  // get device configuration
  depth_func_          = dcrs.read(DCR_ROP_DEPTH_FUNC);
  bool depth_writemask = dcrs.read(DCR_ROP_DEPTH_WRITEMASK) & 0x1;

  stencil_front_func_ = dcrs.read(DCR_ROP_STENCIL_FUNC) & 0xffff;
  stencil_front_zpass_= dcrs.read(DCR_ROP_STENCIL_ZPASS) & 0xffff;
  stencil_front_zfail_= dcrs.read(DCR_ROP_STENCIL_ZFAIL) & 0xffff;
  stencil_front_fail_ = dcrs.read(DCR_ROP_STENCIL_FAIL) & 0xffff;
  stencil_front_ref_  = dcrs.read(DCR_ROP_STENCIL_REF) & 0xffff;
  stencil_front_mask_ = dcrs.read(DCR_ROP_STENCIL_MASK) & 0xffff;

  stencil_back_func_  = dcrs.read(DCR_ROP_STENCIL_FUNC) >> 16;
  stencil_back_zpass_ = dcrs.read(DCR_ROP_STENCIL_ZPASS) >> 16;
  stencil_back_zfail_ = dcrs.read(DCR_ROP_STENCIL_ZFAIL) >> 16;
  stencil_back_fail_  = dcrs.read(DCR_ROP_STENCIL_FAIL) >> 16;    
  stencil_back_ref_   = dcrs.read(DCR_ROP_STENCIL_REF) >> 16;
  stencil_back_mask_  = dcrs.read(DCR_ROP_STENCIL_MASK) >> 16;

  depth_enabled_ = !((depth_func_ == ROP_DEPTH_FUNC_ALWAYS) && !depth_writemask);
  
  stencil_front_enabled_ = !((stencil_front_func_  == ROP_DEPTH_FUNC_ALWAYS) 
                          && (stencil_front_zpass_ == ROP_STENCIL_OP_KEEP)
                          && (stencil_front_zfail_ == ROP_STENCIL_OP_KEEP));
  
  stencil_back_enabled_ = !((stencil_back_func_  == ROP_DEPTH_FUNC_ALWAYS) 
                          && (stencil_back_zpass_ == ROP_STENCIL_OP_KEEP)
                          && (stencil_back_zfail_ == ROP_STENCIL_OP_KEEP));
}

bool DepthTencil::test(uint32_t is_backface, 
                       uint32_t depth, 
                       uint32_t depthstencil_val, 
                       uint32_t* depthstencil_result) {
  auto depth_val   = depthstencil_val & ROP_DEPTH_MASK;
  auto stencil_val = depthstencil_val >> ROP_DEPTH_BITS;
  auto depth_ref   = depth & ROP_DEPTH_MASK;
    
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
  *depthstencil_result = (stencil_result << ROP_DEPTH_BITS) | depth_ref;
  return passed;
}

///////////////////////////////////////////////////////////////////////////////

Blender::Blender() {}
Blender::~Blender() {}

void Blender::configure(const RopDCRS& dcrs) {
  // get device configuration
  blend_mode_rgb_ = dcrs.read(DCR_ROP_BLEND_MODE) & 0xffff;
  blend_mode_a_   = dcrs.read(DCR_ROP_BLEND_MODE) >> 16;
  blend_src_rgb_  = (dcrs.read(DCR_ROP_BLEND_FUNC) >>  0) & 0xff;
  blend_src_a_    = (dcrs.read(DCR_ROP_BLEND_FUNC) >>  8) & 0xff;
  blend_dst_rgb_  = (dcrs.read(DCR_ROP_BLEND_FUNC) >> 16) & 0xff;
  blend_dst_a_    = (dcrs.read(DCR_ROP_BLEND_FUNC) >> 24) & 0xff;
  blend_const_    = dcrs.read(DCR_ROP_BLEND_CONST);
  logic_op_       = dcrs.read(DCR_ROP_LOGIC_OP);  

  enabled_        = !((blend_mode_rgb_ == ROP_BLEND_MODE_ADD)
                    && (blend_mode_a_   == ROP_BLEND_MODE_ADD) 
                    && (blend_src_rgb_  == ROP_BLEND_FUNC_ONE) 
                    && (blend_src_a_    == ROP_BLEND_FUNC_ONE) 
                    && (blend_dst_rgb_  == ROP_BLEND_FUNC_ZERO) 
                    && (blend_dst_a_    == ROP_BLEND_FUNC_ZERO));
}

uint32_t Blender::blend(uint32_t srcColor, uint32_t dstColor) {
  cocogfx::ColorARGB src(srcColor);
  cocogfx::ColorARGB dst(dstColor);
  cocogfx::ColorARGB cst(blend_const_);

  auto s_rgb = DoBlendFunc(blend_src_rgb_, src, dst, cst);
  auto s_a   = DoBlendFunc(blend_src_a_, src, dst, cst);
  auto d_rgb = DoBlendFunc(blend_dst_rgb_, src, dst, cst);
  auto d_a   = DoBlendFunc(blend_dst_a_, src, dst, cst);
  auto rgb   = DoBlendMode(blend_mode_rgb_, logic_op_, src, dst, s_rgb, d_rgb);
  auto a     = DoBlendMode(blend_mode_a_, logic_op_, src, dst, s_a, d_a);
  cocogfx::ColorARGB result(a.a, rgb.r, rgb.g, rgb.b);

  return result.value;
}

///////////////////////////////////////////////////////////////////////////////

inline FloatE EvalEdgeFunction(const vec3e_t& e, int x, int y) {
  return (e.x * x) + (e.y * y) + e.z;
}

inline FloatE CalcEdgeExtents(const vec3e_t& e) {
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
inline cocogfx::TFixed<F> ShiftLeft(const cocogfx::TFixed<F>& value, uint32_t dist) {
  return (value << dist);
}

template <uint32_t F>
inline cocogfx::TFixed<F> ShiftRight(const cocogfx::TFixed<F>& value, uint32_t dist) {
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
  static inline cocogfx::TFixed<F> run(const cocogfx::TFixed<F>& value) {
    return (value >> 1);
  }
};

template <>
struct HalfScaler<1> {
  static inline float run(float value) {
    return value * 1.5;
  }

  template <uint32_t F>
  static inline cocogfx::TFixed<F> run(const cocogfx::TFixed<F>& value) {
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
  scissor_left_  = dcrs.read(DCR_RASTER_SCISSOR_X) & 0xffff;
  scissor_right_ = dcrs.read(DCR_RASTER_SCISSOR_X) >> 16;
  scissor_top_   = dcrs.read(DCR_RASTER_SCISSOR_Y) & 0xffff;
  scissor_bottom_= dcrs.read(DCR_RASTER_SCISSOR_Y) >> 16;
}

void Rasterizer::renderPrimitive(uint32_t x, 
                                 uint32_t y, 
                                 uint32_t pid, 
                                 const std::array<vec3e_t, 3>& edges) {
  /*printf("*** raster-edges={{0x%x, 0x%x, 0x%x}, {0x%x, 0x%x, 0x%x}, {0x%x, 0x%x, 0x%x}}\n", 
    edges[0].x.data(), edges[0].y.data(), edges[0].z.data(),
    edges[1].x.data(), edges[1].y.data(), edges[1].z.data(),
    edges[2].x.data(), edges[2].y.data(), edges[2].z.data());*/

  delta_t delta;
  
  delta.dx.x = edges[0].x;
  delta.dx.y = edges[1].x;
  delta.dx.z = edges[2].x;

  delta.dy.x = edges[0].y;
  delta.dy.y = edges[1].y;
  delta.dy.z = edges[2].y;

  // Evaluate edge equation tile extends
  delta.extents.x = CalcEdgeExtents(edges[0]);
  delta.extents.y = CalcEdgeExtents(edges[1]);
  delta.extents.z = CalcEdgeExtents(edges[2]);

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
                            const delta_t& delta) {
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

    // draw bottom-right subtile
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
                            const delta_t& delta) {
  // check if quad overlap triangle    
  if ((edges.x + ShiftLeft(delta.extents.x, 1)) < fxZero 
   || (edges.y + ShiftLeft(delta.extents.y, 1)) < fxZero
   || (edges.z + ShiftLeft(delta.extents.z, 1)) < fxZero)
    return;

  uint32_t mask = 0;
  std::array<vec3e_t, 4> bcoords;

  #define PREPARE_QUAD(i, j) { \
      auto ee0 = edges.x + (delta.dx.x * i) + (delta.dy.x * j); \
      auto ee1 = edges.y + (delta.dx.y * i) + (delta.dy.y * j); \
      auto ee2 = edges.z + (delta.dx.z * i) + (delta.dy.z * j); \
      bool coverage_test = (ee0 >= fxZero && ee1 >= fxZero && ee2 >= fxZero); \
      bool scissor_test = ((x+i) >= scissor_left_     \
                        && (x+i) <  scissor_right_    \
                        && (y+j) >= scissor_top_      \
                        && (y+j) <  scissor_bottom_); \
      uint32_t f = j * 2 + i;                         \
      mask |= ((coverage_test && scissor_test) << f); \
      bcoords[f].x = ee0;                             \
      bcoords[f].y = ee1;                             \
      bcoords[f].z = ee2;                             \
  }

  PREPARE_QUAD(0, 0)
  PREPARE_QUAD(1, 0)
  PREPARE_QUAD(0, 1)
  PREPARE_QUAD(1, 1)
  
  if (mask) {
    auto quad_x = x / 2;
    auto quad_y = y / 2;
    // printf("*** raster-quad: x_loc = %d, y_loc = %d, pid = %d, mask=%d, bcoords = %d %d %d %d, %d %d %d %d, %d %d %d %d\n",
    //   quad_x, quad_y, pid_, mask,
    //   bcoords[0].x.data(), bcoords[1].x.data(), bcoords[2].x.data(), bcoords[3].x.data(),
    //   bcoords[0].y.data(), bcoords[1].y.data(), bcoords[2].y.data(), bcoords[3].y.data(),
    //   bcoords[0].z.data(), bcoords[1].z.data(), bcoords[2].z.data(), bcoords[3].z.data());
    auto pos_mask = (quad_y << (4 + RASTER_DIM_BITS-1)) | (quad_x << 4) | mask;
    shader_cb_(pos_mask, bcoords, pid, cb_arg_);
  }
}  