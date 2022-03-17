#include "rop_unit.h"
#include "core.h"
#include <VX_config.h>
#include <cocogfx/include/fixed.hpp>
#include <cocogfx/include/math.hpp>
#include <cocogfx/include/color.hpp>
#include <algorithm>

using namespace vortex;

using fixed24_t = cocogfx::TFixed<23>;

static bool DoCompare(uint32_t func, uint32_t a, uint32_t b) {
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

static uint32_t DoStencilOp(uint32_t op, uint32_t ref, uint32_t val) {
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

static uint32_t DoLogicOp(uint32_t op, uint32_t src, uint32_t dst) {
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

class DepthTencil {
private:
  const Arch& arch_;
  const RopUnit::DCRS& dcrs_;
  RAM* mem_;
  uint32_t buf_baseaddr_;
  uint32_t buf_pitch_;
  uint32_t depth_func_;
  bool     depth_mask_;
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
  bool initialized_;

  void initialize() {
    // get device configuration
    buf_baseaddr_       = dcrs_.read(DCR_ROP_ZBUF_ADDR);
    buf_pitch_          = dcrs_.read(DCR_ROP_ZBUF_PITCH);
    depth_func_         = dcrs_.read(DCR_ROP_DEPTH_FUNC);
    depth_mask_         = dcrs_.read(DCR_ROP_DEPTH_MASK);
    stencil_front_func_ = dcrs_.read(DCR_ROP_STENCIL_FUNC) & 0xffff;
    stencil_front_zpass_= dcrs_.read(DCR_ROP_STENCIL_ZPASS) & 0xffff;
    stencil_front_zfail_= dcrs_.read(DCR_ROP_STENCIL_ZFAIL) & 0xffff;
    stencil_front_fail_ = dcrs_.read(DCR_ROP_STENCIL_FAIL) & 0xffff;
    stencil_front_mask_ = dcrs_.read(DCR_ROP_STENCIL_MASK) & 0xffff;
    stencil_front_ref_  = dcrs_.read(DCR_ROP_STENCIL_REF) & 0xffff;
    stencil_back_func_  = dcrs_.read(DCR_ROP_STENCIL_FUNC) >> 16;
    stencil_back_zpass_ = dcrs_.read(DCR_ROP_STENCIL_ZPASS) >> 16;
    stencil_back_zfail_ = dcrs_.read(DCR_ROP_STENCIL_ZFAIL) >> 16;
    stencil_back_fail_  = dcrs_.read(DCR_ROP_STENCIL_FAIL) >> 16;
    stencil_back_mask_  = dcrs_.read(DCR_ROP_STENCIL_MASK) >> 16;
    stencil_back_ref_   = dcrs_.read(DCR_ROP_STENCIL_REF) >> 16;

    depth_enabled_      = !((depth_func_ == ROP_DEPTH_FUNC_ALWAYS) && !depth_mask_);
    
    stencil_front_enabled_ = !((stencil_front_func_  == ROP_DEPTH_FUNC_ALWAYS) 
                            && (stencil_front_zpass_ == ROP_STENCIL_OP_KEEP)
                            && (stencil_front_zfail_ == ROP_STENCIL_OP_KEEP));
    
    stencil_back_enabled_ = !((stencil_back_func_  == ROP_DEPTH_FUNC_ALWAYS) 
                           && (stencil_back_zpass_ == ROP_STENCIL_OP_KEEP)
                           && (stencil_back_zfail_ == ROP_STENCIL_OP_KEEP));

    initialized_        = true;
  }

  bool doDepthStencilTest(uint32_t x, uint32_t y, uint32_t isBackface, uint32_t depth)  { 
    auto depth_ref     = depth & fixed24_t::MASK;    
    auto stencil_func  = isBackface ? stencil_back_func_ : stencil_front_func_;    
    auto stencil_mask  = isBackface ? stencil_back_mask_ : stencil_front_mask_;
    auto stencil_ref   = isBackface ? stencil_back_ref_ : stencil_front_ref_;    
    auto stencil_ref_m = stencil_ref & stencil_mask;

    uint32_t buf_addr = buf_baseaddr_ + y * buf_pitch_ + x * 4;

    uint32_t stored_value;          
    mem_->read(&stored_value, buf_addr, 4);          

    uint32_t stencil_val = stored_value >> 24;
    uint32_t depth_val   = stored_value & 0xffffff;   

    uint32_t stencil_val_m = stencil_val & stencil_mask;

    uint32_t writeMask = stencil_mask << 24;

    uint32_t stencil_op;

    auto passed = DoCompare(stencil_func, stencil_ref_m, stencil_val_m);
    if (passed) {
      passed = DoCompare(depth_func_, depth_ref, depth_val);
      if (passed) {
        if (depth_mask_) {
          writeMask |= 0xffffff;
        }
        stencil_op = isBackface ? stencil_back_zpass_ : stencil_front_zpass_;              
      } else {
        stencil_op = isBackface ? stencil_back_zfail_ : stencil_front_zfail_;
      } 
    } else {
      stencil_op = isBackface ? stencil_back_fail_ : stencil_front_fail_;
    }

    auto stencil_result = DoStencilOp(stencil_op, stencil_ref, stencil_val);

    // Write the depth stencil value
    auto merged_value = (stencil_result << 24) | depth_ref;
    auto write_value = (stored_value & ~writeMask) | (merged_value & writeMask);
    mem_->write(&write_value, buf_addr, 4);

    return passed;
  }

public:
  DepthTencil(const Arch& arch, const RopUnit::DCRS& dcrs) 
    : arch_(arch)
    , dcrs_(dcrs)  
    , initialized_(false)
  {}

  ~DepthTencil() {}

  void clear() {
    initialized_ = false;
  }

  void attach_ram(RAM* mem) {
    mem_ = mem;
  }

  bool write(uint32_t x, uint32_t y, bool isBackface, uint32_t depth) {
    if (!initialized_) {
      this->initialize();
    }
    auto stencil_enabled = isBackface ? stencil_back_enabled_ : stencil_front_enabled_;
    if (stencil_enabled || depth_enabled_) {
      return this->doDepthStencilTest(x, y, isBackface, depth);
    }
    return true;
  }
};

class Blender {
private:
  const Arch& arch_;
  const RopUnit::DCRS& dcrs_;
  RAM* mem_;
  uint32_t buf_baseaddr_;
  uint32_t buf_pitch_;
  uint32_t blend_mode_rgb_;
  uint32_t blend_mode_a_;
  uint32_t blend_src_rgb_;
  uint32_t blend_src_a_;
  uint32_t blend_dst_rgb_;
  uint32_t blend_dst_a_;
  uint32_t blend_const_;
  uint32_t logic_op_;
  uint32_t write_mask_;
  bool blend_enabled_;
  bool initialized_;

  void initialize() {
    // get device configuration
    buf_baseaddr_   = dcrs_.read(DCR_ROP_CBUF_ADDR);
    buf_pitch_      = dcrs_.read(DCR_ROP_CBUF_PITCH);
    write_mask_     = dcrs_.read(DCR_ROP_CBUF_MASK);
    blend_mode_rgb_ = dcrs_.read(DCR_ROP_BLEND_MODE) & 0xffff;
    blend_mode_a_   = dcrs_.read(DCR_ROP_BLEND_MODE) >> 16;
    blend_src_rgb_  = (dcrs_.read(DCR_ROP_BLEND_FUNC) >>  0) & 0xff;
    blend_src_a_    = (dcrs_.read(DCR_ROP_BLEND_FUNC) >>  8) & 0xff;
    blend_dst_rgb_  = (dcrs_.read(DCR_ROP_BLEND_FUNC) >> 16) & 0xff;
    blend_dst_a_    = (dcrs_.read(DCR_ROP_BLEND_FUNC) >> 24) & 0xff;
    blend_const_    = dcrs_.read(DCR_ROP_BLEND_CONST);
    logic_op_       = dcrs_.read(DCR_ROP_LOGIC_OP);    
    blend_enabled_  = !((blend_mode_rgb_ == ROP_BLEND_MODE_LOGICOP) 
                    && (logic_op_ == ROP_LOGIC_OP_COPY));
    initialized_    = true;
  }

public:
  Blender(const Arch& arch, const RopUnit::DCRS& dcrs) 
    : arch_(arch)
    , dcrs_(dcrs) 
    , initialized_(false)
  {}

  ~Blender() {}

  void clear() {
    initialized_ = false;
  }

  void attach_ram(RAM* mem) {
    mem_ = mem;
  }

  cocogfx::ColorARGB doBlend(cocogfx::ColorARGB src, cocogfx::ColorARGB dst, cocogfx::ColorARGB cst) {    
    auto s_rgb = DoBlendFunc(blend_src_rgb_, src, dst, cst);
    auto s_a   = DoBlendFunc(blend_src_a_, src, dst, cst);
    auto d_rgb = DoBlendFunc(blend_dst_rgb_, src, dst, cst);
    auto d_a   = DoBlendFunc(blend_dst_a_, src, dst, cst);
    auto rgb   = DoBlendMode(blend_mode_rgb_, logic_op_, src, dst, s_rgb, d_rgb);
    auto a     = DoBlendMode(blend_mode_a_, logic_op_, src, dst, s_a, d_a);
    return cocogfx::ColorARGB(a.a, rgb.r, rgb.g, rgb.b);
  }

  void write(uint32_t x, uint32_t y, uint32_t color) {
    if (!initialized_) {
      this->initialize();
    }

    uint32_t buf_addr = buf_baseaddr_ + y * buf_pitch_ + x * 4;

    uint32_t stored_value;
    mem_->read(&stored_value, buf_addr, 4);   

    cocogfx::ColorARGB src(color);
    cocogfx::ColorARGB dst(stored_value);
    cocogfx::ColorARGB cst(blend_const_);
    auto new_color = this->doBlend(src, dst, cst);

    // printf("color[%d,%d]=(%d,%d,%d,%d)\n", x, y, new_color.r, new_color.g, new_color.b, new_color.a);

    auto write_value = (stored_value & ~write_mask_) | (new_color & write_mask_);
    mem_->write(&write_value, buf_addr, 4);
  }
};

class RopUnit::Impl {
private:
    RopUnit* simobject_;    
    const Arch& arch_;    
    PerfStats perf_stats_;
    DepthTencil depthtencil_;
    Blender blender_;

public:
    Impl(RopUnit* simobject,      
         const Arch &arch,
         const DCRS& dcrs) 
      : simobject_(simobject)
      , arch_(arch)
      , depthtencil_(arch, dcrs)
      , blender_(arch, dcrs)
    {
        this->clear();
    }

    ~Impl() {}

    void clear() {
      depthtencil_.clear();
      blender_.clear();
    }

    void attach_ram(RAM* mem) {
      depthtencil_.attach_ram(mem);
      blender_.attach_ram(mem);
    }   

    void write(uint32_t x, uint32_t y, bool isBackface, uint32_t color, uint32_t depth) {
      if (depthtencil_.write(x, y, isBackface, depth))
        blender_.write(x, y, color);
    }

    void tick() {
      //--
    }    

    const PerfStats& perf_stats() const { 
      return perf_stats_; 
    }
};

///////////////////////////////////////////////////////////////////////////////

RopUnit::RopUnit(const SimContext& ctx, 
                 const char* name,                         
                 const Arch &arch, 
                 const DCRS& dcrs) 
  : SimObject<RopUnit>(ctx, name)
  , MemReq(this)
  , MemRsp(this)
  , impl_(new Impl(this, arch, dcrs)) 
{}

RopUnit::~RopUnit() {
  delete impl_;
}

void RopUnit::reset() {
  impl_->clear();
}

void RopUnit::attach_ram(RAM* mem) {
  impl_->attach_ram(mem);
}

void RopUnit::write(uint32_t x, uint32_t y, bool isBackface, uint32_t color, uint32_t depth) {
  impl_->write(x, y, isBackface, color, depth);
}

void RopUnit::tick() {
  impl_->tick();
}

const RopUnit::PerfStats& RopUnit::perf_stats() const {
  return impl_->perf_stats();
}