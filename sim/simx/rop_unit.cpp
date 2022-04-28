#include "rop_unit.h"
#include "core.h"
#include <VX_config.h>
#include <cocogfx/include/fixed.hpp>
#include <cocogfx/include/math.hpp>
#include <cocogfx/include/color.hpp>
#include <algorithm>

using namespace vortex;

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

public:
  DepthTencil(const Arch& arch, const RopUnit::DCRS& dcrs) 
    : arch_(arch)
    , dcrs_(dcrs)
  {}

  ~DepthTencil() {}

  void initialize() {
    // get device configuration
    depth_func_         = dcrs_.read(DCR_ROP_DEPTH_FUNC);
    bool depth_writemask = dcrs_.read(DCR_ROP_DEPTH_WRITEMASK) & 0x1;

    stencil_front_func_ = dcrs_.read(DCR_ROP_STENCIL_FUNC) & 0xffff;
    stencil_front_zpass_= dcrs_.read(DCR_ROP_STENCIL_ZPASS) & 0xffff;
    stencil_front_zfail_= dcrs_.read(DCR_ROP_STENCIL_ZFAIL) & 0xffff;
    stencil_front_fail_ = dcrs_.read(DCR_ROP_STENCIL_FAIL) & 0xffff;
    stencil_front_ref_  = dcrs_.read(DCR_ROP_STENCIL_REF) & 0xffff;
    stencil_front_mask_ = dcrs_.read(DCR_ROP_STENCIL_MASK) & 0xffff;

    stencil_back_func_  = dcrs_.read(DCR_ROP_STENCIL_FUNC) >> 16;
    stencil_back_zpass_ = dcrs_.read(DCR_ROP_STENCIL_ZPASS) >> 16;
    stencil_back_zfail_ = dcrs_.read(DCR_ROP_STENCIL_ZFAIL) >> 16;
    stencil_back_fail_  = dcrs_.read(DCR_ROP_STENCIL_FAIL) >> 16;    
    stencil_back_ref_   = dcrs_.read(DCR_ROP_STENCIL_REF) >> 16;
    stencil_back_mask_  = dcrs_.read(DCR_ROP_STENCIL_MASK) >> 16;

    depth_enabled_ = !((depth_func_ == ROP_DEPTH_FUNC_ALWAYS) && !depth_writemask);
    
    stencil_front_enabled_ = !((stencil_front_func_  == ROP_DEPTH_FUNC_ALWAYS) 
                            && (stencil_front_zpass_ == ROP_STENCIL_OP_KEEP)
                            && (stencil_front_zfail_ == ROP_STENCIL_OP_KEEP));
    
    stencil_back_enabled_ = !((stencil_back_func_  == ROP_DEPTH_FUNC_ALWAYS) 
                           && (stencil_back_zpass_ == ROP_STENCIL_OP_KEEP)
                           && (stencil_back_zfail_ == ROP_STENCIL_OP_KEEP));
  }

  bool run(uint32_t is_backface, uint32_t depth, uint32_t depthstencil_val, uint32_t* depthstencil_result) {
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

  bool depth_enabled() const {
    return depth_enabled_;
  }

  bool stencil_enabled(bool is_backface) const {
    return is_backface ? stencil_back_enabled_ : stencil_front_enabled_;
  }
};

class Blender {
private:
  const Arch& arch_;
  const RopUnit::DCRS& dcrs_;
  RAM* mem_;

  uint32_t blend_mode_rgb_;
  uint32_t blend_mode_a_;
  uint32_t blend_src_rgb_;
  uint32_t blend_src_a_;
  uint32_t blend_dst_rgb_;
  uint32_t blend_dst_a_;
  uint32_t blend_const_;
  uint32_t logic_op_;
  
  bool enabled_;

public:
  Blender(const Arch& arch, const RopUnit::DCRS& dcrs) 
    : arch_(arch)
    , dcrs_(dcrs)
  {}

  ~Blender() {}

  void initialize() {
    // get device configuration
    blend_mode_rgb_ = dcrs_.read(DCR_ROP_BLEND_MODE) & 0xffff;
    blend_mode_a_   = dcrs_.read(DCR_ROP_BLEND_MODE) >> 16;
    blend_src_rgb_  = (dcrs_.read(DCR_ROP_BLEND_FUNC) >>  0) & 0xff;
    blend_src_a_    = (dcrs_.read(DCR_ROP_BLEND_FUNC) >>  8) & 0xff;
    blend_dst_rgb_  = (dcrs_.read(DCR_ROP_BLEND_FUNC) >> 16) & 0xff;
    blend_dst_a_    = (dcrs_.read(DCR_ROP_BLEND_FUNC) >> 24) & 0xff;
    blend_const_    = dcrs_.read(DCR_ROP_BLEND_CONST);
    logic_op_       = dcrs_.read(DCR_ROP_LOGIC_OP);  

    enabled_        = !((blend_mode_rgb_ == ROP_BLEND_MODE_ADD)
                     && (blend_mode_a_   == ROP_BLEND_MODE_ADD) 
                     && (blend_src_rgb_  == ROP_BLEND_FUNC_ONE) 
                     && (blend_src_a_    == ROP_BLEND_FUNC_ONE) 
                     && (blend_dst_rgb_  == ROP_BLEND_FUNC_ZERO) 
                     && (blend_dst_a_    == ROP_BLEND_FUNC_ZERO));
  }

  uint32_t run(uint32_t srcColor, uint32_t dstColor) {
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

  bool enabled() const {
    return enabled_;
  }
};

class MemoryUnit {
private:
  const Arch& arch_;
  const RopUnit::DCRS& dcrs_;
  RAM* mem_;
  
  uint32_t zbuf_baseaddr_;
  uint32_t zbuf_pitch_;
  bool     depth_writemask_;
  uint32_t stencil_front_writemask_; 
  uint32_t stencil_back_writemask_;

  uint32_t cbuf_baseaddr_;
  uint32_t cbuf_pitch_;
  uint32_t cbuf_writemask_;

  bool color_read_;
  bool color_write_;

public:  
  MemoryUnit(const Arch& arch, const RopUnit::DCRS& dcrs) 
    : arch_(arch)
    , dcrs_(dcrs)
  {}

  void initialize() {
    // get device configuration
    zbuf_baseaddr_ = dcrs_.read(DCR_ROP_ZBUF_ADDR);
    zbuf_pitch_    = dcrs_.read(DCR_ROP_ZBUF_PITCH);
    depth_writemask_ = dcrs_.read(DCR_ROP_DEPTH_WRITEMASK) & 0x1;
    stencil_front_writemask_ = dcrs_.read(DCR_ROP_STENCIL_WRITEMASK) & 0xffff;
    stencil_back_writemask_ = dcrs_.read(DCR_ROP_STENCIL_WRITEMASK) >> 16;

    cbuf_baseaddr_ = dcrs_.read(DCR_ROP_CBUF_ADDR);
    cbuf_pitch_    = dcrs_.read(DCR_ROP_CBUF_PITCH);
    auto cbuf_writemask = dcrs_.read(DCR_ROP_CBUF_WRITEMASK) & 0xf;
    cbuf_writemask_ = (((cbuf_writemask >> 0) & 0x1) * 0x000000ff) 
                    | (((cbuf_writemask >> 1) & 0x1) * 0x0000ff00) 
                    | (((cbuf_writemask >> 2) & 0x1) * 0x00ff0000) 
                    | (((cbuf_writemask >> 3) & 0x1) * 0xff000000);
    color_read_  = (cbuf_writemask != 0xf);
    color_write_ = (cbuf_writemask != 0x0);
  }

  void attach_ram(RAM* mem) {
    mem_ = mem;
  }

  void read(bool depth_enable,
            bool stencil_enable, 
            bool blend_enable,
            uint32_t x, 
            uint32_t y,
            uint32_t* depthstencil,
            uint32_t* color, 
            RopUnit::TraceData::Ptr trace_data) {
    if (depth_enable || stencil_enable) {
      uint32_t zbuf_addr = zbuf_baseaddr_ + y * zbuf_pitch_ + x * 4;
      mem_->read(depthstencil, zbuf_addr, 4);     
      trace_data->mem_rd_addrs.push_back({zbuf_addr, 4});
    }

    if (color_write_ && (color_read_ || blend_enable)) {
      uint32_t cbuf_addr = cbuf_baseaddr_ + y * cbuf_pitch_ + x * 4;
      mem_->read(color, cbuf_addr, 4);
      trace_data->mem_rd_addrs.push_back({cbuf_addr, 4});
    }
  }

  void write(bool depth_enable,
             bool stencil_enable, 
             bool ds_passed,
             bool is_backface,
             uint32_t dst_depthstencil,
             uint32_t dst_color,
             uint32_t x, 
             uint32_t y, 
             uint32_t depthstencil, 
             uint32_t color, 
             RopUnit::TraceData::Ptr trace_data) {
    auto stencil_writemask = is_backface ? stencil_back_writemask_ : stencil_front_writemask_;
    uint32_t writeMask = (stencil_enable ? (stencil_writemask << ROP_DEPTH_BITS) : 0) 
                       | ((depth_enable && ds_passed && depth_writemask_) ? ROP_DEPTH_MASK : 0);
    if (writeMask) {      
      uint32_t write_value = (dst_depthstencil & ~writeMask) | (depthstencil & writeMask);
      uint32_t zbuf_addr = zbuf_baseaddr_ + y * zbuf_pitch_ + x * 4;        
      mem_->write(&write_value, zbuf_addr, 4);
      trace_data->mem_wr_addrs.push_back({zbuf_addr, 4});
      DT(3, "rop-depthstencil: x=" << std::dec << x << ", y=" << y << ", depthstencil=0x" << std::hex << write_value);
    }

    if (color_write_ && ds_passed) {   
      uint32_t write_value = (dst_color & ~cbuf_writemask_) | (color & cbuf_writemask_);
      uint32_t cbuf_addr = cbuf_baseaddr_ + y * cbuf_pitch_ + x * 4;
      mem_->write(&write_value, cbuf_addr, 4);
      trace_data->mem_wr_addrs.push_back({cbuf_addr, 4});
      DT(3, "rop-color: x=" << std::dec << x << ", y=" << y << ", color=0x" << std::hex << write_value);
    }
  }
};

class RopSlices {
public:
  RopSlices(const Arch& arch, const RopUnit::DCRS& dcrs) 
    : memoryUnits_(ROP_NUM_SLICES, {arch, dcrs})
    , depthStencils_(ROP_NUM_SLICES, {arch, dcrs})
    , blenders_(ROP_NUM_SLICES, {arch, dcrs})
    , slice_idx_(0)
  {}

  void clear() {
    for (int i = 0; i < ROP_NUM_SLICES; ++i) {
      depthStencils_.at(i).initialize();
      blenders_.at(i).initialize();
      memoryUnits_.at(i).initialize();
    }
  }

  void attach_ram(RAM* mem) {
    for (int i = 0; i < ROP_NUM_SLICES; ++i) {
      memoryUnits_.at(i).attach_ram(mem);
    }
  }   

  void write(uint32_t x, uint32_t y, bool is_backface, uint32_t color, uint32_t depth, RopUnit::TraceData::Ptr trace_data) {   
    auto& memoryUnit   = memoryUnits_.at(slice_idx_);
    auto& depthStencil = depthStencils_.at(slice_idx_);
    auto& blender      = blenders_.at(slice_idx_);

    auto depth_enabled   = depthStencil.depth_enabled();
    auto stencil_enabled = depthStencil.stencil_enabled(is_backface);
    auto blend_enabled   = blender.enabled();

    uint32_t depthstencil;    
    uint32_t dst_depthstencil;
    uint32_t dst_color;    

    memoryUnit.read(depth_enabled, stencil_enabled, blend_enabled, x, y, &dst_depthstencil, &dst_color, trace_data);
    
    auto ds_passed = !(depth_enabled || stencil_enabled)
                  || depthStencil.run(is_backface, depth, dst_depthstencil, &depthstencil);
    
    if (blend_enabled && ds_passed) {
      color = blender.run(color, dst_color);
    }
    
    memoryUnit.write(depth_enabled, stencil_enabled, ds_passed, is_backface, dst_depthstencil, dst_color, x, y, depthstencil, color, trace_data);     

    slice_idx_ = (slice_idx_ + 1) % ROP_NUM_SLICES;    
  }

private:
  std::vector<::MemoryUnit> memoryUnits_;
  std::vector<DepthTencil>  depthStencils_;
  std::vector<Blender>      blenders_;
  uint32_t                  slice_idx_;
};

class RopUnit::Impl {
private:
  struct pending_req_t {
    TraceData::Ptr data;
    uint32_t count;
  };

  RopUnit* simobject_;    
  const Arch& arch_;    
  PerfStats perf_stats_;
  RopSlices slices_;
  HashTable<pending_req_t> pending_reqs_;

public:
  Impl(RopUnit* simobject,      
       const Arch &arch,
       const DCRS& dcrs) 
    : simobject_(simobject)
    , arch_(arch)
    , slices_(arch, dcrs)
    , pending_reqs_(ROP_MEM_QUEUE_SIZE)
  {
    this->clear();
  }

  ~Impl() {}

  void clear() {
    slices_.clear();
  }

  void attach_ram(RAM* mem) {
    slices_.attach_ram(mem);
  }   

  void write(uint32_t x, uint32_t y, bool is_backface, uint32_t color, uint32_t depth, TraceData::Ptr trace_data) {      
    slices_.write(x, y, is_backface, color, depth, trace_data);
  }

  void tick() {
    // handle memory response
    for (auto& port : simobject_->MemRsps) {
      if (port.empty())
        continue;
      auto& mem_rsp = port.front();
      auto& entry = pending_reqs_.at(mem_rsp.tag);
      assert(entry.count);
      --entry.count; // track remaining blocks 
      if (0 == entry.count) {
        auto& mem_wr_addrs = entry.data->mem_wr_addrs;
        for (uint32_t i = 0, n = mem_wr_addrs.size(); i < n; ++i) {
          uint32_t j = i % simobject_->MemReqs.size();
          MemReq mem_req;
          mem_req.addr  = mem_wr_addrs.at(i).addr;
          mem_req.write = true;
          mem_req.tag   = 0;
          mem_req.core_id = mem_rsp.core_id;
          mem_req.uuid = mem_rsp.uuid;
          simobject_->MemReqs.at(j).send(mem_req, 2);
          ++perf_stats_.writes;
        }
        pending_reqs_.release(mem_rsp.tag);
      }   
      port.pop();
    }    

    for (int i = 0, n = pending_reqs_.size(); i < n; ++i) {
      if (pending_reqs_.contains(i))
        perf_stats_.latency += pending_reqs_.at(i).count;
    }

    // check input queue
    if (simobject_->Input.empty())
      return;

    // check pending queue capacity
    auto data = simobject_->Input.front();
    if (!data->mem_rd_addrs.empty()) {
      if (pending_reqs_.full())
        return;
      auto tag = pending_reqs_.allocate({data, (uint32_t)data->mem_rd_addrs.size()});
      for (uint32_t i = 0, n = data->mem_rd_addrs.size(); i < n; ++i) {
        uint32_t j = i % simobject_->MemReqs.size();
        MemReq mem_req;
        mem_req.addr  = data->mem_rd_addrs.at(i).addr;
        mem_req.write = false;
        mem_req.tag   = tag;
        mem_req.core_id = data->core_id;
        mem_req.uuid = data->uuid;
        simobject_->MemReqs.at(j).send(mem_req, 1);
        ++perf_stats_.reads;
      }
    } else {
      for (uint32_t i = 0, n = data->mem_wr_addrs.size(); i < n; ++i) {
        uint32_t j = i % simobject_->MemReqs.size();
        MemReq mem_req;
        mem_req.addr  = data->mem_wr_addrs.at(i).addr;
        mem_req.write = true;
        mem_req.tag   = 0;
        mem_req.core_id = data->core_id;
        mem_req.uuid = data->uuid;
        simobject_->MemReqs.at(j).send(mem_req, 1);
        ++perf_stats_.writes;
      }
    }
    auto time = simobject_->Input.pop();
    perf_stats_.stalls += (SimPlatform::instance().cycles() - time);
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
  , MemReqs(arch.num_threads(), this)
  , MemRsps(arch.num_threads(), this)
  , Input(this)
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

void RopUnit::write(uint32_t x, uint32_t y, bool is_backface, uint32_t color, uint32_t depth, TraceData::Ptr trace_data) {
  impl_->write(x, y, is_backface, color, depth, trace_data);
}

void RopUnit::tick() {
  impl_->tick();
}

const RopUnit::PerfStats& RopUnit::perf_stats() const {
  return impl_->perf_stats();
}