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
  bool initialized_;

  void initialize() {
    // get device configuration
    buf_baseaddr_       = dcrs_.read(DCR_ROP_ZBUF_ADDR);
    buf_pitch_          = dcrs_.read(DCR_ROP_ZBUF_PITCH);
    
    depth_func_         = dcrs_.read(DCR_ROP_DEPTH_FUNC);
    depth_writemask_    = dcrs_.read(DCR_ROP_DEPTH_WRITEMASK) & 0x1;

    stencil_front_func_ = dcrs_.read(DCR_ROP_STENCIL_FUNC) & 0xffff;
    stencil_front_zpass_= dcrs_.read(DCR_ROP_STENCIL_ZPASS) & 0xffff;
    stencil_front_zfail_= dcrs_.read(DCR_ROP_STENCIL_ZFAIL) & 0xffff;
    stencil_front_fail_ = dcrs_.read(DCR_ROP_STENCIL_FAIL) & 0xffff;
    stencil_front_ref_  = dcrs_.read(DCR_ROP_STENCIL_REF) & 0xffff;
    stencil_front_mask_ = dcrs_.read(DCR_ROP_STENCIL_MASK) & 0xffff;    
    stencil_front_writemask_ = dcrs_.read(DCR_ROP_STENCIL_WRITEMASK) & 0xffff;

    stencil_back_func_  = dcrs_.read(DCR_ROP_STENCIL_FUNC) >> 16;
    stencil_back_zpass_ = dcrs_.read(DCR_ROP_STENCIL_ZPASS) >> 16;
    stencil_back_zfail_ = dcrs_.read(DCR_ROP_STENCIL_ZFAIL) >> 16;
    stencil_back_fail_  = dcrs_.read(DCR_ROP_STENCIL_FAIL) >> 16;    
    stencil_back_ref_   = dcrs_.read(DCR_ROP_STENCIL_REF) >> 16;
    stencil_back_mask_  = dcrs_.read(DCR_ROP_STENCIL_MASK) >> 16;
    stencil_back_writemask_ = dcrs_.read(DCR_ROP_STENCIL_WRITEMASK) >> 16;

    depth_enabled_      = !((depth_func_ == ROP_DEPTH_FUNC_ALWAYS) 
                         && !depth_writemask_);
    
    stencil_front_enabled_ = !((stencil_front_func_  == ROP_DEPTH_FUNC_ALWAYS) 
                            && (stencil_front_zpass_ == ROP_STENCIL_OP_KEEP)
                            && (stencil_front_zfail_ == ROP_STENCIL_OP_KEEP));
    
    stencil_back_enabled_ = !((stencil_back_func_  == ROP_DEPTH_FUNC_ALWAYS) 
                           && (stencil_back_zpass_ == ROP_STENCIL_OP_KEEP)
                           && (stencil_back_zfail_ == ROP_STENCIL_OP_KEEP));

    initialized_ = true;
  }

  bool doDepthStencilTest(uint32_t x, uint32_t y, uint32_t is_backface, uint32_t depth, RopUnit::TraceData::Ptr trace_data)  { 
    auto depth_ref     = depth & ROP_DEPTH_MASK;    
    auto stencil_func  = is_backface ? stencil_back_func_ : stencil_front_func_;    
    auto stencil_ref   = is_backface ? stencil_back_ref_ : stencil_front_ref_;    
    auto stencil_mask  = is_backface ? stencil_back_mask_ : stencil_front_mask_;
    auto stencil_writemask = is_backface ? stencil_back_writemask_ : stencil_front_writemask_;
    auto stencil_ref_m = stencil_ref & stencil_mask;

    uint32_t buf_addr = buf_baseaddr_ + y * buf_pitch_ + x * 4;

    uint32_t stored_value;          
    mem_->read(&stored_value, buf_addr, 4);     
    trace_data->ds_mem_addrs.push_back({buf_addr, 4});

    uint32_t stencil_val = stored_value >> ROP_DEPTH_BITS;
    uint32_t depth_val   = stored_value & ROP_DEPTH_MASK;   

    uint32_t stencil_val_m = stencil_val & stencil_mask;

    uint32_t writeMask = stencil_writemask << ROP_DEPTH_BITS;

    uint32_t stencil_op;

    auto passed = DoCompare(stencil_func, stencil_ref_m, stencil_val_m);
    if (passed) {
      passed = DoCompare(depth_func_, depth_ref, depth_val);
      if (passed) {
        writeMask |= (depth_writemask_ ? ROP_DEPTH_MASK : 0);
        stencil_op = is_backface ? stencil_back_zpass_ : stencil_front_zpass_;              
      } else {
        stencil_op = is_backface ? stencil_back_zfail_ : stencil_front_zfail_;
      } 
    } else {
      stencil_op = is_backface ? stencil_back_fail_ : stencil_front_fail_;
    }

    auto stencil_result = DoStencilOp(stencil_op, stencil_ref, stencil_val);

    // Write the depth stencil value
    DT(3, "rop-depthstencil: x=" << std::dec << x << ", y=" << y << ", depth_val=0x" << std::hex << depth_val << ", depth_ref=0x" << depth_ref << ", stencil_val=0x" << stencil_val << ", stencil_result=0x" << stencil_result << ", passed=" << passed);    
    if (writeMask) {
      auto merged_value = (stencil_result << ROP_DEPTH_BITS) | depth_ref;
      auto write_value = (stored_value & ~writeMask) | (merged_value & writeMask);
      mem_->write(&write_value, buf_addr, 4);
      trace_data->ds_write;
    }

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

  bool write(uint32_t x, uint32_t y, bool is_backface, uint32_t depth, RopUnit::TraceData::Ptr trace_data) {
    if (!initialized_) {
      this->initialize();
    }
    auto stencil_enabled = is_backface ? stencil_back_enabled_ : stencil_front_enabled_;
    if (stencil_enabled || depth_enabled_) {
      return this->doDepthStencilTest(x, y, is_backface, depth, trace_data);
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
  bool initialized_;

  void initialize() {
    // get device configuration
    buf_baseaddr_   = dcrs_.read(DCR_ROP_CBUF_ADDR);
    buf_pitch_      = dcrs_.read(DCR_ROP_CBUF_PITCH);
    buf_writemask_  = dcrs_.read(DCR_ROP_CBUF_WRITEMASK) & 0xf;

    blend_mode_rgb_ = dcrs_.read(DCR_ROP_BLEND_MODE) & 0xffff;
    blend_mode_a_   = dcrs_.read(DCR_ROP_BLEND_MODE) >> 16;
    blend_src_rgb_  = (dcrs_.read(DCR_ROP_BLEND_FUNC) >>  0) & 0xff;
    blend_src_a_    = (dcrs_.read(DCR_ROP_BLEND_FUNC) >>  8) & 0xff;
    blend_dst_rgb_  = (dcrs_.read(DCR_ROP_BLEND_FUNC) >> 16) & 0xff;
    blend_dst_a_    = (dcrs_.read(DCR_ROP_BLEND_FUNC) >> 24) & 0xff;
    blend_const_    = dcrs_.read(DCR_ROP_BLEND_CONST);
    logic_op_       = dcrs_.read(DCR_ROP_LOGIC_OP);  

    blend_enabled_  = !((blend_mode_rgb_ == ROP_BLEND_MODE_ADD)
                     && (blend_mode_a_   == ROP_BLEND_MODE_ADD) 
                     && (blend_src_rgb_  == ROP_BLEND_FUNC_ONE) 
                     && (blend_src_a_    == ROP_BLEND_FUNC_ONE) 
                     && (blend_dst_rgb_  == ROP_BLEND_FUNC_ZERO) 
                     && (blend_dst_a_    == ROP_BLEND_FUNC_ZERO));

    initialized_ = true;
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

  void write(uint32_t x, uint32_t y, uint32_t color, RopUnit::TraceData::Ptr trace_data) {
    if (!initialized_) {
      this->initialize();
    }

    if (0 == buf_writemask_)
      return;

    uint32_t buf_addr = buf_baseaddr_ + y * buf_pitch_ + x * 4;

    uint32_t stored_value = 0;
    mem_->read(&stored_value, buf_addr, 4);
    trace_data->color_mem_addrs.push_back({buf_addr, 4});

    cocogfx::ColorARGB src(color);
    cocogfx::ColorARGB dst(stored_value);
    cocogfx::ColorARGB cst(blend_const_);

    cocogfx::ColorARGB result_color;
    if (blend_enabled_) {
      result_color = this->doBlend(src, dst, cst);
    } else {
      result_color = src;
    }

    DT(3, "rop-color: x=" << std::dec << x << ", y=" << y << std::hex << ", color=0x" << result_color);
    uint32_t writemask = (((buf_writemask_ >> 0) & 0x1) * 0x000000ff) 
                       | (((buf_writemask_ >> 1) & 0x1) * 0x0000ff00) 
                       | (((buf_writemask_ >> 2) & 0x1) * 0x00ff0000) 
                       | (((buf_writemask_ >> 3) & 0x1) * 0xff000000);
    if (writemask) {
      auto write_value = (stored_value & ~writemask) | (result_color & writemask);
      mem_->write(&write_value, buf_addr, 4);
      trace_data->color_write;
    }
  }
};

class RopSlices {
public:
  RopSlices(const Arch& arch, const RopUnit::DCRS& dcrs) 
    : depthtencils_(ROP_NUM_SLICES, {arch, dcrs})
    , blenders_(ROP_NUM_SLICES, {arch, dcrs})
    , slice_idx_(0)
  {}

  void clear() {
    for (int i = 0; i < ROP_NUM_SLICES; ++i) {
      depthtencils_.at(i).clear();
      blenders_.at(i).clear();
    }
  }

  void attach_ram(RAM* mem) {
    for (int i = 0; i < ROP_NUM_SLICES; ++i) {
      depthtencils_.at(i).attach_ram(mem);
      blenders_.at(i).attach_ram(mem);
    }
  }   

  void write(uint32_t x, uint32_t y, bool is_backface, uint32_t color, uint32_t depth, RopUnit::TraceData::Ptr trace_data) {      
    if (depthtencils_.at(slice_idx_).write(x, y, is_backface, depth, trace_data)) {
      blenders_.at(slice_idx_).write(x, y, color, trace_data);
    }
    slice_idx_ = (slice_idx_ + 1) % ROP_NUM_SLICES;    
  }

private:
  std::vector<DepthTencil> depthtencils_;
  std::vector<Blender>     blenders_;
  uint32_t                 slice_idx_;
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
        if (entry.data->ds_write) {
          for (uint32_t i = 0, n = entry.data->ds_mem_addrs.size(); i < n; ++i) {
            MemReq mem_req;
            mem_req.addr  = entry.data->ds_mem_addrs.at(i).addr;
            mem_req.write = true;
            mem_req.tag   = mem_rsp.tag;
            mem_req.core_id = mem_rsp.core_id;
            mem_req.uuid = mem_rsp.uuid;
            simobject_->MemReqs.at(i).send(mem_req, 2);
            ++perf_stats_.writes;
          }
        }
        if (entry.data->color_write) {
          for (uint32_t i = 0, n = entry.data->color_mem_addrs.size(); i < n; ++i) {
            MemReq mem_req;
            mem_req.addr  = entry.data->color_mem_addrs.at(i).addr;
            mem_req.write = true;
            mem_req.tag   = mem_rsp.tag;
            mem_req.core_id = mem_rsp.core_id;
            mem_req.uuid = mem_rsp.uuid;
            simobject_->MemReqs.at(i).send(mem_req, 2);
            ++perf_stats_.writes;
          }
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
    if (pending_reqs_.full())          
      return;

    auto data = simobject_->Input.front();

    uint32_t num_addrs = data->ds_mem_addrs.size() + data->color_mem_addrs.size();

    auto tag = pending_reqs_.allocate({data, num_addrs});

    for (uint32_t i = 0, n = data->ds_mem_addrs.size(); i < n; ++i) {
      MemReq mem_req;
      mem_req.addr  = data->ds_mem_addrs.at(i).addr;
      mem_req.write = false;
      mem_req.tag   = tag;
      mem_req.core_id = data->core_id;
      mem_req.uuid = data->uuid;
      simobject_->MemReqs.at(i).send(mem_req, 1);
      ++perf_stats_.reads;
    }

    for (uint32_t i = 0, n = data->color_mem_addrs.size(); i < n; ++i) {
      MemReq mem_req;
      mem_req.addr  = data->color_mem_addrs.at(i).addr;
      mem_req.write = false;
      mem_req.tag   = tag;
      mem_req.core_id = data->core_id;
      mem_req.uuid = data->uuid;
      simobject_->MemReqs.at(i).send(mem_req, 1);
      ++perf_stats_.reads;        
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