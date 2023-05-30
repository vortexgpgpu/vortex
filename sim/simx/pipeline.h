
#pragma once

#include <memory>
#include <iostream>
#include <util.h>
#include "types.h"
#include "arch.h"
#include "debug.h"

namespace vortex {

class ITraceData {
public:
    using Ptr = std::shared_ptr<ITraceData>;
    ITraceData() {}
    virtual ~ITraceData() {}
};

struct LsuTraceData : public ITraceData {
  using Ptr = std::shared_ptr<LsuTraceData>;
  std::vector<mem_addr_size_t> mem_addrs;
  LsuTraceData(uint32_t num_threads) : mem_addrs(num_threads) {}
};

struct GPUTraceData : public ITraceData {
  using Ptr = std::shared_ptr<GPUTraceData>;
  union {
    const WarpMask active_warps;
    struct {
      uint32_t bar_id;
      uint32_t bar_count;
    };
  };
  GPUTraceData(const WarpMask& active_warps) : active_warps(active_warps) {}
  GPUTraceData(uint32_t bar_id, uint32_t bar_count) : bar_id(bar_id), bar_count(bar_count) {}
};

struct pipeline_trace_t {
public:
  //--
  const uint64_t  uuid;
  
  //--
  uint32_t    cid;
  uint32_t    wid;  
  ThreadMask  tmask;
  Word        PC;

  //--
  uint32_t    rdest;
  RegType     rdest_type;
  bool        wb;

  //--
  RegMask     used_iregs;
  RegMask     used_fregs;
  RegMask     used_vregs;

  //- 
  ExeType     exe_type; 

  //--
  union {
    uint32_t unit_type;
    LsuType  lsu_type;
    AluType  alu_type;
    FpuType  fpu_type;
    GpuType  gpu_type;
  };

  ITraceData::Ptr data;

  bool fetch_stall;

  pipeline_trace_t(uint64_t uuid) 
    : uuid(uuid)
    , cid(0)
    , wid(0)
    , PC(0)    
    , rdest(0)
    , rdest_type(RegType::None)
    , wb(false)
    , exe_type(ExeType::ALU)
    , unit_type(0)
    , data(nullptr)
    , fetch_stall(false)
    , log_once_(false) 
  {}
  
  ~pipeline_trace_t() {}

  bool log_once(bool enable) {
    bool old = log_once_;
    log_once_ = enable;
    return old;
  }

private:
  bool log_once_;
};

inline std::ostream &operator<<(std::ostream &os, const pipeline_trace_t& state) {
  os << "cid=" << state.cid << ", wid=" << state.wid << ", PC=" << std::hex << state.PC;
  os << ", wb=" << state.wb;
  if (state.wb) {
     os << ", rd=" << state.rdest_type << std::dec << state.rdest;
  }
  os << ", ex=" << state.exe_type;
  os << " (#" << std::dec << state.uuid << ")";
  return os;
}

class PipelineLatch {
public:
  PipelineLatch(const char* name = nullptr) 
    : name_(name) 
  {}
  
  bool empty() const {
    return queue_.empty();
  }

  pipeline_trace_t* front() {
    return queue_.front();
  }

  pipeline_trace_t* back() {
    return queue_.back();
  }

  void push(pipeline_trace_t* value) {    
    queue_.push(value);
  }

  void pop() {
    queue_.pop();
  }

  void clear() {
    std::queue<pipeline_trace_t*> empty;
    std::swap(queue_, empty );
  }

protected:
  const char* name_;
  std::queue<pipeline_trace_t*> queue_;
};

}