
#pragma once

#include <memory>
#include <iostream>
#include <util.h>
#include "types.h"
#include "archdef.h"
#include "debug.h"

namespace vortex {

struct pipeline_trace_t {
  //--
  uint64_t    uuid;
  
  //--
  uint32_t    cid;
  uint32_t    wid;  
  ThreadMask  tmask;
  Word        PC;

  //--
  bool        fetch_stall;

  //--
  bool        wb;  
  RegType     rdest_type;
  uint32_t    rdest;

  //--
  RegMask     used_iregs;
  RegMask     used_fregs;
  RegMask     used_vregs;

  //- 
  ExeType     exe_type; 

  //--
  std::vector<std::vector<mem_addr_size_t>> mem_addrs;
  
  //--
  union {
    struct {        
      LsuType type;
    } lsu;
    struct {
      AluType type;
    } alu;
    struct {
      FpuType type;
    } fpu;
    struct {
      GpuType type;
      WarpMask active_warps;
    } gpu;
  };

  bool stalled;

  pipeline_trace_t(uint64_t uuid_, const ArchDef& arch) {
    uuid = uuid_;
    cid = 0;
    wid = 0;
    tmask.reset();
    PC = 0;
    fetch_stall = false;
    wb  = false;
    rdest = 0;
    rdest_type = RegType::None;
    used_iregs.reset();
    used_fregs.reset();
    used_vregs.reset();
    exe_type = ExeType::NOP;
    mem_addrs.resize(arch.num_threads());
    stalled = false;
  }

  bool suspend() {
    bool old = stalled;
    stalled = true;
    return old;
  }

  void resume() {
    stalled = false;
  }
};

inline std::ostream &operator<<(std::ostream &os, const pipeline_trace_t& state) {
  os << "coreid=" << state.cid << ", wid=" << state.wid << ", PC=" << std::hex << state.PC;
  os << ", wb=" << state.wb;
  if (state.wb) {
     os << ", rd=" << state.rdest_type << std::dec << state.rdest;
  }
  os << ", ex=" << state.exe_type;
  os << " (#" << std::dec << state.uuid << ")";
  return os;
}

class PipelineLatch {
protected:
  const char* name_;
  std::queue<pipeline_trace_t*> queue_;

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
};

}