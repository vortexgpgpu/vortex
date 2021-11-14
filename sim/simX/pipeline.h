
#pragma once

#include <memory>
#include <iostream>
#include <util.h>
#include "types.h"
#include "debug.h"

namespace vortex {

struct pipeline_state_t {
  //--
  uint64_t    id;
  
  //--
  int         cid;
  int         wid;  
  ThreadMask  tmask;
  Word        PC;

  //--
  bool        stall_warp;
  bool        wb;  
  RegType     rdest_type;
  int         rdest;
  RegMask     used_iregs;
  RegMask     used_fregs;
  RegMask     used_vregs;

  //- 
  ExeType     exe_type; 
  std::vector<uint64_t> mem_addrs;
  
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
    } gpu;
  };

  // stats
  uint64_t icache_latency;
  uint64_t dcache_latency;

  void clear() {
    cid = 0;
    wid = 0;
    tmask.reset();
    PC = 0;
    stall_warp = false;
    wb = false;
    rdest = 0;
    rdest_type = RegType::None;
    used_iregs.reset();
    used_fregs.reset();
    used_vregs.reset();
    exe_type = ExeType::NOP;
    mem_addrs.clear();    
    icache_latency = 0;
    dcache_latency = 0;
  }
};

inline std::ostream &operator<<(std::ostream &os, const pipeline_state_t& state) {
  os << "coreid=" << state.cid << ", wid=" << state.wid << ", PC=" << std::hex << state.PC;
  os << ", wb=" << state.wb;
  if (state.wb) {
     os << ", rd=" << state.rdest_type << std::dec << state.rdest;
  }
  os << ", ex=" << state.exe_type;
  os << " (#" << std::dec << state.id << ")";
  return os;
}

class PipelineStage : public Queue<pipeline_state_t> {
protected:
  const char* name_;
  friend std::ostream &operator<<(std::ostream &, const pipeline_state_t&);

public:
  PipelineStage(const char* name = nullptr) 
    : name_(name) 
  {}
};

}